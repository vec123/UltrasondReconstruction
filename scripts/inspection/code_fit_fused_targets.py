"""Fit only the code of a frozen SSM-trained flow to real fused fetal targets, with chamfer.

The counterpart of code_fit_partial_tests.py on real data: each case's fused target
(<fused-dir>/<case>.vtp, mm, in the frame of the aligned cohort template) is carried into
the shape model's frame by the exact similarity between the model mean and the aligned
template (they are the same mesh, vertex for vertex), and the code of the run's frozen
flow is fitted to it per variant with code_fit_partial_tests.fit_codes (chamfer: scan ->
fit quadratic, fit -> scan truncated, plus the Gaussian prior of the training codes).
There is no ground truth, so the numbers are distances to the scan:

  * scan -> fit: each scan point to the nearest fit vertex (mean, support-weighted, p95);
  * fit -> scan: each fit vertex to the nearest scan point, weighted by the template's
    landmark_weight (the face);
  * the code's squared Mahalanobis distance under the training codes (training: about n_z).

Writes <output>/summary.json and <output>/<case>.png: the scan as a surface, each code fit
and the reference fits of the same case (pair LDDMM, cohort svf over all 110 targets),
from the front, left and right.

Run with:
    python scripts/inspection/code_fit_fused_targets.py \\
        --run outputs/baby_ssm_cohort/resnet_lddmm_cohort_autodecoder_sobol1000_global_l2_mlp6x256_kin200 \\
        --cases 16_26 13_26 11_36
"""

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial import cKDTree
from vtk.util.numpy_support import vtk_to_numpy

from scripts.inspection import render_frontal_views as views  # noqa: E402
import scripts.inspection.render_ssm_cohort_views  # noqa: E402,F401  (RGB panels)
from scripts.inspection.code_fit_partial_tests import (ERROR_MAX_MM, code_parametrization, fit_codes, flow_decoder,  # noqa: E402
                                                       load_run_config, ssm_decoder)
from src.resnet_lddmm.resnet_flow.losses.fine_tune import FieldChangeKinetic  # noqa: E402
from src.resnet_lddmm.resnet_flow.runner import _normalization_domain  # noqa: E402
from scripts.inspection.render_ssm_posterior_views import ramp_colors, with_colors  # noqa: E402
from scripts.inspection.eval_ssm_cohort import load_run  # noqa: E402
from scripts.SSM.target_to_ssm_frame import similarity  # noqa: E402
from src.SSM.model.model_memory_io import load_model_file  # noqa: E402
from src.shared.vtk.create import create_polydata  # noqa: E402
from src.shared.vtk.io import load_vtp  # noqa: E402

CASOS = "outputs/Casos_Mar"
# All 110 fused targets (not the 30-pair cherry-tree subset), so every case has a fit.
COHORT_SVF = "resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g8c2_dist_svf"


def references(case):
    """{label: deformed.vtp} of the other methods' fits of `case` that exist (mm, same frame)."""
    number, week = case.split("_")
    out = {"pair LDDMM": f"{CASOS}/resnet_lddmm_baby_mean/{number}/{week} semanas/apply/world/deformed.vtp"}
    out["cohort svf"] = f"{CASOS}/{COHORT_SVF}/apply/{case}_semanas/world/deformed.vtp"
    return {label: path for label, path in out.items() if os.path.exists(path)}


class FreeNorms:
    """The flow's LayerNorm affines, fitted with one case's code (--free-norm), for fit_codes.

    penalty = data_weight * (field_change_ratio * FC + anchor_ratio * sum (p - p0)^2), with
    FC the field_change_kinetic term (1/2 int mean ||v - v0||^2 dt) on --fit-points template
    vertices, in mm^2, and data_weight = n_obs / (2 sigma^2) of the variant. The ratios are
    a training's prices over its data weight; the defaults are those of the cherry30 fit
    (field change 200, anchor 0.01, against 1 / (2 * 0.01^2) = 5000), so the data term and
    the two penalties keep that fit's balance whatever the variant's sigma.
    """

    def __init__(self, flow, template, mm_per_unit, to_code, args, device):
        self.flow, self.template, self.args, self.device = flow, template, args, device
        self.to_code = to_code                 # fitted parameters theta -> the flow's code z
        self.mm2 = mm_per_unit ** 2
        self.params = [p for name, p in flow.named_parameters() if "norm" in name]
        if not self.params:
            raise ValueError("--free-norm: the flow has no LayerNorm (field.norm)")
        for p in self.params:
            p.requires_grad_(True)
        self.initial = [p.detach().clone() for p in self.params]
        self.lr = args.norm_lr
        self.change = FieldChangeKinetic()     # its reference: the field at the first call, i.e. p0
        self.states = {}

    def reset(self):
        with torch.no_grad():
            for p, p0 in zip(self.params, self.initial):
                p.copy_(p0)

    def penalty(self, theta, data_weight, gen):
        z = self.to_code(None, theta)
        idx = torch.randint(len(self.template), (self.args.fit_points,), device=self.device, generator=gen)
        traj = self.flow(self.template[idx][None].expand(len(z), -1, -1), z)
        change = self.change(SimpleNamespace(fwd_traj=traj, flow=self.flow, code=z)) * self.mm2
        anchor = sum(((p - p0) ** 2).sum() for p, p0 in zip(self.params, self.initial))
        return data_weight * (self.args.field_change_ratio * change + self.args.anchor_ratio * anchor)

    def save(self, key):
        self.states[key] = [p.detach().clone() for p in self.params]

    def load(self, state):
        with torch.no_grad():
            for p, value in zip(self.params, state):
                p.copy_(value)

    def rms_change(self, state):
        return float(torch.sqrt(sum(((s - p0) ** 2).sum() for s, p0 in zip(state, self.initial))
                                / sum(p0.numel() for p0 in self.initial)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="the SSM-trained cohort run")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--cases", nargs="+", default=["16_26", "13_26", "11_36"])
    parser.add_argument("--fused-dir", default=f"{CASOS}/cohort/fused_targets")
    parser.add_argument("--aligned-template", default=f"{CASOS}/cohort/template_baby_mean_aligned.vtp")
    parser.add_argument("--model", default="assets/shape_models/baby/baby_pca.npz")
    parser.add_argument("--variants", nargs="+", default=["data", "gauss1", "gauss3", "gauss10", "gauss30"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--points", type=int, default=2000, help="scan points per code and step")
    parser.add_argument("--fit-points", type=int, default=6000, help="template vertices flowed per step")
    parser.add_argument("--trunc-mm", type=float, default=2.0, help="cap of the fit -> scan distance")
    parser.add_argument("--no-ssm", dest="ssm", action="store_false",
                        help="skip the SSM fit under the same energy (default: fit it too)")
    parser.add_argument("--ssm-lr", type=float, default=0.05, help="Adam step size of the SSM coefficients (sigma units)")
    parser.add_argument("--output", default=None,
                        help="default: outputs/posterior_SSM_tests/fused_targets/<run name>")
    parser.add_argument("--panel-px", type=int, default=260)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data", choices=("chamfer", "varifold"), default="chamfer")
    parser.add_argument("--fit-faces", type=int, default=2000, help="varifold: fixed triangle subset per surface")
    parser.add_argument("--varifold-sigmas", type=float, nargs="+", default=[3.0, 6.0, 12.0],
                        help="varifold: kernel widths (mm)")
    parser.add_argument("--lr-decay", action="store_true", help="cosine decay of the step size to 0 over --steps")
    parser.add_argument("--code-prior", choices=("gauss", "pca", "pca_residual"), default="gauss",
                        help="code parametrisation and prior, see code_parametrization")
    parser.add_argument("--pca-var", type=float, default=0.99, help="pca*: variance share of the kept components")
    parser.add_argument("--theta-lr", type=float, default=0.05, help="pca*: Adam step size (parameters in sigma units)")
    parser.add_argument("--residual-w", type=float, nargs=2, default=[1.0, 100.0],
                        help="pca_residual: residual weight near the face and far from it")
    parser.add_argument("--residual-near", type=float, default=0.5,
                        help="pca_residual: grid cells from the template up to which a node gets the low weight")
    parser.add_argument("--residual-far", type=float, default=1.0,
                        help="pca_residual: grid cells from the template from which a node gets the high weight")
    parser.add_argument("--keops", action="store_true",
                        help="varifold with KeOps: all scan triangles, --fit-faces fixed template triangles")
    parser.add_argument("--prior-run", default=None,
                        help="take the prior's training codes from this run's final.pt (same n_z), not from --run")
    parser.add_argument("--free-norm", action="store_true",
                        help="also fit the flow's LayerNorm affines, per case (see FreeNorms)")
    parser.add_argument("--norm-lr", type=float, default=1e-4, help="--free-norm: Adam step size of the affines")
    parser.add_argument("--field-change-ratio", type=float, default=200.0 / 5000.0,
                        help="--free-norm: field change price over the data weight")
    parser.add_argument("--anchor-ratio", type=float, default=0.01 / 5000.0,
                        help="--free-norm: parameter anchor price over the data weight")
    args = parser.parse_args()
    device = torch.device(args.device)
    out_dir = args.output or os.path.join("outputs/posterior_SSM_tests/fused_targets",
                                          os.path.basename(os.path.normpath(args.run))
                                          + ("" if args.data == "chamfer" else f"_{args.data}")
                                          + ("" if args.code_prior == "gauss" else f"_{args.code_prior}")
                                          + (f"_prior-{os.path.basename(os.path.normpath(args.prior_run))}"
                                             if args.prior_run else "")
                                          + (f"_layernorm-per-case_fieldchange{args.field_change_ratio:g}"
                                             f"_anchor{args.anchor_ratio:g}_lr{args.norm_lr:g}" if args.free_norm else ""))
    os.makedirs(out_dir, exist_ok=True)

    model = load_model_file(args.model)
    faces = model.faces
    aligned = load_vtp(args.aligned_template)
    aligned_pts = vtk_to_numpy(aligned.GetPoints().GetData()).astype(np.float64)
    face_w = vtk_to_numpy(aligned.GetPointData().GetArray("landmark_weight")).astype(np.float64)
    s, rot, trans = similarity(model.mean.reshape(-1, 3), aligned_pts)   # aligned = s R mean + t

    _, flow, train_codes, _, template_norm, transform = load_run(args.run, args.step, device)
    if args.prior_run:
        # E.g. the codes of the samples a flow was trained on, for a flow adapted later (same code space).
        prior_codes = torch.load(os.path.join(args.prior_run, "checkpoints", "final.pt"), map_location="cpu",
                                 weights_only=False)["codes"]["codes.weight"].float()
        if prior_codes.shape[1] != train_codes.shape[1]:
            raise ValueError(f"--prior-run codes have n_z {prior_codes.shape[1]}, the run's {train_codes.shape[1]}")
        print(f"prior codes: {len(prior_codes)} from {args.prior_run} (not the run's {len(train_codes)})")
        train_codes = prior_codes
    for p in flow.parameters():
        p.requires_grad_(False)
    z_train = train_codes.double().numpy()
    train_cfg = load_run_config(args.run)
    wrap, mu, s_inv, prior_label = code_parametrization(args.code_prior, train_cfg, train_codes, template_norm,
                                                        _normalization_domain(train_cfg), args, device)
    print(f"code prior: {prior_label}")
    center, scale = transform.center.to(device), transform.scale.to(device)
    template = template_norm.points[0].to(device)

    scans, targets, candidates = {}, [], []
    for case in args.cases:
        scan = load_vtp(os.path.join(args.fused_dir, f"{case}_semanas.vtp"))
        pts = vtk_to_numpy(scan.GetPoints().GetData()).astype(np.float64)
        scans[case] = (scan, pts, vtk_to_numpy(scan.GetPointData().GetArray("support")).astype(np.float64))
        # mm, unposed: the model frame scaled by s (x = s R a + t  ->  s a = R^T (x - t)).
        targets.append(torch.as_tensor((pts - trans) @ rot, dtype=torch.float32, device=device))
        candidates.append(torch.arange(len(pts), device=device))
    # Varifold: the scan's own triangles, in the same unposed mm frame as its points.
    vf_inputs = {}
    if args.data == "varifold":
        from scripts.inspection.render_ssm_posterior_views import faces_of
        obs_tris = []
        for case, target in zip(args.cases, targets):
            obs_tris.append(target[torch.as_tensor(faces_of(scans[case][0]), dtype=torch.long, device=device)])
        vf_inputs = {"template_mm": torch.as_tensor(model.mean.reshape(-1, 3) * s, dtype=torch.float32, device=device),
                     "template_faces": torch.as_tensor(faces, dtype=torch.long, device=device),
                     "obs_triangles": obs_tris}
    # The flow's code, and (--ssm) the SSM's coefficients under the same energy and data term.
    n_vert = len(template)
    code_label = "code + LN fit" if args.free_norm else "code fit"
    methods = {code_label: (wrap(flow_decoder(flow, template, center, scale, s)), mu, s_inv,
                            args.lr if args.code_prior == "gauss" else args.theta_lr)}
    free = (FreeNorms(flow, template, float(scale) * s, wrap(lambda idx, code: code), args, device)
            if args.free_norm else None)
    norm_change = {}                                                     # (case, variant) -> rms change of the affines
    if args.ssm:
        k = model.n_components
        methods["SSM fit"] = (ssm_decoder(model, s, device), torch.zeros(k, device=device),
                              torch.eye(k, device=device), args.ssm_lr)
    fits = {}                                                            # (method, variant) -> [C, N, 3] mm (posed), d2 [C]
    for method, (decoder, m, si, lr) in methods.items():
        print(f"== {method}")
        if method == code_label and free is not None:
            # One case at a time: each case gets its own affines, from the run's.
            per_case = []
            for ci, case in enumerate(args.cases):
                print(f"-- {case}")
                codes = fit_codes(decoder, n_vert, targets[ci:ci + 1], candidates[ci:ci + 1], m, si, args, device,
                                  lr=lr, free=free, **{k: (v[ci:ci + 1] if k == "obs_triangles" else v)
                                                       for k, v in vf_inputs.items()})
                per_case.append((codes, dict(free.states)))
            with torch.no_grad():
                full = torch.arange(n_vert, device=device)[None]
                for v in args.variants:
                    shapes_v, d2_v = [], []
                    for ci, (codes, states) in enumerate(per_case):
                        free.load(states[v])
                        norm_change[(args.cases[ci], v)] = free.rms_change(states[v])
                        shapes_v.append(decoder(full, codes[v]).double().cpu().numpy()[0])
                        d2_v.append(float(torch.einsum("ji,ik,jk->j", codes[v] - m, si, codes[v] - m)[0]))
                    fits[(method, v)] = (np.stack(shapes_v) @ rot.T + trans, np.array(d2_v))
            free.reset()
            continue
        codes = fit_codes(decoder, n_vert, targets, candidates, m, si, args, device, lr=lr, **vf_inputs)
        with torch.no_grad():
            for v, z in codes.items():
                full = torch.arange(n_vert, device=device)[None].expand(len(args.cases), -1)
                unposed = decoder(full, z).double().cpu().numpy()       # s * model frame
                fits[(method, v)] = (unposed @ rot.T + trans,
                                     torch.einsum("ji,ik,jk->j", z - m, si, z - m).cpu().numpy())

    summary = {"run": args.run, "step": args.step, "prior_run": args.prior_run, "variants": args.variants,
               "steps": args.steps,
               "trunc_mm": args.trunc_mm, "train_code_d2_mean": float(z_train.shape[1]), "cases": {}}
    if free is not None:
        summary["free_norm"] = {"per_case": True, "lr": args.norm_lr, "field_change_ratio": args.field_change_ratio,
                                "anchor_ratio": args.anchor_ratio, "values": sum(p.numel() for p in free.params)}
    names, lmk_idx = model.landmark_names(), model.landmark_indices()
    for ci, case in enumerate(args.cases):
        scan, pts, support = scans[case]
        tree = cKDTree(pts)

        def distances(fit):
            d_scan, _ = cKDTree(fit).query(pts)
            d_fit, _ = tree.query(fit)
            return {"scan_to_fit_mm_mean": float(d_scan.mean()),
                    "scan_to_fit_mm_support_weighted": float((d_scan * support).sum() / support.sum()),
                    "scan_to_fit_mm_p95": float(np.percentile(d_scan, 95)),
                    "fit_to_scan_mm_face_weighted": float((d_fit * face_w).sum() / face_w.sum())}, d_fit

        entry, shapes = {}, {}                                           # label -> (poly, fit -> scan distance)
        for (method, v), (pts_all, d2) in fits.items():
            fit = pts_all[ci]
            metrics, d_fit = distances(fit)
            entry.setdefault(method, {})[v] = {**metrics, "mahalanobis_sq": float(d2[ci])}
            if (case, v) in norm_change:
                entry[method][v]["layernorm_rms_change"] = norm_change[(case, v)]
            shapes[f"{method} {v}"] = (create_polydata(fit, faces), d_fit,
                                       f"{metrics['scan_to_fit_mm_mean']:.2f} mm, d2 {d2[ci]:.0f}")
        for label, path in references(case).items():
            poly = load_vtp(path)
            metrics, d_fit = distances(vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64))
            entry[label] = {**metrics, "path": path}
            shapes[label] = (poly, d_fit, f"{metrics['scan_to_fit_mm_mean']:.2f} mm")
        summary["cases"][case] = entry
        print(f"{case}:")
        for key, r in entry.items():
            rows_ = r.items() if key in methods else [("", r)]
            print(f"    {key:<14}" + "  ".join(
                f"{v} {m['scan_to_fit_mm_mean']:.2f}/{m['scan_to_fit_mm_p95']:.2f}/{m['fit_to_scan_mm_face_weighted']:.2f}"
                + (f" d2 {m['mahalanobis_sq']:.0f}" if "mahalanobis_sq" in m else "") for v, m in rows_))

        # As code_fit_partial_tests.py's figures: frontal, left and right rows, then a frontal row of
        # each shape coloured by its vertices' distance to the nearest scan point (no ground truth here).
        lmk = {n: aligned_pts[i] for n, i in zip(names, lmk_idx)}
        panels = [(f"scan {case}", scan, views.SCAN_COLOR)]
        panels += [(f"{label} ({info})", poly, views.MESH_COLOR) for label, (poly, _, info) in shapes.items()]
        cams = {view: views.view_camera(lmk, [p for _, p, _ in panels], view) for view in ("frontal", "left", "right")}
        rows = [(cams[view], [(t + ("" if view == "frontal" else f" ({view})"), p, c, []) for t, p, c in panels])
                for view in cams]
        top = float(np.percentile(support, 95))
        err_row = [(f"scan confidence 0 -> {top:.1f}", with_colors(scan, ramp_colors(support, top)), None, [])]
        err_row += [(f"{label}: fit -> scan", with_colors(poly, ramp_colors(d_fit, ERROR_MAX_MM)), None, [])
                    for label, (poly, d_fit, _) in shapes.items()]
        rows.append((cams["frontal"], err_row))
        header = (f"{case}: {args.data} fits of the " + ("code and LayerNorm affines" if free is not None
                                                          else "frozen flow's code")
                  + f" ({os.path.basename(os.path.normpath(args.run))}) "
                  f"and of the SSM coefficients, and reference fits (title: mean scan -> fit, d2; training code d2 ~ "
                  f"{z_train.shape[1]}). Last row: distance to the nearest scan point, light 0 -> dark {ERROR_MAX_MM:g} mm")
        views.write_grid(rows, len(panels), header, os.path.join(out_dir, f"{case}.png"), args.panel_px)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
