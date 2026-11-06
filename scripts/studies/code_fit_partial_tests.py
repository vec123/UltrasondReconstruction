"""Fit only the code of a frozen SSM-trained flow to the partial observations of posterior_ssm_tests.py.

posterior_ssm_tests.py draws ground truths from the shape model, poses them randomly,
adds noise and observes only part of each (face_center, right_half, lower_face,
upper_face, sparse_face), then fits the SSM posterior. This script takes the SAME tests
(re-drawn with the same seed and checked against the saved ground_truth.vtp /
observed.vtp) and fits instead the code z of a cohort auto-decoder trained on samples of
that model, with its flow frozen, by minimising

    E(z) = sum_obs |phi_z(T)_i - y_i|^2 / (2 sigma^2)  +  w * 1/2 (z - mu)^T S^-1 (z - mu)

over the observed vertices i (correspondence is known: the observations are the
template's vertices), with mu, S the mean and covariance of the run's training codes.
The variants (--variants) set sigma (mm) and the prior weight w: "data" is the data term
alone (w = 0), "gauss<sigma>" the MAP estimate under the Gaussian code prior (w = 1), the
counterpart of the posterior fit with the same sigma.

The fit is given the TRUE pose (the posterior estimates its own): the observations are
mapped back into the model frame with it, and every fit is posed with it again for the
comparison with the ground truth.

Writes <output>/summary.json (per test and variant: median error on the observed and the
free vertices, 95th percentile on the free ones, the code's squared Mahalanobis distance;
the posterior's numbers next to them) and <output>/<test>.png: ground truth | observations
| each code fit | the posterior with the largest sigma, from the front, left and right, plus
a frontal row of error maps.

Run with:
    python scripts/studies/code_fit_partial_tests.py \\
        --run outputs/baby_ssm_cohort/resnet_lddmm_cohort_autodecoder_sobol1000_global_l2_mlp6x256_kin200
"""

import argparse
import json
import os
import sys

# KeOps compiles its CUDA kernels against the CUDA headers and libraries that PyTorch's pip
# packages ship, linked under the standard names into <env>/cuda_keops (include/, lib/).
_KEOPS_CUDA = os.path.join(sys.prefix, "cuda_keops")
if os.path.isdir(_KEOPS_CUDA):
    os.environ.setdefault("CUDA_PATH", _KEOPS_CUDA)
    os.environ["LD_LIBRARY_PATH"] = os.path.join(_KEOPS_CUDA, "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

import numpy as np
import torch
from vtk.util.numpy_support import vtk_to_numpy

from src.shared.vtk import render as views  # noqa: E402
from src.models.gpmm.shape_model import GpmmShapeModel  # noqa: E402
from src.models.lddmm.resnet_flow.apply.trained_run import flow_points, load_run, load_run_config  # noqa: E402
from src.models.lddmm.shape_model import build_code_prior  # noqa: E402
from src.shared.fitting.code_fit import fit_codes as shared_fit_codes, parse_variant  # noqa: E402
from src.models.lddmm.resnet_flow.runner import _normalization_domain  # noqa: E402
from scripts.studies.posterior_ssm_tests import MM_PER_UNIT, TESTS, random_pose, region  # noqa: E402
from src.shared.vtk.process import sub_surface  # noqa: E402
from src.shared.vtk.render import ramp_colors, with_colors  # noqa: E402
from src.models.gpmm.model.model_memory_io import load_model_file  # noqa: E402
from src.shared.vtk.create import create_polydata  # noqa: E402
from src.shared.vtk.io import load_vtp  # noqa: E402

ERROR_MAX_MM = 3.0


def redraw_tests(model, seed, noise_mm, tests_dir):
    """Per test: (name, posed truth [N,3] mm, observations [N,3] mm, mask [N], pose matrix [4,4]),
    re-drawn exactly as posterior_ssm_tests.py draws them and checked against its files."""
    rng = np.random.default_rng(seed)
    lam = np.clip(model.eigenvalues, 0.0, None)
    names, idx = model.landmark_names(), model.landmark_indices()
    out = []
    for t, name in enumerate(TESTS):
        alpha = rng.standard_normal(model.n_components) * np.sqrt(lam)
        truth = (model.mean + model.eigenfunctions @ alpha).reshape(-1, 3) * MM_PER_UNIT
        pose, _, _ = random_pose(rng, truth.mean(axis=0))
        posed = pose.apply_points(truth)
        observed = posed + rng.normal(scale=noise_mm, size=truth.shape)
        lmk = {n: posed[i] for n, i in zip(names, idx)}
        mask = region(name, posed, lmk, rng)
        test = f"test_{t:02d}_{name}"
        saved = vtk_to_numpy(load_vtp(os.path.join(tests_dir, test, "ground_truth.vtp")).GetPoints().GetData())
        saved_obs = load_vtp(os.path.join(tests_dir, test, "observed.vtp"))
        saved_mask = vtk_to_numpy(saved_obs.GetPointData().GetArray("observed")).astype(bool)
        if np.abs(saved - posed).max() > 1e-3 or not np.array_equal(saved_mask, mask):
            raise ValueError(f"{test}: the re-drawn test does not match {tests_dir}; "
                             f"check --seed / --noise-mm against the posterior run")
        obs_pts = vtk_to_numpy(saved_obs.GetPoints().GetData()).astype(np.float64)
        out.append((test, posed, obs_pts, mask, pose.matrix, lmk))
    return out


def flow_decoder(flow, template, center, scale, mm_per_unit):
    """decoder(idx [T, M], z [T, n_z]) -> [T, M, 3] mm: template vertices idx flowed with z
    (trained_run.flow_points, which the lddmm ShapeModel decodes with)."""
    def decode(idx, z):
        return flow_points(flow, template, center, scale, z, idx) * mm_per_unit
    return decode


def ssm_decoder(model, mm_per_unit, device):
    """decoder(idx [T, M], s [T, K]) -> [T, M, 3] mm: mean + Phi sqrt(lambda) s, s in sigma units
    (the gpmm ShapeModel)."""
    shape_model = GpmmShapeModel(model, device)

    def decode(idx, s):
        return shape_model.decode(s, idx) * mm_per_unit
    return decode


def laplacian_smooth(points, faces, iterations, lam=0.5):
    """[N, 3] points after `iterations` uniform Laplacian steps over the edges of `faces`.

    Vertices in no face keep their position. For the varifold's observations: vertex noise
    comparable to the edge length turns the triangle normals the varifold compares
    into noise, which a few steps remove (at the cost of a slight shrinkage).
    """
    if iterations <= 0:
        return points
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = torch.cat([edges, edges.flip(1)])
    degree = torch.zeros(len(points), device=points.device).index_add_(0, edges[:, 0], torch.ones(len(edges), device=points.device))
    has = degree > 0
    for _ in range(iterations):
        mean = torch.zeros_like(points).index_add_(0, edges[:, 0], points[edges[:, 1]])
        mean[has] = mean[has] / degree[has, None]
        points = torch.where(has[:, None], (1 - lam) * points + lam * mean, points)
    return points


def code_parametrization(kind, train_cfg, train_codes, template_norm, domain, args, device):
    """(wrap, theta mean [P], prior precision [P, P], label): how the fitted code z is parametrised.

    gauss: theta = z, prior N(mean, covariance) of the training codes (needs fewer code
        entries than training codes, else the covariance is singular).
    pca: theta = s, z = mean + U sqrt(lambda) s over the principal components of the training
        codes holding --pca-var of their variance; prior 1/2 |s|^2. Statistical only.
    pca_residual: theta = (s, r), z = mean + U sqrt(lambda) s + sqrt(v) r, v the training codes'
        variance per entry (so s and r are both in sigma units); prior 1/2 |s|^2 +
        1/2 sum_e w_e r_e^2. On a direct
        grid, w_e comes from each node's distance to the template surface, in grid cells
        (as code.grid_reg kind distance, independent of how the run was trained): 0 up to
        --residual-near, 1 from --residual-far on, linear in between, mapped onto
        --residual-w LOW HIGH. A low w lets the data deform that node's region beyond the
        statistical part, a high w holds it at the statistical part.
    wrap(decoder) turns a decoder of z into a decoder of theta. The priors are built by
    src/models/lddmm/shape_model.build_code_prior, which the lddmm ShapeModel uses too.
    """
    prior = build_code_prior(kind, train_cfg, train_codes, template_norm, domain, device, args.pca_var,
                             tuple(args.residual_w), args.residual_near, args.residual_far)

    def wrap(dec):
        return lambda idx, theta: dec(idx, prior.code(theta))
    return wrap, prior.mean, prior.precision, prior.label


def fit_codes(decoder, n_template, targets, candidates, mu, s_inv, args, device, lr=None,
              template_mm=None, template_faces=None, obs_triangles=None, free=None):
    """{variant: [T, D] codes} minimising E(z) for each target, the decoder fixed: the fit of
    src/shared/fitting/code_fit.py with this script's options (args: --variants, --steps,
    --lr, --points, --data, --fit-points, --trunc-mm, --lr-decay, --fit-faces,
    --varifold-sigmas, --keops). lr overrides args.lr (SSM sigmas need a larger one)."""
    return shared_fit_codes(
        decoder, n_template, targets, candidates, mu, s_inv, {v: parse_variant(v) for v in args.variants},
        args.steps, lr or args.lr, args.points, device, data=args.data, fit_points=args.fit_points,
        trunc_mm=args.trunc_mm, lr_decay=args.lr_decay, template_mm=template_mm, template_faces=template_faces,
        obs_triangles=obs_triangles, fit_faces=getattr(args, "fit_faces", 2000),
        varifold_sigmas=getattr(args, "varifold_sigmas", (3.0, 6.0, 12.0)), keops=getattr(args, "keops", False),
        free=free)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="the SSM-trained cohort run")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--tests-dir", default="outputs/posterior_SSM_tests")
    parser.add_argument("--model", default="assets/shape_models/baby/baby_pca.npz")
    parser.add_argument("--seed", type=int, default=0, help="the seed posterior_ssm_tests.py ran with")
    parser.add_argument("--noise-mm", type=float, default=0.3, help="the noise posterior_ssm_tests.py ran with")
    parser.add_argument("--variants", nargs="+", default=["data", "gauss0.5", "gauss3"])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--points", type=int, default=2000, help="observed vertices per code and step")
    parser.add_argument("--data", choices=("chamfer", "varifold", "l2"), default="chamfer",
                        help="chamfer or varifold (no correspondence), or l2 (the observations' own vertex indices)")
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
                        help="varifold with KeOps: all observed triangles, --fit-faces fixed template triangles")
    parser.add_argument("--smooth-obs", type=int, default=0,
                        help="varifold: Laplacian steps on the observed vertices before their triangles are taken")
    parser.add_argument("--fit-points", type=int, default=6000, help="chamfer: template vertices flowed per step")
    parser.add_argument("--trunc-mm", type=float, default=2.0, help="chamfer: cap of the fit -> observation distance")
    parser.add_argument("--no-ssm", dest="ssm", action="store_false",
                        help="skip the SSM fit under the same energy (default: fit it too)")
    parser.add_argument("--ssm-lr", type=float, default=0.05, help="Adam step size of the SSM coefficients (sigma units)")
    parser.add_argument("--output", default=None, help="default: <tests-dir>/code_fit_<run name>_<data>")
    parser.add_argument("--panel-px", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    out_dir = args.output or os.path.join(args.tests_dir, f"code_fit_{os.path.basename(os.path.normpath(args.run))}_{args.data}"
                                          + ("" if args.code_prior == "gauss" else f"_{args.code_prior}"))
    os.makedirs(out_dir, exist_ok=True)

    model = load_model_file(args.model)
    faces = model.faces
    tests = redraw_tests(model, args.seed, args.noise_mm, args.tests_dir)
    _, flow, train_codes, _, template_norm, transform = load_run(args.run, args.step, device)
    for p in flow.parameters():
        p.requires_grad_(False)
    z_train = train_codes.double().numpy()
    train_cfg = load_run_config(args.run)
    wrap, mu, s_inv, prior_label = code_parametrization(args.code_prior, train_cfg, train_codes, template_norm,
                                                        _normalization_domain(train_cfg), args, device)
    print(f"code prior: {prior_label}")
    center, scale = transform.center.to(device), transform.scale.to(device)
    template = template_norm.points[0].to(device)                       # [N, 3] normalised

    # Targets in the unposed model frame (mm). One code per test, the tests of a variant optimised
    # together, the variants one after the other (a chamfer step flows many template vertices).
    targets, obs_idx = [], []
    for test, posed, obs, mask, matrix, _ in tests:
        inv = np.linalg.inv(matrix)
        targets.append(torch.as_tensor(obs @ inv[:3, :3].T + inv[:3, 3], dtype=torch.float32, device=device))
        obs_idx.append(torch.as_tensor(np.flatnonzero(mask), device=device))
    # Varifold: the observed surface is the template triangles with all three vertices observed.
    vf_inputs = {}
    if args.data == "varifold":
        faces_t = torch.as_tensor(faces, dtype=torch.long, device=device)
        obs_tris = []
        for (test, posed, obs, mask, matrix, _), target in zip(tests, targets):
            keep = torch.as_tensor(mask[faces].all(1), device=device)
            obs_tris.append(laplacian_smooth(target, faces_t[keep], args.smooth_obs)[faces_t[keep]])
        vf_inputs = {"template_mm": torch.as_tensor(model.mean.reshape(-1, 3) * MM_PER_UNIT, dtype=torch.float32,
                                                    device=device),
                     "template_faces": faces_t, "obs_triangles": obs_tris}
    # The flow's code, and (--ssm) the SSM's coefficients under the same energy, data term and pose.
    n_vert = len(template)
    methods = {"code fit": (wrap(flow_decoder(flow, template, center, scale, MM_PER_UNIT)), mu, s_inv,
                             args.lr if args.code_prior == "gauss" else args.theta_lr)}
    if args.ssm:
        k = model.n_components
        methods["SSM fit"] = (ssm_decoder(model, MM_PER_UNIT, device), torch.zeros(k, device=device),
                              torch.eye(k, device=device), args.ssm_lr)
    fits = {}                                                            # (method, variant) -> [T, N, 3] mm, d2 [T]
    for method, (decoder, m, si, lr) in methods.items():
        print(f"== {method}")
        codes = fit_codes(decoder, n_vert, targets, obs_idx, m, si, args, device, lr=lr, **vf_inputs)
        with torch.no_grad():
            for v, z in codes.items():
                full = torch.arange(n_vert, device=device)[None].expand(len(tests), -1)
                fits[(method, v)] = (decoder(full, z).cpu().numpy(),
                                     torch.einsum("ji,ik,jk->j", z - m, si, z - m).cpu().numpy())

    # Full-mesh evaluation, posed with the true pose.
    summary = {"run": args.run, "step": args.step, "tests_dir": args.tests_dir, "variants": args.variants,
               "steps": args.steps, "data": args.data, "trunc_mm": args.trunc_mm,
               "pose": "true pose for the code and SSM fits (the posterior estimates its own)",
               "posterior": "ssm_posterior stage: corresponded observations", "tests": {}}
    posterior_summary = json.load(open(os.path.join(args.tests_dir, "summary.json")))["tests"]
    for ti, (test, posed, obs, mask, matrix, lmk) in enumerate(tests):
        entry = {"n_observed": int(mask.sum()), "posterior": {
            s: {k: f[k] for k in ("error_observed_median_mm", "error_free_median_mm", "error_free_p95_mm",
                                  "mahalanobis_sq")} for s, f in posterior_summary[test]["fits"].items()}}
        polys = {}
        for (method, v), (pts, d2) in fits.items():
            fit = pts[ti] @ matrix[:3, :3].T + matrix[:3, 3]
            err = np.linalg.norm(fit - posed, axis=1)
            entry.setdefault(method, {})[v] = {"error_observed_median_mm": float(np.median(err[mask])),
                                               "error_free_median_mm": float(np.median(err[~mask])),
                                               "error_free_p95_mm": float(np.percentile(err[~mask], 95)),
                                               "mahalanobis_sq": float(d2[ti])}
            polys[f"{method} {v}"] = (create_polydata(fit, faces), err)
        summary["tests"][test] = entry
        print(f"{test} ({mask.sum()} obs):")
        for method in list(methods) + ["posterior"]:
            print(f"    {method:<10}" + "  ".join(
                f"{v}: {r['error_observed_median_mm']:.2f}/{r['error_free_median_mm']:.2f}/"
                f"{r['error_free_p95_mm']:.2f} d2 {r['mahalanobis_sq']:.0f}" for v, r in entry[method].items()))

        # Views: truth | observations | code fits | SSM fits | posterior (largest sigma); then error maps.
        truth_mesh = create_polydata(posed, faces)
        observed_mesh = sub_surface(load_vtp(os.path.join(args.tests_dir, test, "observed.vtp")), mask)
        post_dir = sorted(d for d in os.listdir(os.path.join(args.tests_dir, test)) if d.startswith("posterior_"))
        post_sigmas = {float(d.rsplit("sigma", 1)[1].rstrip("mm")): d for d in post_dir}
        s_max = max(post_sigmas)
        post = load_vtp(os.path.join(args.tests_dir, test, post_sigmas[s_max], "posterior_mean.vtp"))
        post_err = np.linalg.norm(vtk_to_numpy(post.GetPoints().GetData()).astype(float) - posed, axis=1)
        panels = [("Ground truth", truth_mesh), (f"Observations {mask.sum()}/{len(mask)}", observed_mesh)]
        panels += [(label, poly) for label, (poly, _) in polys.items()]
        panels += [(f"SSM posterior (corresp.) s{s_max:g}", post)]
        cams = {view: views.view_camera(lmk, [truth_mesh], view) for view in ("frontal", "left", "right")}
        rows = [(cams[view], [(t + ("" if view == "frontal" else f" ({view})"), p,
                               views.SCAN_COLOR if t.startswith("Obs") else views.MESH_COLOR, [])
                              for t, p in panels]) for view in cams]
        rgb = np.where(mask[:, None], views.SCAN_COLOR, views.MESH_COLOR) * 255
        err_row = [("Ground truth: observed (blue) / free", with_colors(truth_mesh, rgb.round().astype(np.uint8)), None, []), None]
        err_row += [(f"{label}: free med {np.median(e[~mask]):.2f} mm", with_colors(p, ramp_colors(e, ERROR_MAX_MM)), None, [])
                    for label, (p, e) in polys.items()]
        err_row += [(f"posterior: free med {np.median(post_err[~mask]):.2f} mm",
                     with_colors(post, ramp_colors(post_err, ERROR_MAX_MM)), None, [])]
        rows.append((cams["frontal"], err_row))
        header = (f"{test}: {args.data} fits, true pose: code of the frozen flow "
                  f"({os.path.basename(os.path.normpath(args.run))}) and SSM coefficients; SSM posterior on corresponded "
                  f"observations for reference; error light 0 -> dark {ERROR_MAX_MM:g} mm")
        views.write_grid(rows, len(panels), header, os.path.join(out_dir, f"{test}.png"), args.panel_px)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
