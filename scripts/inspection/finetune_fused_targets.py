"""Fine-tune an SSM-trained flow per fused fetal target, after the code fit with the flow frozen.

Stage 1 fits the code alone, all cases in one batch, as code_fit_fused_targets.py (chamfer,
--code-prior pca, prior 1/2 |s|^2, data noise --sigma). Stage 2 starts from that code and
also trains a part of the flow, per case and per variant, with data noise --ft-sigma:

  * code:   no stage 2 (the stage-1 fit, the reference);
  * head:   the last two layers of the velocity MLP (mix, proj);
  * lora:   rank --lora-rank updates W + B A of every hidden linear layer (B zero at start);
  * detail: an additive per-case field v = v_flow(x, z) + u(x), u a Fourier MLP with
            --detail-n-e bands (finer than the flow's) and zero-initialised projection;
  * full:   every layer of the flow;
  * kernel: as detail, but u(x) = sum_j exp(-|x - c_j|^2 / (2 l^2)) a_j, a Gaussian-kernel
            field (smooth everywhere, no ReLU kinks) on --kernel-points centres c_j: template
            vertices with w_i < HIGH, at their mid-path position under the stage-1 code;
            widths l = --kernel-mm (several: one weight set per width, multi-scale),
            weights a_j zero at start.

All variants minimise the same energy, so they differ only in what may change:

    E = data(chamfer) / (2 sigma^2) * n_obs  +  1/2 |s|^2
        + n_obs * mean_{k,i} w_i |dv_k,i|^2 / (2 tau^2)
        + n_obs * mean_{k,(i,j)} w_ij |dv_k,i - dv_k,j|^2 / l_ij^2 / (2 kappa^2)

dv = v_new - v_flow the change of the velocity (mm / unit time) on the template vertices
along their new paths, against the frozen flow with the same code. The first penalty, its
kinetic energy, bounds by tau how far the fine-tuned field may leave the statistical one.
w_i comes from the distance d_i of vertex i of the stage-1 fit to the nearest scan point:
--dv-w LOW up to --dv-near mm, HIGH from --dv-far mm on, linear in between. So the field is
free where there is data and held at the statistical fit where there is none. The second
penalty, over template edges (i, j) of rest length l_ij (mm), bounds by kappa (--dv-grad,
1 / unit time) the spatial gradient of the change. w_ij is 1 (--grad-weight uniform: a step
is forbidden everywhere, also on the data, while kappa still allows a nose-sized change) or
the mean of w_i and w_j (--grad-weight w: free on the data, smooth only in the transition and
far from it). Stage 2 keeps the stage-1 code fixed: the statistical part stays at --sigma and
only the free part of the flow takes the local detail. The scan -> fit term weights each
scan point by its confidence (support), so the cut edge of the fused scan pulls less.
The fit -> scan term leaves out the fit vertices whose nearest scan point lies within
--rim-mm of the scan's boundary (outer edge and holes): beside a cut edge their nearest
point is on the edge line, and pulling them onto it makes a crease.

Both stages use fixed point sets (--points scan points; both ends of --fit-edges template
edges, drawn once) and a cosine decay of the step sizes to 0, so the fit is deterministic.

Per case and variant: scan -> fit (mean, p95), fit -> scan (face weighted), the code's
d2 = |s|^2, and the share of triangles whose normal turned by more than --crease-deg against
the stage-1 fit (creases and folds). Writes <output>/summary.json and <output>/<case>.png
(scan, each variant, pair LDDMM and cohort svf; front, left, right; a row of fit -> scan
distances; a row of the normal turn against the stage-1 fit, w_i in the first panel).

Run with:
    python scripts/inspection/finetune_fused_targets.py \\
        --run outputs/baby_ssm_cohort/resnet_lddmm_cohort_autodecoder_sobol1000_g8c2_l2_mlp6x256_kin200
"""

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from vtk.util.numpy_support import vtk_to_numpy

from scripts.inspection import render_frontal_views as views  # noqa: E402
import scripts.inspection.render_ssm_cohort_views  # noqa: E402,F401  (RGB panels)
from scripts.inspection.code_fit_fused_targets import CASOS, references  # noqa: E402
from scripts.inspection.code_fit_partial_tests import ERROR_MAX_MM, code_parametrization, load_run_config  # noqa: E402
from scripts.inspection.eval_ssm_cohort import load_run  # noqa: E402
from scripts.inspection.render_ssm_posterior_views import faces_of, ramp_colors, with_colors  # noqa: E402
from scripts.SSM.target_to_ssm_frame import similarity  # noqa: E402
from src.resnet_lddmm.resnet_flow.fields.fourier import FourierStationaryField  # noqa: E402
from src.resnet_lddmm.resnet_flow.runner import _normalization_domain  # noqa: E402
from src.SSM.model.model_memory_io import load_model_file  # noqa: E402
from src.vtk.create import create_polydata  # noqa: E402
from src.vtk.io import load_vtp  # noqa: E402

VARIANTS = ("code", "head", "lora", "detail", "kernel", "full")


class DetailField(nn.Module):
    """v(x, z) = base(x, z) + extra(x): the flow's field plus a free field without code."""

    def __init__(self, base, extra):
        super().__init__()
        self.base, self.extra = base, extra

    def forward(self, x, step=None, code=None):
        return self.base(x, step, code) + self.extra(x)


class KernelField(nn.Module):
    """u(x) = sum_l sum_j exp(-|x - c_j|^2 / (2 l^2)) a_lj: a smooth stationary field, zero at start.

    One weight set a_l per width l, so a coarse width can carry a large correction far out
    while a fine one adds the detail. Each level is divided by its mean kernel overlap at the
    centres, so a weight is about the displacement it makes, at any width.
    """

    def __init__(self, centres, widths):
        super().__init__()
        self.register_buffer("centres", centres)                        # [J, 3]
        self.widths = list(widths)
        d2 = torch.cdist(centres, centres).pow(2)
        self.overlap = [float(torch.exp(-d2 / (2 * l ** 2)).sum(1).mean()) for l in self.widths]
        self.a = nn.Parameter(centres.new_zeros(len(self.widths), *centres.shape))

    def forward(self, x, step=None, code=None):
        d2 = torch.cdist(x, self.centres[None].expand(len(x), -1, -1)).pow(2)
        return sum(torch.exp(-d2 / (2 * l ** 2)) @ a / n for l, a, n in zip(self.widths, self.a, self.overlap))


class LoRALinear(nn.Module):
    """linear(x) + x A^T B^T, A [r, in] random, B [out, r] zero: the identity update at start."""

    def __init__(self, linear, rank):
        super().__init__()
        self.linear = linear
        self.a = nn.Parameter(torch.randn(rank, linear.in_features, device=linear.weight.device)
                              / math.sqrt(linear.in_features))
        self.b = nn.Parameter(torch.zeros(linear.out_features, rank, device=linear.weight.device))

    def forward(self, x):
        return self.linear(x) + (x @ self.a.T) @ self.b.T


def build_variant(variant, base_flow, args, kernel=None):
    """(flow, trainable parameters): a copy of the frozen flow with the variant's free part.

    kernel: (centres [J, 3], width), normalised units, for the kernel variant.
    """
    flow = copy.deepcopy(base_flow)
    for p in flow.parameters():
        p.requires_grad_(False)
    block = flow.field.blocks[0]
    if variant == "head":
        params = list(block.mix.parameters()) + list(block.proj.parameters())
    elif variant == "full":
        params = list(flow.parameters())
    elif variant == "lora":
        block.lift = LoRALinear(block.lift, args.lora_rank)
        for layer in block.hidden:
            layer.linear = LoRALinear(layer.linear, args.lora_rank)
        block.mix = LoRALinear(block.mix, args.lora_rank)
        params = [p for n, p in flow.named_parameters() if n.endswith((".a", ".b"))]
    elif variant == "detail":
        extra = FourierStationaryField(mlp=args.detail_mlp, n_e=args.detail_n_e)
        flow.field = DetailField(flow.field, extra.to(next(base_flow.parameters()).device))
        params = list(extra.parameters())
    elif variant == "kernel":
        extra = KernelField(*kernel)
        flow.field = DetailField(flow.field, extra)
        params = list(extra.parameters())
    elif variant == "code":
        params = []
    else:
        raise ValueError(f"unknown variant {variant!r}")
    for p in params:
        p.requires_grad_(True)
    return flow, params


def optimise(base_flow, variant, theta0, z_of, geometry, targets, sigma, steps, args, device, dv_weight=None,
             kernel=None):
    """(theta [C, K], flow): minimise E for the C targets together, see the module docstring.

    variant "code" fits the codes alone with the frozen flow (stage 1); any other variant
    also trains its part of a copy of the flow (stage 2, C = 1). targets are (points [N_c, 3]
    mm, support [N_c]) per case; dv_weight [N] is w_i on the template vertices.
    """
    template, center, scale, s, edges, edge_len = geometry
    flow, params = (base_flow, []) if variant == "code" else build_variant(variant, base_flow, args, kernel)
    # Stage 2 keeps the stage-1 code: the statistical part stays at its sigma, the free part
    # of the flow takes the local detail.
    theta = theta0.clone().requires_grad_(not params)
    opt = torch.optim.Adam([{"params": params, "lr": args.ft_lr}] if params else [{"params": [theta], "lr": args.theta_lr}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator(device=device).manual_seed(0)
    sel = torch.randperm(len(edges), device=device, generator=gen)[:args.fit_edges]
    fit_idx = edges[sel].reshape(-1)                                     # both ends of each edge, [2E']
    m = min([args.points] + [len(t) for t, _, _ in targets])               # equal counts, to stack the cases
    picks = [torch.randperm(len(t), device=device, generator=gen)[:m] for t, _, _ in targets]
    y = torch.stack([t[p] for (t, _, _), p in zip(targets, picks)])
    y_w = torch.stack([sup[p] / sup[p].sum() for (_, sup, _), p in zip(targets, picks)])
    y_inner = torch.stack([~rim[p] for (_, _, rim), p in zip(targets, picks)]).float()
    n_obs = torch.tensor([float(len(t)) for t, _, _ in targets], device=device)
    q0 = template[fit_idx][None].expand(len(targets), -1, -1)
    w = None if dv_weight is None else dv_weight[fit_idx]
    w_edge = None if w is None else w.reshape(-1, 2).mean(1)            # [E'] mean of the two ends
    if w_edge is not None and args.grad_weight == "uniform":
        w_edge = torch.ones_like(w_edge)
    mm = scale * s                                                       # mm per normalised unit
    t0 = time.time()
    for step in range(steps):
        z = z_of(theta)
        traj = flow(q0, z)
        pred = (traj.points[-1] * scale + center) * s                    # [C, P, 3] mm
        dist = torch.cdist(y, pred)
        sq = (dist.min(2).values.pow(2) * y_w).sum(1)
        # fit -> scan without the fit vertices whose nearest scan point is on the scan's rim:
        # beside a cut edge they would be drawn onto the edge line (a crease).
        nearest = dist.min(1)
        back = (nearest.values.pow(2).clamp(max=args.trunc_mm ** 2) * y_inner.gather(1, nearest.indices)).mean(1)
        loss = ((sq + back) * n_obs / (2 * sigma ** 2) + 0.5 * theta.pow(2).sum(1)).sum()
        dv2 = grad2 = torch.zeros((), device=device)
        if params:
            if variant in ("detail", "kernel"):                          # the change is the added field itself
                dv = torch.stack([flow.field.extra(q) for q in traj.points[:-1]])
            else:
                dv = traj.velocities - torch.stack([base_flow.field(q, k, z) for k, q in enumerate(traj.points[:-1])])
            dv = dv * mm
            dv2 = (dv.pow(2).sum(-1) * w).mean()                         # [K, C, P] weighted per vertex
            ends = dv.reshape(*dv.shape[:2], -1, 2, 3)                   # [K, C, E', 2, 3]
            grad2 = ((ends[..., 0, :] - ends[..., 1, :]).pow(2).sum(-1) / edge_len[sel] ** 2 * w_edge).mean()
            loss = loss + n_obs.sum() * (dv2 / (2 * args.tau ** 2) + grad2 / (2 * args.dv_grad ** 2))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"    {variant:<6} step {step:5d}  rms scan -> fit "
                  + " ".join(f"{v:.2f}" for v in sq.sqrt().tolist())
                  + f" mm  rms w dv {dv2.sqrt():.2f} mm  rms grad dv {grad2.sqrt():.3f}  |s|^2 "
                  + " ".join(f"{v:.0f}" for v in theta.pow(2).sum(1).tolist())
                  + f"  {time.time() - t0:.0f} s", flush=True)
    return theta.detach(), flow


def dv_weights(fit_mm, target_mm, args):
    """[N] w_i from the distance of each vertex of the stage-1 fit to the nearest scan point."""
    d, _ = cKDTree(target_mm).query(fit_mm)
    ramp = np.clip((d - args.dv_near) / (args.dv_far - args.dv_near), 0.0, 1.0)
    low, high = args.dv_w
    return low + (high - low) * ramp


def scan_rim(points, faces, rim_mm):
    """[N] bool: the scan points within rim_mm of a boundary edge (an edge of one triangle only)."""
    if rim_mm <= 0:
        return np.zeros(len(points), dtype=bool)
    pairs = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges, count = np.unique(pairs, axis=0, return_counts=True)
    boundary = np.unique(edges[count == 1])
    d, _ = cKDTree(points[boundary]).query(points)
    return d <= rim_mm


def normals(points, faces):
    n = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    return n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-12, None)


def normal_turn(points, ref_normals, faces):
    """([F] degrees each triangle's normal turned against ref_normals, [N] the largest per vertex)."""
    deg = np.degrees(np.arccos(np.clip((normals(points, faces) * ref_normals).sum(1), -1.0, 1.0)))
    per_vertex = np.zeros(len(points))
    for c in range(3):
        np.maximum.at(per_vertex, faces[:, c], deg)
    return deg, per_vertex


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="the SSM-trained cohort run")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--cases", nargs="+", default=["16_26", "13_26", "11_36"])
    parser.add_argument("--variants", nargs="+", default=["code", "detail", "lora"], choices=VARIANTS)
    parser.add_argument("--fused-dir", default=f"{CASOS}/cohort/fused_targets")
    parser.add_argument("--aligned-template", default=f"{CASOS}/cohort/template_baby_mean_aligned.vtp")
    parser.add_argument("--model", default="src/SSM/saved_models/baby/baby_pca.npz")
    parser.add_argument("--sigma", type=float, default=3.0, help="stage 1: data noise (mm), as the gauss<sigma> variants")
    parser.add_argument("--ft-sigma", type=float, default=1.0, help="stage 2: data noise (mm)")
    parser.add_argument("--tau", type=float, default=1.0, help="bound of the velocity change (mm / unit time)")
    parser.add_argument("--dv-w", type=float, nargs=2, default=[0.0, 1.0],
                        help="w_i near the scan and far from it")
    parser.add_argument("--dv-near", type=float, default=2.0, help="mm to the scan up to which w_i is LOW")
    parser.add_argument("--dv-far", type=float, default=15.0, help="mm to the scan from which w_i is HIGH")
    parser.add_argument("--dv-grad", type=float, default=0.1,
                        help="kappa: bound of the spatial gradient of the velocity change (1 / unit time)")
    parser.add_argument("--grad-weight", choices=("uniform", "w"), default="uniform",
                        help="weight of the gradient penalty per edge: 1 everywhere, or w_ij (free on the data)")
    parser.add_argument("--rim-mm", type=float, default=1.0,
                        help="scan points within this distance of the scan's boundary (outer edge and holes) "
                             "are not matched by the fit -> scan term; 0: off")
    parser.add_argument("--crease-deg", type=float, default=30.0,
                        help="normal turn against the stage-1 fit counted as a crease or fold")
    parser.add_argument("--code-steps", type=int, default=1000)
    parser.add_argument("--ft-steps", type=int, default=300)
    parser.add_argument("--theta-lr", type=float, default=0.05)
    parser.add_argument("--ft-lr", type=float, default=1e-3, help="Adam step size of the flow parameters")
    parser.add_argument("--pca-var", type=float, default=0.99)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--detail-n-e", type=int, default=5)
    parser.add_argument("--detail-mlp", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--kernel-points", type=int, default=1000, help="kernel: number of centres")
    parser.add_argument("--kernel-mm", type=float, nargs="+", default=[4.0, 20.0],
                        help="kernel: Gaussian widths l (mm), one weight set each")
    parser.add_argument("--points", type=int, default=4000, help="fixed scan points per case")
    parser.add_argument("--fit-edges", type=int, default=4000,
                        help="fixed template edges whose two ends are flowed per step")
    parser.add_argument("--trunc-mm", type=float, default=2.0, help="cap of the fit -> scan distance")
    parser.add_argument("--output", default=None)
    parser.add_argument("--panel-px", type=int, default=260)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    name = (f"{os.path.basename(os.path.normpath(args.run))}_pca_sigma{args.sigma:g}_code{args.code_steps}"
            f"_ftsigma{args.ft_sigma:g}_tau{args.tau:g}_dvw{args.dv_w[0]:g}-{args.dv_w[1]:g}"
            f"_near{args.dv_near:g}_far{args.dv_far:g}_grad{'w' if args.grad_weight == 'w' else ''}{args.dv_grad:g}_support_codefrozen_rim{args.rim_mm:g}"
            f"_ftlr{args.ft_lr:g}_ft{args.ft_steps}")
    out_dir = args.output or os.path.join("outputs/finetunedflow", name)
    os.makedirs(out_dir, exist_ok=True)

    model = load_model_file(args.model)
    faces = model.faces
    aligned = load_vtp(args.aligned_template)
    aligned_pts = vtk_to_numpy(aligned.GetPoints().GetData()).astype(np.float64)
    face_w = vtk_to_numpy(aligned.GetPointData().GetArray("landmark_weight")).astype(np.float64)
    s, rot, trans = similarity(model.mean.reshape(-1, 3), aligned_pts)   # aligned = s R mean + t

    _, flow, train_codes, _, template_norm, transform = load_run(args.run, args.step, device)
    for p in flow.parameters():
        p.requires_grad_(False)
    train_cfg = load_run_config(args.run)
    wrap, mu, _, prior_label = code_parametrization("pca", train_cfg, train_codes, template_norm,
                                                    _normalization_domain(train_cfg), args, device)
    z_of = lambda theta: wrap(lambda idx, z: z)(None, theta)  # noqa: E731
    z_of.k = len(mu)
    center, scale = transform.center.to(device), transform.scale.to(device)
    template = template_norm.points[0].to(device)
    print(f"code prior: {prior_label}, output {out_dir}")

    pairs = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    edges = torch.as_tensor(pairs, dtype=torch.long, device=device)
    edge_len = ((template[edges[:, 0]] - template[edges[:, 1]]) * scale * s).norm(dim=-1)   # mm at rest
    geometry = (template, center, scale, s, edges, edge_len)

    def decode(ft_flow, theta):
        """[N, 3] mm, unposed: the full template flowed with one case's code."""
        with torch.no_grad():
            return (ft_flow(template[None], z_of(theta)).points[-1][0] * scale + center) * s

    scans, targets = {}, []
    for case in args.cases:
        scan = load_vtp(os.path.join(args.fused_dir, f"{case}_semanas.vtp"))
        pts = vtk_to_numpy(scan.GetPoints().GetData()).astype(np.float64)
        scans[case] = (scan, pts, vtk_to_numpy(scan.GetPointData().GetArray("support")).astype(np.float64))
        # mm, unposed: the model frame scaled by s (x = s R a + t  ->  s a = R^T (x - t)).
        targets.append((torch.as_tensor((pts - trans) @ rot, dtype=torch.float32, device=device),
                        torch.as_tensor(scans[case][2], dtype=torch.float32, device=device),
                        torch.as_tensor(scan_rim(pts, faces_of(scan), args.rim_mm), device=device)))
        print(f"{case}: {100 * targets[-1][2].float().mean():.0f}% of the scan points on the rim")
    print("stage 1: codes of all cases")
    theta1, _ = optimise(flow, "code", torch.zeros(len(targets), z_of.k, device=device), z_of, geometry, targets,
                    args.sigma, args.code_steps, args, device)

    summary = {"run": args.run, "args": vars(args), "cases": {}}
    names, lmk_idx = model.landmark_names(), model.landmark_indices()
    for ci, case in enumerate(args.cases):
        scan, pts, support = scans[case]
        print(f"{case}:")
        t0 = time.time()
        code_fit = decode(flow, theta1[ci:ci + 1])
        w_np = dv_weights(code_fit.cpu().numpy(), targets[ci][0].cpu().numpy(), args)
        print(f"    w_i: {100 * (w_np <= args.dv_w[0] + 1e-9).mean():.0f}% at LOW, "
              f"{100 * (w_np >= args.dv_w[1] - 1e-9).mean():.0f}% at HIGH")
        w = torch.as_tensor(w_np, dtype=torch.float32, device=device)
        # Kernel centres: vertices where the field may change (w_i < HIGH), at mid-path.
        free = np.flatnonzero(w_np < args.dv_w[1] - 1e-9)
        pick = torch.as_tensor(np.random.default_rng(0).choice(free, min(args.kernel_points, len(free)), replace=False),
                               device=device)
        with torch.no_grad():
            path = flow(template[pick][None], z_of(theta1[ci:ci + 1])).points
        kernel = (path[len(path) // 2][0], [l / float(scale * s) for l in args.kernel_mm])
        fits = {}
        for variant in args.variants:
            theta, ft_flow = (theta1[ci:ci + 1], flow) if variant == "code" else \
                optimise(flow, variant, theta1[ci:ci + 1], z_of, geometry, [targets[ci]], args.ft_sigma,
                    args.ft_steps, args, device, dv_weight=w, kernel=kernel)
            unposed = decode(ft_flow, theta)
            fits[variant] = (unposed.double().cpu().numpy() @ rot.T + trans, float(theta.pow(2).sum()))
        print(f"    {time.time() - t0:.0f} s")

        tree = cKDTree(pts)
        n_ref = normals(fits["code"][0], faces)

        def distances(fit):
            d_scan, _ = cKDTree(fit).query(pts)
            d_fit, _ = tree.query(fit)
            m = {"scan_to_fit_mm_mean": float(d_scan.mean()),
                 "scan_to_fit_mm_p95": float(np.percentile(d_scan, 95)),
                 "fit_to_scan_mm_face_weighted": float((d_fit * face_w).sum() / face_w.sum())}
            turn = None
            if len(fit) == len(fits["code"][0]):                         # same mesh as the stage-1 fit
                deg, turn = normal_turn(fit, n_ref, faces)
                m["creased_triangles"] = float((deg > args.crease_deg).mean())
            return m, d_fit, turn

        entry, shapes = {}, {}
        for variant, (fit, d2) in fits.items():
            metrics, d_fit, turn = distances(fit)
            entry[variant] = {**metrics, "mahalanobis_sq": d2}
            shapes[variant] = (create_polydata(fit, faces), d_fit, turn,
                               f"{metrics['scan_to_fit_mm_mean']:.2f} mm, d2 {d2:.0f}")
        for label, path in references(case).items():
            poly = load_vtp(path)
            metrics, d_fit, turn = distances(vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64))
            entry[label] = {**metrics, "path": path}
            shapes[label] = (poly, d_fit, turn, f"{metrics['scan_to_fit_mm_mean']:.2f} mm")
        summary["cases"][case] = entry
        for key, m in entry.items():
            print(f"    {key:<11} {m['scan_to_fit_mm_mean']:.2f}/{m['scan_to_fit_mm_p95']:.2f}/"
                  f"{m['fit_to_scan_mm_face_weighted']:.2f}"
                  + (f" d2 {m['mahalanobis_sq']:.0f}" if "mahalanobis_sq" in m else "")
                  + (f" creased {100 * m['creased_triangles']:.2f}%" if "creased_triangles" in m else ""))

        lmk = {n: aligned_pts[i] for n, i in zip(names, lmk_idx)}
        panels = [(f"scan {case}", scan, views.SCAN_COLOR)]
        panels += [(f"{label} ({info})", poly, views.MESH_COLOR) for label, (poly, _, _, info) in shapes.items()]
        cams = {view: views.view_camera(lmk, [p for _, p, _ in panels], view) for view in ("frontal", "left", "right")}
        rows = [(cams[view], [(t + ("" if view == "frontal" else f" ({view})"), p, c, []) for t, p, c in panels])
                for view in cams]
        top = float(np.percentile(support, 95))
        err_row = [(f"scan confidence 0 -> {top:.1f}", with_colors(scan, ramp_colors(support, top)), None, [])]
        err_row += [(f"{label}: fit -> scan", with_colors(poly, ramp_colors(d_fit, ERROR_MAX_MM)), None, [])
                    for label, (poly, d_fit, _, _) in shapes.items()]
        rows.append((cams["frontal"], err_row))
        turn_max = 2 * args.crease_deg
        turn_row = [(f"w_i on the code fit, 0 -> 1", with_colors(shapes["code"][0], ramp_colors(w_np, 1.0)), None, [])]
        turn_row += [(f"{label}: normal turn", poly, views.MESH_COLOR, []) if turn is None else
                     (f"{label}: normal turn ({100 * entry[label]['creased_triangles']:.1f}% > {args.crease_deg:g} deg)",
                      with_colors(poly, ramp_colors(turn, turn_max)), None, [])
                     for label, (poly, _, turn, _) in shapes.items()]
        rows.append((cams["frontal"], turn_row))
        header = (f"{case}: code fit ({prior_label}, sigma {args.sigma:g} mm) then per-case fine-tuning of "
                  f"{os.path.basename(os.path.normpath(args.run))} (sigma {args.ft_sigma:g} mm, tau {args.tau:g} mm, "
                  f"w {args.dv_w[0]:g} -> {args.dv_w[1]:g} from {args.dv_near:g} to {args.dv_far:g} mm to the scan, "
                  f"kappa {args.dv_grad:g}, {args.ft_steps} steps). Row 4: distance to the nearest scan point, "
                  f"light 0 -> dark {ERROR_MAX_MM:g} mm. Row 5: normal turn against the code fit, light 0 -> dark "
                  f"{turn_max:g} deg")
        views.write_grid(rows, len(panels), header, os.path.join(out_dir, f"{case}.png"), args.panel_px)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
