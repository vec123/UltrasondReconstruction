#!/usr/bin/env python
"""Update the code prior of a frozen SSM-trained flow from partial observations - EM in code space.

The nonlinear counterpart of update_ssm_em.py. The shape model is the flow of a cohort
auto-decoder trained on samples of a linear SSM (e.g. the BabySSM sobol1000 run),
``x = phi_z(T)``, frozen, with a Gaussian prior ``z ~ N(m, C)`` on its code. EM learns
``m`` and ``C`` from the same observations as update_ssm_em.py: the trusted vertices of
a resnet_lddmm apply run that deformed the same mean shape, vertex for vertex.

    E-step   per case, the MAP code under N(m, C) by Gauss-Newton - the flow linearised
             at the current code, ``phi_{z + d}(T)_I ~ phi_z(T)_I + J d``, J from central
             differences of the flow - with the similarity pose refitted at every step.
             At the MAP, the Laplace posterior ``z_i | y_i ~ N(a_i, S_i)``, from the same
             closed form as the linear posterior (condition_whitened with B = J L).
    M-step   update_ssm_em.py's, with the base prior N(m0, C0) = the Gaussian of the
             flow's own training codes in place of the PCA's N(0, Lambda0):
             C = (nu C0 + sum w_i ((a_i - m)(a_i - m)^T + S_i)) / (nu + sum w_i).

The pose makes the mean's own size free (see update_ssm_em.py), so the mean code is kept
where its shape change from ``phi_m0(T)`` has no part along the 7 similarity directions,
to first order: ``m = m0 + P (m_bar - m0)`` with P from the flow's Jacobian at m0 over
the whole mesh. The likelihood is reported in mm (Laplace evidence of each case).

Output, in ``<output-root>/<run name>/<variant_name>``:

    prior.npz          m, C (the learned prior), m0, C0, n_em - init.code_prior_file of a
                       resnet_lddmm code fit reads it as it is
    codes.npz          per case: name, code a, posterior covariance S, under base and final prior
    mean_mm.vtp        phi_m(T) in mm, with point fields mean_shift_mm (from phi_m0(T)),
                       new_std_mm / base_std_mm (from --std-samples draws), n_cases_observed
    em_summary.json    settings, EM history, per-case statistics, the mean's size check
    generations/       base_mean, base_draw_<k>, updated_mean, updated_draw_<k> (mm, model frame)
    fits/<case>/       base_fit.vtp, updated_fit.vtp in the case's world frame, for the
                       --eval-cases (held out, from --eval-run) and --n-fit-cases update cases
    fits/fits.json     per fit case: run, status, log-likelihood (mm) under each prior

Usage:
    python scripts/SSM/update_code_em.py \\
        --run outputs/Casos_Mar/resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g8c2_dist_kin8000_cherry \\
        --flow-run outputs/baby_ssm_cohort/resnet_lddmm_cohort_autodecoder_sobol1000_global_l2_mlp6x256_kin200
"""

import argparse
import json
import os
import time

# The per-case solves are K x K (K ~ 128): threaded BLAS makes each one ~30x slower.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import torch

from scripts.inspection.eval_ssm_cohort import load_run  # noqa: E402
from scripts.SSM.update_ssm_em import (DIRECTION_TOKENS, POSE_TOKENS, case_weights,  # noqa: E402
                                       load_cases, mm_stats, whitener)
from src.models.gpmm.model.model_memory_io import load_model_file  # noqa: E402
from src.models.gpmm.model.inference.posterior import condition_whitened  # noqa: E402
from src.shared.transforms.similarity import SimilarityTransform  # noqa: E402
from src.shared.vtk.create import create_polydata  # noqa: E402
from src.shared.vtk.fields import add_point_fields  # noqa: E402
from src.shared.vtk.io import save_vtp  # noqa: E402


def variant_name(args) -> str:
    """``code_em_<flow run>_<pose>_<dir>_<noise><thr>mm[_spacing<s>mm]_sigma<s>mm_nu<nu>_<weighting>weight``."""
    flow = os.path.basename(os.path.normpath(args.flow_run)).replace("_", "-")
    parts = ["code_em", flow, POSE_TOKENS[args.pose], DIRECTION_TOKENS[args.error_direction],
             f"{args.noise}{args.threshold:g}mm"]
    if args.spacing > 0:
        parts.append(f"spacing{args.spacing:g}mm")
    parts += [f"sigma{args.sigma:g}mm", f"nu{args.nu:g}", f"{args.weighting}weight"]
    return "_".join(parts)


class FlowDecoder:
    """``phi_z(T)`` of a frozen flow at chosen template vertices, in the run's world frame."""

    def __init__(self, flow, template_norm, transform, device, chunk=64):
        self.flow = flow
        self.template = template_norm.points[0].to(device)
        self.center, self.scale = transform.center.to(device), transform.scale.to(device)
        self.device = device
        self.chunk = chunk

    @torch.no_grad()
    def __call__(self, idx, z):
        """idx [C, R] long, z [C, K] -> [C, R, 3] float64."""
        out = [self.flow(self.template[idx[i:i + self.chunk]], z[i:i + self.chunk].float()).points[-1]
               for i in range(0, len(z), self.chunk)]
        return (torch.cat(out) * self.scale + self.center).double()

    def jacobian(self, idx, z, eps):
        """[C, 3R, K] d phi_z(T)_idx / dz by central differences, step eps [K] per code entry."""
        n, k = z.shape
        cols = []
        for j in range(k):
            dz = torch.zeros_like(z)
            dz[:, j] = eps[j]
            cols.append(((self(idx, z + dz) - self(idx, z - dz)) / (2.0 * eps[j])).reshape(n, -1))
        return torch.stack(cols, dim=-1)


def padded_indices(cases, device):
    """[C, R_max] observed vertex ids, each case padded with its first id (its first R rows count)."""
    r_max = max(case.observations.n for case in cases)
    idx = np.zeros((len(cases), r_max), dtype=np.int64)
    for i, case in enumerate(cases):
        r = case.observations.n
        idx[i, :r], idx[i, r:] = case.observations.indices, case.observations.indices[0]
    return torch.as_tensor(idx, device=device)


def fit_pose(x, case, args):
    """Similarity model frame -> observations, of the shape's observed vertices x [R, 3]."""
    if args.pose == "none":
        return None, case.observations
    pose = SimilarityTransform.from_correspondences(
        x, case.observations.targets, with_scale=args.pose == "similarity",
        weights=1.0 / case.observations.variances, stage="posterior_pose")
    return pose, case.observations.transformed(pose.inverse())


def e_step(decoder, cases, idx, z, m, C, args, iterations):
    """Gauss-Newton MAP codes of every case under N(m, C), from the codes z [C, K]; then Laplace.

    Returns (codes [C, K], per-case dicts: S, lml, lml_mm, pose, residuals).
    """
    L = whitener(C)
    C_inv = np.linalg.inv(C)
    eps = torch.as_tensor(args.rel_eps * np.sqrt(np.diag(C)), dtype=torch.float64, device=z.device)

    def energy(x, z_now, observations, r):
        diff = (x[:r] - observations.targets) ** 2
        return 0.5 * float((diff.sum(1) / observations.variances).sum()) + \
            0.5 * float((z_now - m) @ C_inv @ (z_now - m))

    fits = [None] * len(cases)
    for _ in range(iterations):
        x_all = decoder(idx, z).cpu().numpy()
        J_all = decoder.jacobian(idx, z, eps).cpu().numpy()
        z_new = z.clone()
        for i, case in enumerate(cases):
            r = case.observations.n
            x = x_all[i, :r]
            J = J_all[i].reshape(-1, 3, J_all.shape[-1])[:r].reshape(3 * r, -1)
            pose, observations = fit_pose(x, case, args)
            z_i = z[i].cpu().numpy()
            w = observations.precision()
            res = observations.targets.reshape(-1) - x.reshape(-1) - J @ (m - z_i)
            beta, A_inv, lml = condition_whitened(J @ L, w, res)
            step = m + L @ beta
            fits[i] = {"S": L @ A_inv @ L.T, "lml": lml, "pose": pose, "observations": observations,
                       "energy": energy(x, z_i, observations, r)}
            z_new[i] = torch.as_tensor(step, dtype=z.dtype, device=z.device)
        # Backtracking: a step that raises a case's energy is halved, once and again.
        for _ in range(2):
            x_new = decoder(idx, z_new).cpu().numpy()
            worse = []
            for i, case in enumerate(cases):
                r = case.observations.n
                if energy(x_new[i], z_new[i].cpu().numpy(), fits[i]["observations"], r) > fits[i]["energy"]:
                    worse.append(i)
            if not worse:
                break
            for i in worse:
                z_new[i] = 0.5 * (z[i] + z_new[i])
        z = z_new

    x_all = decoder(idx, z).cpu().numpy()
    out = []
    for i, case in enumerate(cases):
        r = case.observations.n
        pose, observations = fit_pose(x_all[i, :r], case, args)
        length = case.mm * (pose.scale if pose is not None else 1.0)
        S = fits[i]["S"]
        out.append({"S": 0.5 * (S + S.T), "lml": fits[i]["lml"],
                    "lml_mm": fits[i]["lml"] - 3 * r * np.log(length), "pose": pose, "length": length,
                    "residuals": np.linalg.norm(x_all[i, :r] - observations.targets, axis=1)})
    return z, out


def gauge_projector(decoder, m0, n_vertices, args, device):
    """(K, K) P: m0 + P (m - m0) has no first-order similarity change of phi_m(T) from phi_m0(T)."""
    idx = torch.arange(n_vertices, device=device)[None]
    z0 = torch.as_tensor(m0, dtype=torch.float64, device=device)[None]
    x = decoder(idx, z0)[0].cpu().numpy()
    eps = torch.as_tensor(np.full(len(m0), args.gauge_eps), dtype=torch.float64, device=device)
    J = decoder.jacobian(idx, z0, eps)[0].cpu().numpy()                    # (3N, K)
    d = x - x.mean(axis=0)
    generators = [np.tile(e, (len(x), 1)) for e in np.eye(3)] + [np.cross(e, d) for e in np.eye(3)] + [d]
    G = np.stack([g.reshape(-1) for g in generators], axis=1)
    M = G.T @ J
    Q_inv_Mt = np.linalg.solve(J.T @ J + 1e-12 * np.trace(J.T @ J) * np.eye(J.shape[1]), M.T)
    return np.eye(len(m0)) - Q_inv_Mt @ np.linalg.pinv(M @ Q_inv_Mt) @ M


def centroid_size(x):
    return float(np.linalg.norm(x - x.mean(axis=0)))


def decode_full(decoder, codes, n_vertices, device):
    """[n, N, 3] float64 numpy: phi_z(T) on the whole mesh for codes [n, K]."""
    idx = torch.arange(n_vertices, device=device)[None].expand(len(codes), -1)
    return decoder(idx, torch.as_tensor(codes, dtype=torch.float64, device=device)).cpu().numpy()


def write_fits(out, label, cases, codes, results, decoder, n_vertices, faces, device):
    """fits/<case>/<label>.vtp: each case's MAP shape, posed, in the case's world frame."""
    shapes = decode_full(decoder, codes, n_vertices, device)
    for case, shape, result in zip(cases, shapes, results):
        posed = result["pose"].apply_points(shape) if result["pose"] is not None else shape
        world = case.to_model.inverse().apply_points(posed)
        folder = os.path.join(out, "fits", case.name)
        os.makedirs(folder, exist_ok=True)
        save_vtp(create_polydata(world, faces=faces), os.path.join(folder, f"{label}.vtp"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="resnet_lddmm run folder with apply/<case>/ (the EM data)")
    parser.add_argument("--flow-run", required=True, help="the SSM-trained cohort auto-decoder run (frozen)")
    parser.add_argument("--model", default="assets/shape_models/baby/baby_pca.npz",
                        help="the linear model whose mean the flow's template and the run's template are")
    parser.add_argument("--output-root", default="outputs/UpdatedSSM")
    parser.add_argument("--output-dir", default=None, help="default: <output-root>/<run name>/<variant name>")
    parser.add_argument("--error-direction", choices=tuple(DIRECTION_TOKENS), default="target_to_recon")
    parser.add_argument("--noise", default="cutoff")
    parser.add_argument("--sigma", type=float, default=2.0, help="observation std, mm")
    parser.add_argument("--threshold", type=float, default=1.5, help="error threshold, mm")
    parser.add_argument("--spacing", type=float, default=3.0, help="observation thinning, mm")
    parser.add_argument("--pose", choices=tuple(POSE_TOKENS), default="similarity")
    parser.add_argument("--nu", type=float, default=10.0, help="weight of the base prior, in cases")
    parser.add_argument("--weighting", choices=("fetus", "scan"), default="fetus")
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-2, help="stop below this rise per case in the weighted lml")
    parser.add_argument("--gn-first", type=int, default=8, help="Gauss-Newton steps of the first E-step")
    parser.add_argument("--gn", type=int, default=3, help="Gauss-Newton steps of later E-steps (warm start)")
    parser.add_argument("--rel-eps", type=float, default=0.3, help="Jacobian step, x the prior std per entry")
    parser.add_argument("--gauge-eps", type=float, default=None, help="Jacobian step at m0 (default: rel-eps x mean std)")
    parser.add_argument("--std-samples", type=int, default=100, help="draws for the per-vertex std fields")
    parser.add_argument("--n-samples", type=int, default=5, help="draws per prior written to generations/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-run", default=None, help="run with apply/<case>/ for the held-out --eval-cases")
    parser.add_argument("--eval-cases-json", default=None, help="JSON with a 'cases' list")
    parser.add_argument("--n-fit-cases", type=int, default=6, help="update cases written to fits/, spread over the gain")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = args.output_dir or os.path.join(args.output_root, os.path.basename(os.path.normpath(args.run)),
                                          variant_name(args))
    os.makedirs(out, exist_ok=True)
    start = time.time()
    device = torch.device(args.device)

    base = load_model_file(args.model)
    _, flow, train_codes, _, template_norm, transform = load_run(args.flow_run, None, device)
    for p in flow.parameters():
        p.requires_grad_(False)
    decoder = FlowDecoder(flow, template_norm, transform, device)
    template = (template_norm.points[0] * transform.scale + transform.center).double().cpu().numpy()
    gap = np.abs(template - base.mean.reshape(-1, 3)).max()
    if gap > 1e-6 * np.linalg.norm(np.ptp(template, axis=0)):
        raise ValueError(f"the flow's template is not {args.model}'s mean (max gap {gap:.3g})")
    n_vertices, faces = base.n_vertices, base.faces

    z_train = train_codes.double().numpy()
    m0, C0 = z_train.mean(axis=0), np.cov(z_train, rowvar=False)
    if args.gauge_eps is None:
        args.gauge_eps = args.rel_eps * float(np.sqrt(np.diag(C0)).mean())
    P = gauge_projector(decoder, m0, n_vertices, args, device) if args.pose != "none" else np.eye(len(m0))

    cases = load_cases(args, base)
    weights = case_weights(cases, args.weighting)
    idx = padded_indices(cases, device)
    mm = float(np.median([case.mm for case in cases]))
    print(f"update_code_em: {len(cases)} cases, {len(set(c.fetus for c in cases))} fetuses, K={len(m0)}, "
          f"{len(z_train)} training codes, observations median {np.median([c.observations.n for c in cases]):.0f} "
          f"({time.time() - start:.1f} s)")

    m, C = m0.copy(), C0.copy()
    z = torch.as_tensor(np.tile(m0, (len(cases), 1)), dtype=torch.float64, device=device)
    history = []
    for iteration in range(args.max_iterations + 1):
        z, results = e_step(decoder, cases, idx, z, m, C, args,
                            args.gn_first if iteration == 0 else args.gn)
        lml = float(weights @ np.array([r["lml_mm"] for r in results]))
        mean_size = centroid_size(decode_full(decoder, m[None], n_vertices, device)[0])
        history.append({"iteration": iteration, "weighted_lml_mm": lml, "mean_centroid_size": mean_size,
                        "seconds": time.time() - start})
        print(f"  iteration {iteration:3d}: weighted log marginal likelihood (mm) {lml:.4f}, "
              f"mean size {mean_size / history[0]['mean_centroid_size']:.4f} ({time.time() - start:.1f} s)",
              flush=True)
        if iteration == 0:
            base_codes, base_results = z.clone().cpu().numpy(), results
        elif abs(lml - history[-2]["weighted_lml_mm"]) < args.tol * weights.sum():
            break
        if iteration == args.max_iterations:
            break
        a = z.cpu().numpy()
        S = np.stack([r["S"] for r in results])
        m_bar = weights @ a / weights.sum()
        m = m0 + P @ (m_bar - m0)
        d = a - m
        C = (args.nu * C0 + (weights[:, None] * d).T @ d + np.einsum("i,ijk->jk", weights, S)) \
            / (args.nu + weights.sum())
        C = 0.5 * (C + C.T)
    codes = z.cpu().numpy()

    np.savez(os.path.join(out, "prior.npz"), m=m, C=C, m0=m0, C0=C0, n_em=len(m0), flow_run=args.flow_run)
    np.savez(os.path.join(out, "codes.npz"), names=np.array([c.name for c in cases]),
             base_codes=base_codes, base_S=np.stack([r["S"] for r in base_results]),
             codes=codes, S=np.stack([r["S"] for r in results]))

    rng = np.random.default_rng(args.seed)
    draws = {"base": (m0, C0), "updated": (m, C)}
    stds, shapes = {}, {}
    for label, (mean, cov) in draws.items():
        samples = rng.multivariate_normal(mean, cov, size=args.std_samples)
        decoded = np.concatenate([decode_full(decoder, samples[i:i + 10], n_vertices, device)
                                  for i in range(0, len(samples), 10)])
        stds[label] = np.sqrt(((decoded - decoded.mean(axis=0)) ** 2).sum(axis=2).mean(axis=0)) * mm
        shapes[label] = [decode_full(decoder, mean[None], n_vertices, device)[0]] + list(decoded[:args.n_samples])
    os.makedirs(os.path.join(out, "generations"), exist_ok=True)
    for label, row in shapes.items():
        for k, shape in enumerate(row):
            name = "mean" if k == 0 else f"draw_{k}"
            save_vtp(create_polydata(shape * mm, faces=faces), os.path.join(out, "generations", f"{label}_{name}.vtp"))

    n_observed = np.zeros(n_vertices, dtype=np.int32)
    for case in cases:
        n_observed[case.selected] += 1
    mean_shape, base_shape = shapes["updated"][0], shapes["base"][0]
    shift = np.linalg.norm(mean_shape - base_shape, axis=1) * mm
    mesh = create_polydata(mean_shape * mm, faces=faces)
    add_point_fields(mesh, {"new_std_mm": stds["updated"], "base_std_mm": stds["base"],
                            "mean_shift_mm": shift, "n_cases_observed": n_observed})
    save_vtp(mesh, os.path.join(out, "mean_mm.vtp"))

    # Fits: the update cases spread over the gain, and the held-out cases under both priors.
    gains = np.array([r["lml_mm"] - b["lml_mm"] for r, b in zip(results, base_results)])
    picks = np.unique(np.linspace(0, len(cases) - 1, min(args.n_fit_cases, len(cases))).round().astype(int))
    picked = [int(i) for i in np.argsort(gains)[picks]]
    fit_info = {}
    write_fits(out, "base_fit", [cases[i] for i in picked], base_codes[picked],
               [base_results[i] for i in picked], decoder, n_vertices, faces, device)
    write_fits(out, "updated_fit", [cases[i] for i in picked], codes[picked],
               [results[i] for i in picked], decoder, n_vertices, faces, device)
    for i in picked:
        fit_info[cases[i].name] = {"run": args.run, "status": "in update",
                                   "lml_mm_base": base_results[i]["lml_mm"], "lml_mm_updated": results[i]["lml_mm"]}
    if args.eval_run and args.eval_cases_json:
        with open(args.eval_cases_json) as handle:
            names = json.load(handle)["cases"]
        held = load_cases(argparse.Namespace(**{**vars(args), "run": args.eval_run}), base, names)
        h_idx = padded_indices(held, device)
        for label, (mean, cov) in draws.items():
            z0 = torch.as_tensor(np.tile(mean, (len(held), 1)), dtype=torch.float64, device=device)
            h_codes, h_results = e_step(decoder, held, h_idx, z0, mean, cov, args, args.gn_first)
            write_fits(out, f"{label}_fit", held, h_codes.cpu().numpy(), h_results, decoder, n_vertices, faces,
                       device)
            for case, r in zip(held, h_results):
                fit_info.setdefault(case.name, {"run": args.eval_run,
                                                "status": "held out" if case.name not in {c.name for c in cases}
                                                else "in update"})[f"lml_mm_{label}"] = r["lml_mm"]
    with open(os.path.join(out, "fits", "fits.json"), "w") as handle:
        json.dump(fit_info, handle, indent=2)

    seen = n_observed > 0
    summary = {
        "units": "mm, at the cohort's median frame scale",
        "output_dir": out,
        "settings": {**{k: v for k, v in vars(args).items() if k not in ("device",)},
                     "cases": [c.name for c in cases]},
        "mm_per_model_unit": mm,
        "history": history,
        "mean_vs_base": {"centroid_size_ratio": centroid_size(mean_shape) / centroid_size(base_shape)},
        "code_prior": {"trace_C0": float(np.trace(C0)), "trace_C": float(np.trace(C)),
                       "mean_shift_mahalanobis_sq_under_C0": float((m - m0) @ np.linalg.inv(C0) @ (m - m0))},
        "mean_shift_mm": mm_stats(shift),
        "vertex_std_mm": {"base_observed": mm_stats(stds["base"][seen]), "new_observed": mm_stats(stds["updated"][seen]),
                          "base_unobserved": mm_stats(stds["base"][~seen]) if (~seen).any() else {},
                          "new_unobserved": mm_stats(stds["updated"][~seen]) if (~seen).any() else {}},
        "cases": [{"name": c.name, "fetus": c.fetus, "weight": float(w), "n_observed": c.observations.n,
                   "lml_mm_base": b["lml_mm"], "lml_mm_new": r["lml_mm"],
                   "mahalanobis_sq_base": float((bc - m0) @ np.linalg.inv(C0) @ (bc - m0)),
                   "mahalanobis_sq_new": float((nc - m) @ np.linalg.inv(C) @ (nc - m)),
                   "residual_mm_base": mm_stats(b["residuals"] * b["length"]),
                   "residual_mm_new": mm_stats(r["residuals"] * r["length"]),
                   "pose_scale": float(r["pose"].scale) if r["pose"] is not None else 1.0}
                  for c, w, b, r, bc, nc in zip(cases, weights, base_results, results, base_codes, codes)],
    }
    with open(os.path.join(out, "em_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"update_code_em: {history[-1]['iteration']} iterations, mean size "
          f"x{summary['mean_vs_base']['centroid_size_ratio']:.4f}, mean shift median "
          f"{summary['mean_shift_mm']['median']:.2f} mm -> {out} ({time.time() - start:.1f} s)")


if __name__ == "__main__":
    main()
