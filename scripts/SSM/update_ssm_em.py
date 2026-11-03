#!/usr/bin/env python
"""Update a shape model from partial observations of a cohort - EM on its coefficients.

Every case of a resnet_lddmm apply run is a deformation of the model mean, in
vertex correspondence with it, and only part of it is near the scan. As in
scripts/US/pipeline/stages/ssm_posterior.py, a noise model turns each vertex's
error into an observation std and drops the vertices it does not trust, so each
case observes a different subset of the vertices.

The new model keeps the base model's span: ``x = mu0 + Phi alpha`` with
``alpha ~ N(m, C)``. EM estimates ``m`` and ``C``:

    E-step   per case, the posterior of alpha under the current N(m, C), pose
             removed (src.SSM.model.inference.posterior.posterior_with_pose):
             alpha_i | y_i ~ N(a_i, S_i)
    M-step   m = sum w_i a_i / sum w_i
             C = (nu Lambda0 + sum w_i ((a_i - m)(a_i - m)^T + S_i)) / (nu + sum w_i)

``S_i`` is what a PCA of the completed shapes leaves out: without it, a region
that a case did not show counts as the mean there, and the variance comes out
too small. A direction no case observes keeps its base variance. ``nu`` is the
weight of the base covariance ``Lambda0``, in cases; 0 is maximum likelihood on
the cohort alone. ``--weighting fetus`` gives each scan 1 / (scans of its
fetus), so a fetus scanned three times counts once.

With a pose removed, the mean's own pose is free: a smaller model with a larger
pose scale explains the same scan, and in the model frame its density is higher,
so EM shrinks it without end (generalized Procrustes has the same problem). The
M-step therefore keeps the mean's similarity part at the base mean's: ``m`` is
the closest one (per-vertex distance of ``Phi m``) whose shape change has no
part along the 7 similarity directions of ``mu0``. The likelihood is reported
in mm on each scan, which a change of scale alone does not change.

Output, in ``<output-root>/<run name>/<variant_name>`` (``--output-root`` defaults to
outputs/UpdatedSSM) unless ``--output-dir`` is given:

    model.npz          the updated model, GPMM layout, model frame (ssm_fit /
                       ssm_posterior read it as shape_model_mat)
    mean_mm.vtp        new mean scaled to mm, with point fields new_std_mm,
                       base_std_mm, mean_shift_mm, n_cases_observed (cases
                       with the vertex under the cutoff, before thinning)
    em_summary.json    settings, EM history, spectra (mm), per-case statistics

Usage:
    python scripts/SSM/update_ssm_em.py \\
        --run outputs/Casos_Mar/resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g8c2_dist_kin8000_cherry \\
        --model assets/shape_models/baby/baby_pca_sum_matern32.npz
"""

import argparse
import json
import os
import time
from dataclasses import dataclass

# Every solve here is K x K (K ~ 160): threaded BLAS makes each one ~30x slower.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from scripts.US.pipeline.stages.ssm_posterior import (DIRECTION_TOKENS, POSE_TOKENS, fit_frame,
                                                      mesh_points, point_array,
                                                      target_to_recon_errors)
from src.SSM.model.model_memory_io import ModelCacheSpec, load_model_file, save_model
from src.SSM.model.inference.observations import Observations, build_noise_model
from src.SSM.model.inference.posterior import _posterior_model, condition_whitened
from src.shared.transforms.similarity import SimilarityTransform
from src.shared.vtk.create import create_polydata
from src.shared.vtk.fields import add_point_fields
from src.shared.vtk.io import load_vtp, save_vtp


@dataclass
class Case:
    name: str
    fetus: str
    observations: Observations  # model frame, pose not removed
    selected: np.ndarray        # vertex ids under the noise model's cutoff, before thinning
    to_model: SimilarityTransform  # this case's world frame (mm) -> model frame
    phi: np.ndarray             # (3R, K) base Phi rows of the observed vertices
    mu: np.ndarray              # (3R,) base mean at them

    @property
    def mm(self) -> float:
        """mm per model unit of this case's frame."""
        return 1.0 / self.to_model.scale


@dataclass
class Fit:
    """One case's posterior, in the base model's coefficients."""

    a: np.ndarray       # (K,) posterior mean of alpha
    S: np.ndarray       # (K, K) posterior Cov(alpha)
    beta: np.ndarray    # (K,) whitened against the prior it was fitted under
    lml: float          # log p(y) under that prior
    pose: object        # SimilarityTransform model frame -> observations, or None
    residuals: np.ndarray  # (R,) posterior mean to pose-free targets, model frame

    def length(self, case):
        """mm on the case's scan per model-frame unit, pose scale included."""
        return case.mm * (self.pose.scale if self.pose is not None else 1.0)

    def lml_mm(self, case):
        """log p(y) of the observations in mm on the scan: ``lml`` plus the
        Jacobian of the map from the pose-free model frame, 3R coordinates."""
        return self.lml - 3 * case.observations.n * np.log(self.length(case))


def variant_name(args) -> str:
    """``ssm_em_<model>_<pose>_<dir>_<noise><thr>mm[_spacing<s>mm]_sigma<s>mm_nu<nu>_<weighting>``."""
    model = os.path.splitext(os.path.basename(args.model))[0].replace("_", "-")
    parts = ["ssm_em", model, POSE_TOKENS[args.pose], DIRECTION_TOKENS[args.error_direction],
             f"{args.noise}{args.threshold:g}mm"]
    if args.spacing > 0:
        parts.append(f"spacing{args.spacing:g}mm")
    parts += [f"sigma{args.sigma:g}mm", f"nu{args.nu:g}", f"{args.weighting}weight"]
    return "_".join(parts)


def case_errors(case_dir, direction, n_vertices):
    """(N,) per-vertex error of one apply case, in mm."""
    error_path = os.path.join(case_dir, "eval", "recon_w_distance.vtp")
    error_mesh = load_vtp(error_path)
    if direction == "recon_to_target":
        return point_array(error_mesh, "recon_dist_2_target", error_path)
    target = load_vtp(os.path.join(case_dir, "eval", "target_w_distance.vtp"))
    target_points = vtk_to_numpy(target.GetPoints().GetData()).astype(np.float64)
    return target_to_recon_errors(mesh_points(error_mesh, n_vertices, error_path), target_points)


def load_cases(args, model, names=None):
    """Observations of the cases ``names`` (default: every case) of ``<run>/apply``, in the model frame."""
    noise = build_noise_model(args.noise, {"sigma": args.sigma, "threshold": args.threshold})
    apply_dir = os.path.join(args.run, "apply")
    names = names or sorted(name for name in os.listdir(apply_dir)
                            if os.path.isfile(os.path.join(apply_dir, name, "world", "deformed.vtp")))
    if not names:
        raise FileNotFoundError(f"no case with world/deformed.vtp under {apply_dir}")

    cases = []
    for name in names:
        case_dir = os.path.join(apply_dir, name)
        template_path = os.path.join(case_dir, "world", "template_points.vtp")
        observed_path = os.path.join(case_dir, "world", "deformed.vtp")
        template_points = mesh_points(load_vtp(template_path), model.n_vertices, template_path)
        observed_points = mesh_points(load_vtp(observed_path), model.n_vertices, observed_path)
        errors = case_errors(case_dir, args.error_direction, model.n_vertices)

        to_model, _ = fit_frame(model.mean.reshape(-1, 3), template_points, 1e-3, template_path)
        observations = Observations.from_errors(observed_points, errors, noise)
        selected = observations.indices
        observations = observations.thinned(args.spacing)
        if not observations.n:
            raise ValueError(f"{name}: no vertex observed - loosen the noise model")
        observations = observations.transformed(to_model)
        cases.append(Case(name, name.split("_")[0], observations, selected, to_model,
                          model.phi_rows(observations.indices),
                          model.x_rows(model.mean, observations.indices)))
    return cases


def case_weights(cases, weighting):
    if weighting == "scan":
        return np.ones(len(cases))
    fetuses = [case.fetus for case in cases]
    return np.array([1.0 / fetuses.count(fetus) for fetus in fetuses])


def fit_case(base, case, m, L, args):
    """Posterior of one case under alpha ~ N(m, L L^T), pose refitted as in
    src.SSM.model.inference.posterior.posterior_with_pose: first onto the prior mean,
    then onto the latest posterior mean. The pose is fitted on the observed
    rows only, as src.SSM.model.inference.posterior.fit_pose does on the full shape."""
    a, pose, observations = m, None, case.observations
    for _ in range(1 if args.pose == "none" else max(1, args.pose_iterations)):
        if args.pose != "none":
            pose = SimilarityTransform.from_correspondences(
                (case.mu + case.phi @ a).reshape(-1, 3), case.observations.targets,
                with_scale=args.pose == "similarity", weights=1.0 / case.observations.variances,
                stage="posterior_pose")
            observations = case.observations.transformed(pose.inverse())
        w = observations.precision(base.noise)
        r = observations.targets.reshape(-1) - case.mu - case.phi @ m
        beta, A_inv, lml = condition_whitened(case.phi @ L, w, r)
        a = m + L @ beta
    S = L @ A_inv @ L.T
    residuals = np.linalg.norm((case.mu + case.phi @ a).reshape(-1, 3) - observations.targets, axis=1)
    return Fit(a, 0.5 * (S + S.T), beta, lml, pose, residuals)


def similarity_projector(base):
    """(K, K) ``P``: ``P m`` is the ``m'`` closest to ``m`` in ``|Phi (m' - m)|``
    whose shape change ``Phi m'`` has no part along the base mean's similarity
    directions - 3 translations, 3 rotations, 1 scale about its centroid."""
    x = base.mean.reshape(-1, 3)
    d = x - x.mean(axis=0)
    generators = [np.tile(e, (len(x), 1)) for e in np.eye(3)] + [np.cross(e, d) for e in np.eye(3)] + [d]
    G = np.stack([g.reshape(-1) for g in generators], axis=1)               # (3N, 7)
    M = G.T @ base.eigenfunctions                                           # (7, K)
    Q_inv_Mt = np.linalg.solve(base.eigenfunctions.T @ base.eigenfunctions, M.T)
    return np.eye(base.n_components) - Q_inv_Mt @ np.linalg.pinv(M @ Q_inv_Mt) @ M


def m_step(base, results, weights, nu, P):
    a = np.stack([fit.a for fit in results])
    S = np.stack([fit.S for fit in results])
    m = P @ (weights @ a / weights.sum())
    d = a - m
    scatter = (weights[:, None] * d).T @ d + np.einsum("i,ijk->jk", weights, S)
    C = (nu * np.diag(np.clip(base.eigenvalues, 0.0, None)) + scatter) / (nu + weights.sum())
    return m, 0.5 * (C + C.T)


def whitener(C):
    """``L`` with ``L L^T = C``, columns along C's eigenvectors, largest first."""
    values, vectors = np.linalg.eigh(C)
    order = np.argsort(values)[::-1]
    return vectors[:, order] * np.sqrt(np.clip(values[order], 0.0, None))[None, :]


def mm_stats(values):
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "max": float(np.max(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="resnet_lddmm run folder with apply/<case>/")
    parser.add_argument("--model", required=True, help="base model, the one the run deformed the mean of")
    parser.add_argument("--output-root", default="outputs/UpdatedSSM")
    parser.add_argument("--output-dir", default=None, help="default: <output-root>/<run name>/<variant name>")
    parser.add_argument("--error-direction", choices=tuple(DIRECTION_TOKENS), default="target_to_recon")
    parser.add_argument("--noise", default="cutoff", help="noise model with sigma and threshold")
    parser.add_argument("--sigma", type=float, default=2.0, help="observation std, mm")
    parser.add_argument("--threshold", type=float, default=1.5, help="error threshold, mm")
    parser.add_argument("--spacing", type=float, default=3.0, help="observation thinning, mm; 0 keeps all")
    parser.add_argument("--pose", choices=tuple(POSE_TOKENS), default="similarity")
    parser.add_argument("--pose-iterations", type=int, default=3)
    parser.add_argument("--nu", type=float, default=10.0, help="weight of the base covariance, in cases")
    parser.add_argument("--weighting", choices=("fetus", "scan"), default="fetus")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="stop when the weighted log marginal likelihood rises less than this per case")
    args = parser.parse_args()

    out = args.output_dir or os.path.join(args.output_root, os.path.basename(os.path.normpath(args.run)),
                                          variant_name(args))
    os.makedirs(out, exist_ok=True)
    start = time.time()

    base = load_model_file(args.model)
    cases = load_cases(args, base)
    weights = case_weights(cases, args.weighting)
    mm = float(np.median([case.mm for case in cases]))
    P = similarity_projector(base) if args.pose != "none" else np.eye(base.n_components)
    print(f"update_ssm_em: {len(cases)} cases, {len(set(c.fetus for c in cases))} fetuses, "
          f"K={base.n_components}, observations median {np.median([c.observations.n for c in cases]):.0f} "
          f"({time.time() - start:.1f} s)")

    m = np.zeros(base.n_components)
    C = np.diag(np.clip(base.eigenvalues, 0.0, None))
    history = []
    for iteration in range(args.max_iterations + 1):
        L = whitener(C)
        results = [fit_case(base, case, m, L, args) for case in cases]
        lml = float(weights @ np.array([fit.lml_mm(case) for fit, case in zip(results, cases)]))
        history.append({"iteration": iteration, "weighted_lml_mm": lml, "seconds": time.time() - start})
        if iteration % 10 == 0:
            print(f"  iteration {iteration:3d}: weighted log marginal likelihood (mm) {lml:.4f} "
                  f"({time.time() - start:.1f} s)")
        if iteration == 0:
            base_results = results
        elif abs(lml - history[-2]["weighted_lml_mm"]) < args.tol * weights.sum():
            break
        if iteration == args.max_iterations:
            break
        m, C = m_step(base, results, weights, args.nu, P)

    model = _posterior_model(base, m, C)
    provenance = {"base_model": args.model, "run": args.run, "cases": [c.name for c in cases],
                  "error_direction": args.error_direction,
                  "noise": {"name": args.noise, "sigma": args.sigma, "threshold": args.threshold},
                  "spacing": args.spacing, "pose": args.pose, "pose_iterations": args.pose_iterations,
                  "nu": args.nu, "weighting": args.weighting, "iterations": history[-1]["iteration"]}
    save_model(os.path.join(out, "model.npz"), model, ModelCacheSpec(
        kernel_spec={"name": "pca", "kwargs": {"basis": model.kernel.basis}},
        approximation={"name": "ssm_em", "kwargs": provenance},
        inference={"name": "none", "kwargs": {}}))

    n_observed = np.zeros(base.n_vertices, dtype=np.int32)
    for case in cases:
        n_observed[case.selected] += 1
    new_std = model.vertex_std() * mm
    base_std = base.vertex_std() * mm
    shift = np.linalg.norm((model.mean - base.mean).reshape(-1, 3), axis=1) * mm
    mesh = create_polydata(model.mean.reshape(-1, 3) * mm, faces=base.faces)
    add_point_fields(mesh, {"new_std_mm": new_std, "base_std_mm": base_std,
                            "mean_shift_mm": shift, "n_cases_observed": n_observed})
    save_vtp(mesh, os.path.join(out, "mean_mm.vtp"))

    seen = n_observed > 0
    centered = [x.reshape(-1, 3) - x.reshape(-1, 3).mean(axis=0) for x in (model.mean, base.mean)]
    to_base = SimilarityTransform.from_correspondences(model.mean.reshape(-1, 3), base.mean.reshape(-1, 3))
    cos = np.clip((np.trace(to_base.rotation) - 1.0) / 2.0, -1.0, 1.0)
    summary = {
        "units": "mm, at the cohort's median frame scale",
        "output_dir": out,
        "settings": provenance,
        "mm_per_model_unit": mm,
        "history": history,
        "spectrum_std_mm": {"base": (np.sqrt(np.clip(base.eigenvalues, 0, None)) * mm).tolist(),
                            "new": (np.sqrt(np.clip(model.eigenvalues, 0, None)) * mm).tolist()},
        "total_variance_ratio": float(model.eigenvalues.sum() / base.eigenvalues.sum()),
        "mean_shift_mm": mm_stats(shift),
        "mean_vs_base": {"centroid_size_ratio": float(np.linalg.norm(centered[0]) / np.linalg.norm(centered[1])),
                         "procrustes_scale": float(to_base.scale),
                         "procrustes_rotation_deg": float(np.degrees(np.arccos(cos))),
                         "procrustes_translation_mm": float(np.linalg.norm(to_base.translation) * mm)},
        "vertex_std_mm": {"base_observed": mm_stats(base_std[seen]), "new_observed": mm_stats(new_std[seen]),
                          "base_unobserved": mm_stats(base_std[~seen]) if (~seen).any() else {},
                          "new_unobserved": mm_stats(new_std[~seen]) if (~seen).any() else {}},
        "cases": [],
    }
    for case, weight, fit, base_fit in zip(cases, weights, results, base_results):
        summary["cases"].append({
            "name": case.name, "fetus": case.fetus, "weight": float(weight),
            "n_selected": int(case.selected.size), "n_observed": case.observations.n,
            "lml_mm_base": base_fit.lml_mm(case), "lml_mm_new": fit.lml_mm(case),
            "mahalanobis_sq_base": float(base_fit.beta @ base_fit.beta),
            "mahalanobis_sq_new": float(fit.beta @ fit.beta),
            "residual_mm_base": mm_stats(base_fit.residuals * base_fit.length(case)),
            "residual_mm_new": mm_stats(fit.residuals * fit.length(case)),
            "pose_scale": float(fit.pose.scale) if fit.pose is not None else 1.0,
        })
    with open(os.path.join(out, "em_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"update_ssm_em: {history[-1]['iteration']} iterations, total variance "
          f"x{summary['total_variance_ratio']:.3f}, mean shift median {summary['mean_shift_mm']['median']:.2f} mm, "
          f"mean size x{summary['mean_vs_base']['centroid_size_ratio']:.4f} -> {out} ({time.time() - start:.1f} s)")


if __name__ == "__main__":
    main()
