"""Synthetic check of the SSM posterior (src/SSM/model/inference/posterior.py).

Draws one ground-truth shape from the model, observes only one side of it with
known noise, conditions the model on that side and reports:

- the residual at the observed vertices (should be near the noise level),
- the posterior std on the observed and on the free side, against the prior's
  (observed: much smaller; free: smaller or equal, never larger),
- how often the ground truth lies within 2 posterior std on the free side
  (calibration: a large share),
- the log marginal likelihood at the true noise against a wrong one (the true
  one should win),
- that a saved posterior model loads back unchanged through load_model_file,
- that thinned observations keep their minimum spacing,
- that posterior_with_pose recovers a known rotation and scale of the
  observations, and keeps the coefficients plausible where the plain posterior
  does not.

All in model units, reported in mm through the model-to-template scale.

Run with:
    python scripts/inspection/check_ssm_posterior.py
    python scripts/inspection/check_ssm_posterior.py --model shape_models/baby/baby_pca.npz
"""

import argparse
import os
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree

from src.SSM.model.model_memory_io import load_model_file  # noqa: E402
from src.SSM.model.inference.observations import Observations, build_noise_model  # noqa: E402
from src.SSM.model.inference.posterior import posterior, posterior_with_pose, save_posterior  # noqa: E402
from src.transforms.similarity import SimilarityTransform  # noqa: E402

DEFAULT_MODEL = "shape_models/baby/baby_pca_sum_matern32.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Prior model .npz.")
    parser.add_argument("--mm-per-unit", type=float, default=5335.0,
                        help="Model-to-template scale, to report in mm.")
    parser.add_argument("--sigma-mm", type=float, default=0.3, help="Observation noise std (mm).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    model = load_model_file(args.model)
    mm = args.mm_per_unit
    sigma = args.sigma_mm / mm
    print(f"model: {args.model} ({model.n_vertices} vertices, {model.n_components} modes, "
          f"inner product {model.inner_product}, noise std {np.sqrt(model.noise) * mm:.3f} mm)")

    truth = model.sample(1, rng=rng)[0].reshape(-1, 3)
    points = truth + rng.standard_normal(truth.shape) * sigma

    # One side observed, the other free: a stand-in for a partial scan.
    side = model.mean.reshape(-1, 3)[:, 0]
    errors = np.where(side > np.median(side), 0.0, 10.0)
    cutoff = build_noise_model("cutoff", {"sigma": sigma, "threshold": 1.0})
    obs = Observations.from_errors(points, errors, cutoff)
    print(obs.describe())

    result = posterior(model, obs)
    observed = obs.mask(model.n_vertices)
    std = result.vertex_std()
    prior_std = result.prior_std()
    error = np.linalg.norm(result.mean.reshape(-1, 3) - truth, axis=1)

    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")

    residual = np.median(result.residuals()) * mm
    check("residual at observed vertices", residual < 3 * args.sigma_mm,
          f"median {residual:.3f} mm, noise {args.sigma_mm} mm")
    ratio = np.median(std[observed] / prior_std[observed])
    check("posterior std shrinks where observed", ratio < 0.5, f"median posterior/prior std {ratio:.3f}")
    grew = np.max(std / prior_std)
    check("posterior std never exceeds prior std", grew <= 1.0 + 1e-6, f"max posterior/prior std {grew:.4f}")
    covered = np.mean(error[~observed] <= 2 * std[~observed])
    check("ground truth within 2 std on the free side", covered > 0.8, f"{covered:.1%} of free vertices")
    print(f"      free side: error median {np.median(error[~observed]) * mm:.3f} mm, "
          f"posterior std median {np.median(std[~observed]) * mm:.3f} mm, "
          f"prior std median {np.median(prior_std[~observed]) * mm:.3f} mm")

    wrong = posterior(model, Observations(obs.indices, obs.targets, obs.variances * 100.0))
    check("log marginal likelihood prefers the true noise",
          result.log_marginal_likelihood > wrong.log_marginal_likelihood,
          f"{result.log_marginal_likelihood:.1f} (true) vs {wrong.log_marginal_likelihood:.1f} (10x std)")

    logistic = build_noise_model("logistic", {"sigma": sigma, "sigma_max": 1.0, "threshold": 1.0, "width": 1e-3})
    soft = posterior(model, Observations.from_errors(points, errors, logistic))
    gap = np.max(np.linalg.norm((soft.mean - result.mean).reshape(-1, 3), axis=1)) * mm
    check("logistic with width -> 0 matches the cutoff", gap < 0.05, f"max mean difference {gap:.4f} mm")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "posterior.npz")
        save_posterior(path, result, provenance={"check": "synthetic"})
        loaded = load_model_file(path)
    same = (np.allclose(loaded.mean, result.model.mean)
            and np.allclose(loaded.eigenvalues, result.model.eigenvalues)
            and np.allclose(loaded.eigenfunctions, result.model.eigenfunctions))
    check("saved posterior loads back unchanged", same, f"{loaded.n_components} modes")

    # The same observations, rotated 5 degrees about the face's centre and scaled by 0.92.
    angle, scale = np.radians(5.0), 0.92
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    centre = truth.mean(axis=0)
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = centre - scale * rotation @ centre
    moved = SimilarityTransform(matrix).apply_points(points)
    moved_obs = Observations.from_errors(moved, errors, cutoff)

    # Only near the applied pose: a shape drawn from the model differs from the
    # mean by a little rotation and scale of its own, which the fit may assign
    # to either side.
    plain = posterior(model, moved_obs)
    posed = posterior_with_pose(model, moved_obs, with_scale=True, iterations=3)
    found = posed.summary()["pose"]
    check("pose recovered", abs(found["rotation_deg"] - 5.0) < 1.0 and abs(found["scale"] - scale) < 0.015,
          f"rotation {found['rotation_deg']:.2f} deg (5), scale {found['scale']:.4f} ({scale})")
    check("pose keeps the coefficients plausible",
          posed.mahalanobis_sq() < min(1.5 * result.mahalanobis_sq(), plain.mahalanobis_sq()),
          f"Mahalanobis^2 {posed.mahalanobis_sq():.0f} with pose, {result.mahalanobis_sq():.0f} unmoved, "
          f"{plain.mahalanobis_sq():.0f} without pose")
    posed_error = np.linalg.norm(posed.posed(posed.mean).reshape(-1, 3)
                                 - SimilarityTransform(matrix).apply_points(truth), axis=1)
    free_prior = np.median(prior_std[~observed]) * scale
    check("posed mean beats the prior on the free side",
          np.median(posed_error[~observed]) < 0.5 * free_prior,
          f"median error {np.median(posed_error[~observed]) * mm:.3f} mm, prior std {free_prior * mm:.3f} mm")

    spacing = 3.0 / mm
    thin = obs.thinned(spacing)
    nearest = cKDTree(thin.targets).query(thin.targets, k=2)[0][:, 1]
    check("thinned observations keep the spacing", nearest.min() >= spacing and thin.n < obs.n,
          f"{thin.n}/{obs.n} kept, closest pair {nearest.min() * mm:.2f} mm (3 mm)")

    print(f"{sum(checks)}/{len(checks)} checks passed")
    sys.exit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
