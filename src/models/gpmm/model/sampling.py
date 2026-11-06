"""Draws of a cohort from a linear shape model's prior.

Each member is x_i = mean + Phi alpha_i, with alpha_ik = s_ik sqrt(lambda_k) and
s_ik ~ N(0, 1), so every member is in vertex correspondence with the mean.
StatisticalShapeModel.sample is the short form: all modes, independent draws,
and only the shapes.
"""

import numpy as np

from src.shared.sampling.draws import standard_draws


def draw_cohort(model, n, n_modes=None, clip=None, seed=0, sampling="random"):
    """(alpha [n, K], sigmas [n, K], vertices [n, N, 3]) drawn from the model's prior.

    Args:
        model: StatisticalShapeModel (mean [3N], eigenfunctions [3N, K_all], eigenvalues [K_all])
        n: number of members
        n_modes: use the first K modes only; None = all
        clip: clip every sigma to [-clip, clip]; None = no clip
        seed: seed of the draw, so the same arguments give the same cohort
        sampling: "random" (independent normal draws) or "sobol" (a scrambled Sobol
            sequence through the normal inverse CDF: quasi-random, so the n members cover
            the K-dimensional Gaussian more evenly than independent draws do)
    """
    k = model.n_components if n_modes is None else int(n_modes)
    if not 0 < k <= model.n_components:
        raise ValueError(f"n_modes must be in 1..{model.n_components}, got {k}")
    sigmas = standard_draws(n, k, sampling, seed, clip)
    alpha = sigmas * np.sqrt(model.eigenvalues[:k])
    vertices = model.mean[None, :] + alpha @ model.eigenfunctions[:, :k].T  # [n, 3N]
    return alpha, sigmas, vertices.reshape(n, -1, 3)
