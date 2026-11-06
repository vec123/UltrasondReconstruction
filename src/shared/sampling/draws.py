"""Standard normal draws, the s of every sampled code (theta = prior mean + L s)."""

import numpy as np

SAMPLINGS = ("random", "sobol")


def standard_draws(n, k, sampling="random", seed=0, clip=None):
    """[n, k] draws of N(0, I).

    Args:
        n: number of draws
        k: dimensions
        sampling: "random" (independent normal draws) or "sobol" (a scrambled Sobol
            sequence through the normal inverse CDF: quasi-random, so the n draws cover
            the k-dimensional Gaussian more evenly than independent draws do)
        seed: seed of the draw, so the same arguments give the same draws
        clip: clip every draw to [-clip, clip]; None = no clip
    """
    if sampling == "random":
        draws = np.random.default_rng(seed).standard_normal((n, k))
    elif sampling == "sobol":
        from scipy.stats import norm, qmc
        u = qmc.Sobol(d=k, scramble=True, seed=seed).random(n)
        draws = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    else:
        raise ValueError(f"sampling must be one of {SAMPLINGS}, got {sampling!r}")
    if clip is not None:
        draws = np.clip(draws, -clip, clip)
    return draws
