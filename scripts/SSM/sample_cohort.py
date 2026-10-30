#!/usr/bin/env python
"""Draw a synthetic cohort from a linear shape model, as files a cohort run reads.

Each member is

.. math:: x_i = \\bar{x} + \\Phi \\alpha_i, \\quad \\alpha_{ik} = s_{ik} \\sqrt{\\lambda_k},
          \\quad s_{ik} \\sim N(0, 1)

so every member is in vertex correspondence with the model mean, row for row.
That is what ``loss.data_name: l2`` needs: the mean is written as the template
(``target:`` of a cohort run), the members into their own folder (``source:``),
and the mapping error then draws ONE subsample for both, because their vertex
counts are equal.

Output layout (``--output-dir``):

  members/sample_00000.vtp ...   the cohort (``source:``); file order = code index
  template.vtp                   the model mean (``target:``)
  coefficients.npz               alpha [n, K], sigmas s [n, K], eigenvalues [K],
                                 vertex_idx [N'] (the kept vertices), seed, model path

The coefficients are what the learned codes can be compared against later.

``--subset N`` keeps the same N random vertices on every member and on the
template, so correspondence holds. The meshes then have no faces (point clouds):
the l2 term does not need them, but surface renders do.

Usage:
    python scripts/SSM/sample_cohort.py \\
        --model src/SSM/saved_models/baby/baby_pca.npz \\
        --output-dir outputs/baby_ssm_cohort/n5000 --n 5000 --seed 0
"""

import argparse
from pathlib import Path

import numpy as np

from src.SSM.model.model_memory_io import load_model_file
from src.vtk.create import create_polydata
from src.vtk.io import save_vtp


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
    if sampling == "random":
        sigmas = np.random.default_rng(seed).standard_normal((n, k))
    elif sampling == "sobol":
        from scipy.stats import norm, qmc
        u = qmc.Sobol(d=k, scramble=True, seed=seed).random(n)
        sigmas = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    else:
        raise ValueError(f"sampling must be 'random' or 'sobol', got {sampling!r}")
    if clip is not None:
        sigmas = np.clip(sigmas, -clip, clip)
    alpha = sigmas * np.sqrt(model.eigenvalues[:k])
    vertices = model.mean[None, :] + alpha @ model.eigenfunctions[:, :k].T  # [n, 3N]
    return alpha, sigmas, vertices.reshape(n, -1, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="shape model (.npz / .mat / atlas dir)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=5000, help="number of members")
    parser.add_argument("--modes", type=int, default=None, help="first K modes only (default: all)")
    parser.add_argument("--clip", type=float, default=None, help="clip each mode to +-clip sigma")
    parser.add_argument("--subset", type=int, default=0,
                        help="keep this many random vertices (same on all shapes); 0 = full mesh")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling", choices=("random", "sobol"), default="random",
                        help="independent normal draws, or a scrambled Sobol sequence (even coverage)")
    args = parser.parse_args()

    model = load_model_file(args.model)
    out = Path(args.output_dir)
    members_dir = out / "members"
    if members_dir.exists() and any(members_dir.glob("*.vtp")):
        raise FileExistsError(f"{members_dir} already holds .vtp files; choose an empty --output-dir")

    alpha, sigmas, vertices = draw_cohort(model, args.n, args.modes, args.clip, args.seed, args.sampling)
    mean = model.mean.reshape(-1, 3)
    faces = model.faces

    vertex_idx = np.arange(mean.shape[0])
    if args.subset:
        if not 0 < args.subset < mean.shape[0]:
            raise ValueError(f"--subset must be in 1..{mean.shape[0] - 1}, got {args.subset}")
        rng = np.random.default_rng(args.seed + 1)
        vertex_idx = np.sort(rng.choice(mean.shape[0], args.subset, replace=False))
        faces = None  # the triangulation does not survive dropping vertices

    save_vtp(create_polydata(mean[vertex_idx], faces), str(out / "template.vtp"))
    for i, verts in enumerate(vertices):
        save_vtp(create_polydata(verts[vertex_idx], faces), str(members_dir / f"sample_{i:05d}.vtp"))
    np.savez(out / "coefficients.npz", alpha=alpha, sigmas=sigmas,
             eigenvalues=model.eigenvalues[:alpha.shape[1]], vertex_idx=vertex_idx,
             seed=args.seed, sampling=args.sampling, model=str(args.model))

    print(f"wrote {args.n} members ({len(vertex_idx)} vertices, {alpha.shape[1]} modes) "
          f"+ template to {out}")


if __name__ == "__main__":
    main()
