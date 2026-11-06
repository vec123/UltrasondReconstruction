#!/usr/bin/env python
"""Draw a synthetic cohort from a linear shape model, as files a cohort run reads.

The command line of the stage sample (src/shared/sampling/stage.py) for a GPMM,
which does the work and describes the output layout. From a YAML:
python scripts/run.py configs/gpmm/sample/baby_pca_sobol1000.yaml

Each member is x_i = mean + Phi alpha_i, alpha_ik = s_ik sqrt(lambda_k), s_ik ~ N(0, 1),
so every member is in vertex correspondence with the model mean, row for row.
That is what ``loss.data_name: l2`` needs: the mean is written as the template
(``target:`` of a cohort run), the members into their own folder (``source:``).
coefficients.npz holds s as ``sigmas`` (folders written before this stage also
hold alpha and the eigenvalues).

Usage:
    python scripts/SSM/sample_cohort.py --model assets/shape_models/baby/baby_pca.npz \\
        --output-dir /home/victor/Escritorio/Data/BabySSMData_sobol1000 --n 1000 --sampling sobol
"""

import argparse

from src.shared.sampling.draws import SAMPLINGS
from src.shared.sampling.stage import CodesConfig, Config, run

import scripts.pipeline.registry  # noqa: F401  (registers the shape-model kinds)


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
    parser.add_argument("--sampling", choices=SAMPLINGS, default="random",
                        help="independent normal draws, or a scrambled Sobol sequence (even coverage)")
    args = parser.parse_args()

    run(Config(model={"kind": "gpmm", "path": args.model}, output_dir=args.output_dir, subset=args.subset,
               codes=CodesConfig(source="prior", n=args.n, sampling=args.sampling, seed=args.seed,
                                 clip=args.clip, parameters=args.modes)))


if __name__ == "__main__":
    main()
