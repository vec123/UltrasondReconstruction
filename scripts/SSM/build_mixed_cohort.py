#!/usr/bin/env python
"""Build a cohort of baby AND fetus shapes in the baby model's frame, for a chamfer-trained flow.

The two shape models do not share a topology (baby: a whole head, 31027 vertices; fetus:
a frontal face mask, 5743 vertices), so the members are point clouds compared by chamfer,
not by vertex correspondence:

  * baby members: the shapes of an existing sample_cohort.py folder (--baby-data), rebuilt
    from its coefficients;
  * fetus members: --n-fetus new samples of --fetus-model (sample_cohort.draw_cohort, same
    --sampling), carried into the baby frame by ONE similarity (with scale), fitted between
    the fetus mean's and the baby mean's shared landmarks - so the fetus model's own
    variation, size included, is kept;
  * every member is a random subset of --points vertices (CohortBatchLoader stacks the
    members, so they need equal counts), written without faces.

Output (--output-dir):
  members/baby_00000.vtp ... members/fetus_00000.vtp ...   (file order = code index)
  template.vtp      the baby mean with its point arrays (landmark_weight, ...): --template
  coefficients.npz  baby_sigmas, fetus_sigmas, the similarity (scale, rotation, translation)

Usage:
    python scripts/SSM/build_mixed_cohort.py \\
        --baby-data /home/victor/Escritorio/Data/BabySSMData_sobol1000 \\
        --template outputs/baby_ssm_cohort/fit_16_26/template.vtp \\
        --output-dir /home/victor/Escritorio/Data/BabyFetusSSMData_sobol1000
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from sample_cohort import draw_cohort
from target_to_ssm_frame import similarity
from src.SSM.model.model_memory_io import load_model_file
from src.vtk.create import create_polydata
from src.vtk.io import save_vtp


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--baby-model", default="src/SSM/saved_models/baby/baby_pca.npz")
    parser.add_argument("--baby-data", required=True, help="sample_cohort.py output of the baby samples")
    parser.add_argument("--fetus-model", default="src/SSM/saved_models/fetus/fetus_pca.npz")
    parser.add_argument("--fetus-landmarks", default="src/SSM/FetusModel_x4/landmarks.json",
                        help="fetus landmark name -> vertex index")
    parser.add_argument("--template", required=True, help="the baby mean with its point arrays, in the baby frame")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-fetus", type=int, default=1000)
    parser.add_argument("--sampling", choices=("random", "sobol"), default="sobol")
    parser.add_argument("--points", type=int, default=4096, help="vertices kept per member")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    members = out / "members"
    if members.exists() and any(members.glob("*.vtp")):
        raise FileExistsError(f"{members} already holds .vtp files; choose an empty --output-dir")
    rng = np.random.default_rng(args.seed + 7)

    baby = load_model_file(args.baby_model)
    fetus = load_model_file(args.fetus_model)
    lmk_fetus = json.load(open(args.fetus_landmarks))
    baby_names, baby_idx = baby.landmark_names(), baby.landmark_indices()
    shared = [n for n in baby_names if n in lmk_fetus]
    a = fetus.mean.reshape(-1, 3)[[lmk_fetus[n] for n in shared]]
    b = baby.mean.reshape(-1, 3)[[baby_idx[baby_names.index(n)] for n in shared]]
    scale, rot, trans = similarity(a, b)                                 # baby ~ scale R fetus + t
    residual = np.linalg.norm(scale * a @ rot.T + trans - b, axis=1)
    print(f"fetus -> baby similarity on {len(shared)} landmarks ({', '.join(shared)}): scale {scale:.3g}, "
          f"landmark residual mean {residual.mean():.2e} max {residual.max():.2e} (baby units)")

    baby_sigmas = np.load(Path(args.baby_data) / "coefficients.npz")["sigmas"]
    baby_basis = baby.eigenfunctions[:, :baby_sigmas.shape[1]] * np.sqrt(baby.eigenvalues[:baby_sigmas.shape[1]])
    for i, s in enumerate(baby_sigmas):
        v = (baby.mean + baby_basis @ s).reshape(-1, 3)
        keep = rng.choice(len(v), args.points, replace=False)
        save_vtp(create_polydata(v[keep]), str(members / f"baby_{i:05d}.vtp"))
    _, fetus_sigmas, fetus_vertices = draw_cohort(fetus, args.n_fetus, seed=args.seed, sampling=args.sampling)
    for i, v in enumerate(fetus_vertices):
        v = scale * v @ rot.T + trans
        keep = rng.choice(len(v), min(args.points, len(v)), replace=False)
        save_vtp(create_polydata(v[keep]), str(members / f"fetus_{i:05d}.vtp"))
    shutil.copy(args.template, out / "template.vtp")
    np.savez(out / "coefficients.npz", baby_sigmas=baby_sigmas, fetus_sigmas=fetus_sigmas,
             similarity_scale=scale, similarity_rotation=rot, similarity_translation=trans,
             shared_landmarks=np.array(shared), points=args.points, seed=args.seed)
    print(f"wrote {len(baby_sigmas)} baby + {args.n_fetus} fetus members ({args.points} points each) to {out}")


if __name__ == "__main__":
    main()
