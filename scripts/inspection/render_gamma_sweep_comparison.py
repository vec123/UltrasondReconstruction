"""Grid of a direct SSM fit's gamma/variant sweep: one row per case, one column
per fit (target first, then free/clamped, then gammas ascending), all sharing
one frontal camera - the code_distance_ranking.py / render_frontal_views.py
style, reusing their camera and grid-rendering machinery.

Usage:
    python scripts/inspection/render_gamma_sweep_comparison.py \
        --root outputs/direct_SSM_fit \
        --model assets/shape_models/baby/baby_pca.npz
"""

import argparse
import glob
import os
import re

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from scripts.inspection.render_frontal_views import MESH_COLOR, SCAN_COLOR, view_camera, write_grid  # noqa: E402
from src.vtk.io import load_vtp  # noqa: E402

LANDMARK_NAMES = [
    "exR", "enR", "n", "enL", "exL", "acR", "aR", "prn", "aL", "acL", "sn",
    "chR", "cphR", "ls", "cphL", "chL", "li", "sl", "pg", "tR", "oiR", "tL", "oiL",
]


def model_landmarks(model_npz):
    mean = model_npz["mean"].reshape(-1, 3)
    verts = model_npz["landmark_verts"]
    return {name: mean[idx] for name, idx in zip(LANDMARK_NAMES, verts)}


def discover_fits(case_dir):
    """(label, path) pairs for every nicp_refined_vertices_<name>.vtp: free, clamped,
    then gamma_<value> ascending - the column order every case shares."""
    paths = glob.glob(os.path.join(case_dir, "nicp_refined_vertices_*.vtp"))
    paths = [p for p in paths if "_iter_" not in p and "_colors" not in p]

    gammas, named = [], {}
    for p in paths:
        name = re.match(r"nicp_refined_vertices_(.+)\.vtp$", os.path.basename(p)).group(1)
        m = re.match(r"gamma_([\d.]+)$", name)
        if m:
            gammas.append((float(m.group(1)), p))
        else:
            named[name] = p

    gammas.sort(key=lambda t: t[0])
    ordered = [(name, named[name]) for name in ("free", "clamped") if name in named]
    ordered += [(f"gamma={g:g}", p) for g, p in gammas]
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="outputs/direct_SSM_fit")
    parser.add_argument("--model", default="assets/shape_models/baby/baby_pca.npz")
    parser.add_argument("--output", default=None, help="PNG to write (default: <root>/gamma_sweep_comparison.png)")
    parser.add_argument("--panel-px", type=int, default=260)
    args = parser.parse_args()

    model_npz = np.load(args.model, allow_pickle=True)
    lmk = model_landmarks(model_npz)

    case_dirs = sorted(
        os.path.dirname(p) for p in glob.glob(os.path.join(args.root, "**", "rigid_alignment.vtp"), recursive=True)
    )
    if not case_dirs:
        raise SystemExit(f"No rigid_alignment.vtp found under {args.root}")

    cases = []
    for case_dir in case_dirs:
        fits = discover_fits(case_dir)
        if not fits:
            print(f"skip {case_dir}: no nicp_refined_vertices_*.vtp")
            continue
        case_name = os.path.relpath(case_dir, args.root)
        target = load_vtp(os.path.join(case_dir, "rigid_alignment.vtp"))
        recons = [(label, load_vtp(path)) for label, path in fits]
        cases.append((case_name, target, recons))

    columns = 1 + max(len(recons) for _, _, recons in cases)
    all_meshes = [target for _, target, _ in cases] + [mesh for _, _, recons in cases for _, mesh in recons]
    camera = view_camera(lmk, all_meshes, "frontal")

    rows = []
    for case_name, target, recons in cases:
        row = [(f"{case_name} target", target, SCAN_COLOR, [])]
        row += [(label, mesh, MESH_COLOR, []) for label, mesh in recons]
        rows.append((camera, row))

    out_path = args.output or os.path.join(args.root, "gamma_sweep_comparison.png")
    write_grid(rows, columns,
               "Direct SSM fit gamma/variant sweep (baby model): target (blue) vs reconstruction (tan) per gamma",
               out_path, args.panel_px)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
