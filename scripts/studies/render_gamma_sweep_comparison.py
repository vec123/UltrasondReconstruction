"""Grid of a direct SSM fit's gamma/variant sweep: one row per case, one column
per fit (target first, then free/clamped, then gammas ascending), all sharing
one frontal camera - the code_distance_ranking.py / render_views stage
style, reusing their camera and grid-rendering machinery.

With --per-case it writes instead, for each case folder (the one that holds
rigid_alignment.vtp): <fit name>.png for each fit in that folder, and
fits_comparison.png with all fits in the folder above it. Every PNG has a frontal
row (target and fits) and left and right rows of the fits only, in the same columns.

Usage:
    python scripts/studies/render_gamma_sweep_comparison.py \
        --root outputs/direct_SSM_fit \
        --model assets/shape_models/baby/baby_pca.npz
    python scripts/studies/render_gamma_sweep_comparison.py \
        --root outputs/BabyFM_test --per-case
"""

import argparse
import glob
import os
import re

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from src.shared.vtk.render import MESH_COLOR, SCAN_COLOR, view_camera, write_grid  # noqa: E402
from src.shared.vtk.io import load_vtp  # noqa: E402
from src.shared.vtk.transform import transform_polydata_matrix  # noqa: E402

VIEW_SIZE_MM = 100.0  # the model frame is ~0.02 across; the cameras work in mm

LANDMARK_NAMES = [
    "exR", "enR", "n", "enL", "exL", "acR", "aR", "prn", "aL", "acL", "sn",
    "chR", "cphR", "ls", "cphL", "chL", "li", "sl", "pg", "tR", "oiR", "tL", "oiL",
]


def model_landmarks(model_npz):
    mean = model_npz["mean"].reshape(-1, 3)
    verts = model_npz["landmark_verts"]
    return {name: mean[idx] for name, idx in zip(LANDMARK_NAMES, verts)}


def fit_name(path):
    """``map_1x`` for ``nicp_refined_vertices_map_1x.vtp``."""
    return re.match(r"nicp_refined_vertices_(.+)\.vtp$", os.path.basename(path)).group(1)


def discover_fits(case_dir):
    """(label, path) pairs for every nicp_refined_vertices_<name>.vtp: free, clamped,
    any other name (sorted), then gamma_<value> ascending - the column order every case shares."""
    paths = glob.glob(os.path.join(case_dir, "nicp_refined_vertices_*.vtp"))
    paths = [p for p in paths if "_iter_" not in p and "_colors" not in p]

    gammas, named = [], {}
    for p in paths:
        name = fit_name(p)
        m = re.match(r"gamma_([\d.]+)$", name)
        if m:
            gammas.append((float(m.group(1)), p))
        else:
            named[name] = p

    gammas.sort(key=lambda t: t[0])
    first = [name for name in ("free", "clamped") if name in named]
    ordered = [(name, named[name]) for name in first + sorted(set(named) - set(first))]
    ordered += [(f"gamma={g:g}", p) for g, p in gammas]
    return ordered


def scaled(mesh, factor):
    return transform_polydata_matrix(mesh, np.diag([factor, factor, factor, 1.0]))


def view_rows(cameras, target, recons):
    """A frontal, a left and a right row; each row has the target, then the fits."""
    rows = []
    for camera, side in zip(cameras, ("", " left", " right")):
        rows.append((camera, [(f"target{side}", target, SCAN_COLOR, [])]
                     + [(f"{label}{side}", mesh, MESH_COLOR, []) for label, mesh in recons]))
    return rows


def render_per_case(cases, lmk, panel_px):
    """<fit name>.png beside each case's fits, fits_comparison.png in the folder above."""
    for case_name, case_dir, target, recons in cases:
        cameras = [view_camera(lmk, [target] + [mesh for _, mesh in recons], view)
                   for view in ("frontal", "left", "right")]
        for (label, mesh), (_, path) in zip(recons, discover_fits(case_dir)):
            out_path = os.path.join(case_dir, f"{fit_name(path)}.png")
            write_grid(view_rows(cameras, target, [(label, mesh)]), 2, f"{case_name}: {label}", out_path, panel_px)
            print(f"Saved {out_path}")
        out_path = os.path.join(os.path.dirname(case_dir), "fits_comparison.png")
        write_grid(view_rows(cameras, target, recons), 1 + len(recons),
                   f"{case_name}: target (blue) vs fits (tan)", out_path, panel_px)
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="outputs/direct_SSM_fit")
    parser.add_argument("--model", default="assets/shape_models/baby/baby_pca.npz")
    parser.add_argument("--output", default=None, help="PNG to write (default: <root>/gamma_sweep_comparison.png)")
    parser.add_argument("--per-case", action="store_true",
                        help="One PNG per fit and one comparison PNG per case (see the module docstring).")
    parser.add_argument("--panel-px", type=int, default=260)
    args = parser.parse_args()

    model_npz = np.load(args.model, allow_pickle=True)
    mean = model_npz["mean"].reshape(-1, 3)
    factor = VIEW_SIZE_MM / np.linalg.norm(mean.max(axis=0) - mean.min(axis=0))
    lmk = {name: point * factor for name, point in model_landmarks(model_npz).items()}

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
        target = scaled(load_vtp(os.path.join(case_dir, "rigid_alignment.vtp")), factor)
        recons = [(label, scaled(load_vtp(path), factor)) for label, path in fits]
        cases.append((case_name, case_dir, target, recons))

    if args.per_case:
        render_per_case(cases, lmk, args.panel_px)
        return

    columns = 1 + max(len(recons) for _, _, _, recons in cases)
    all_meshes = [target for _, _, target, _ in cases] + [mesh for _, _, _, recons in cases for _, mesh in recons]
    camera = view_camera(lmk, all_meshes, "frontal")

    rows = []
    for case_name, _, target, recons in cases:
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
