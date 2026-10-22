"""Show every reconstruction of a cohort auto-decoder run, ordered by how far its
code lies from the code mean in the full code space: closest first, farthest last.

The first tile is the face decoded from the code mean itself, the second the face
of the zero code (the auto-decoder's prior), unless the codes are centred and the
two coincide. Each tile is one
member: its reconstruction from the front, the left and the right, then its fused
target from the front, all with one common camera. The titles give the rank, the
code distance to the mean and the mean vertex distance (mm) of the reconstruction
to the mean code's face.

Run with:
    python scripts/inspection/code_distance_ranking.py \
        --apply-config configs/resnet_lddmm/cohort_autodecoder_casos_mar_fieldweighted_nopos_apply.yaml
"""

import argparse
import os
import sys

import numpy as np
import torch
from vtk.util.numpy_support import vtk_to_numpy

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from code_pca_modes import decode, with_points  # noqa: E402
from code_pca_views import load_codes, target_path, template_landmarks  # noqa: E402
from render_frontal_views import MESH_COLOR, SCAN_COLOR, view_camera, write_grid  # noqa: E402
from src.vtk.io import load_vtp  # noqa: E402

VIEWS = ("frontal", "left", "right")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply-config", required=True, help="The run's cohort apply .yaml.")
    parser.add_argument("--output", default=None, help="PNG to write (default: <run>/code_distance_ranking.png).")
    parser.add_argument("--members-per-row", type=int, default=3, help="Member tiles side by side.")
    parser.add_argument("--panel-px", type=int, default=200, help="Width and height of each panel in pixels.")
    args = parser.parse_args()

    # Only the code mean (step 0) is needed; decode returns it once per component.
    # The zero code is decoded too: the auto-decoder's prior, which the mean need not be.
    points, template, run, _, _ = decode(args.apply_config, (0.0,),
                                         extra={"zero": lambda codes: np.zeros(codes.shape[1])})
    mean_points = points[(0, 0.0)]
    names, codes = load_codes(run)
    distance = np.linalg.norm(codes - codes.mean(axis=0), axis=1)
    order = np.argsort(distance)
    zero_distance = np.linalg.norm(codes.mean(axis=0))

    recons = {i: load_vtp(os.path.join(run, "apply", names[i], "world", "deformed.vtp")) for i in order}
    lmk = template_landmarks(run, names[0])
    cameras = [view_camera(lmk, list(recons.values()), view) for view in VIEWS]
    frontal = cameras[0]

    tiles = [[(frontal, ("Code mean (rank 0)", with_points(template, mean_points), MESH_COLOR, []))]
             + [(camera, (view, with_points(template, mean_points), MESH_COLOR, []))
                for camera, view in zip(cameras[1:], VIEWS[1:])]]
    # Skipped when the codes are centred, as the zero code is then the mean.
    if zero_distance > 1e-3 * np.median(distance):
        zero_shift = np.linalg.norm(points["zero"] - mean_points, axis=1).mean()
        titles = ("Code 0 (zero vector)", f"left  code d={zero_distance:.2f}", f"right  {zero_shift:.2f} mm")
        tiles.append([(camera, (title, with_points(template, points["zero"]), MESH_COLOR, []))
                      for camera, title in zip(cameras, titles)])
    for rank, i in enumerate(order, start=1):
        shift = np.linalg.norm(vtk_to_numpy(recons[i].GetPoints().GetData()) - mean_points, axis=1).mean()
        # The titles spread over the three views, as one would not fit a panel.
        titles = (f"#{rank} {names[i].replace('_semanas', ' wk')}", f"left  code d={distance[i]:.2f}",
                  f"right  {shift:.2f} mm")
        tile = [(camera, (title, recons[i], MESH_COLOR, [])) for camera, title in zip(cameras, titles)]
        tile.append((frontal, ("target", load_vtp(target_path(run, names[i])), SCAN_COLOR, [])))
        tiles.append(tile)

    rows = []
    per_row = args.members_per_row
    for start in range(0, len(tiles), per_row):
        panels = []
        for tile in tiles[start:start + per_row]:
            panels += [panel for _, panel in tile] + [None] * (4 - len(tile))
        rows.append(((cameras + [frontal]) * per_row, panels))  # recon views, then the target from the front

    out_path = args.output or os.path.join(run, "code_distance_ranking.png")
    write_grid(rows, per_row * 4,
               f"{os.path.basename(run)}: recon front/left/right + target, ranked by code d = L2 distance of "
               f"the {codes.shape[1]}-d code to the code mean (median {np.median(distance):.2f}); mm = mean vertex "
               f"distance to the code-mean face (rank 0)",
               out_path, args.panel_px)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
        main()
