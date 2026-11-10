"""Show every reconstruction of a cohort auto-decoder run, ordered by how far its
code lies from the code mean in the full code space: closest first, farthest last.

The first row holds the references: the template itself (where every flow starts),
the face decoded from the code mean and the face of the zero code (the
auto-decoder's prior), the last unless the codes are centred and the two coincide.
With a shared flow (field.shared) the zero code is the atlas, the shared flow's output;
it is then always shown, under that name.
The members start on the second row. Each member tile is one
member: its reconstruction from the front, the left and the right, then its fused
target from the front, all with one common camera. The titles give the rank, the
code distance to the mean and the mean vertex distance (mm) of the reconstruction
to the mean code's face.

Run with:
    python scripts/studies/code_distance_ranking.py \
        --apply-config configs/lddmm/cohort/apply/cohort_autodecoder_casos_mar_fieldweighted_nopos_apply.yaml
"""

import argparse
import json
import os

import numpy as np
import torch
from vtk.util.numpy_support import vtk_to_numpy

from scripts.studies.code_pca_modes import decode, with_points  # noqa: E402
from scripts.studies.code_pca_views import load_codes, similarity_map, target_path, template_landmarks  # noqa: E402
from src.shared.vtk.render import MESH_COLOR, SCAN_COLOR, view_camera, write_grid  # noqa: E402
from src.models.gpmm.model.model_memory_io import load_model_file  # noqa: E402
from src.shared.vtk.io import load_vtp  # noqa: E402

VIEWS = ("frontal", "left", "right")


def has_shared_flow(run):
    """True when the run's field has a shared flow (field.shared): its zero code is then the
    atlas, the output of the shared flow (src/models/lddmm/resnet_flow/fields/composite/shared.py)."""
    with open(os.path.join(run, "config.json")) as f:
        return json.load(f).get("field", {}).get("shared") is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply-config", required=True, help="The run's cohort apply .yaml.")
    parser.add_argument("--output", default=None, help="PNG to write (default: <run>/code_distance_ranking.png).")
    parser.add_argument("--members-per-row", type=int, default=3, help="Member tiles side by side.")
    parser.add_argument("--panel-px", type=int, default=200, help="Width and height of each panel in pixels.")
    parser.add_argument("--similarity", default=None,
                        help="similarity.json mapping the run's world frame to mm (a shape-model-frame run, "
                             "see scripts/tools/target_to_ssm_frame.py); default: world is mm already.")
    parser.add_argument("--model", default="assets/shape_models/baby/baby_pca.npz",
                        help="landmark fallback for members with no eval (see template_landmarks)")
    args = parser.parse_args()
    to_mm = similarity_map(args.similarity)
    model = load_model_file(args.model)

    # Only the code mean (step 0) is needed; decode returns it once per component.
    # The zero code is decoded too: the auto-decoder's prior, which the mean need not be.
    points, template, run, _, _ = decode(args.apply_config, (0.0,),
                                         extra={"zero": lambda codes: np.zeros(codes.shape[1])})
    points = {key: to_mm(p) for key, p in points.items()}
    mean_points = points[(0, 0.0)]
    template_points = to_mm(vtk_to_numpy(template.GetPoints().GetData()))
    names, codes = load_codes(run)
    distance = np.linalg.norm(codes - codes.mean(axis=0), axis=1)
    order = np.argsort(distance)
    zero_distance = np.linalg.norm(codes.mean(axis=0))

    recons = {i: with_points(template, to_mm(vtk_to_numpy(load_vtp(
        os.path.join(run, "apply", names[i], "world", "deformed.vtp")).GetPoints().GetData())))
        for i in order}
    lmk = template_landmarks(run, names[0], to_mm, model=model)
    # The template is framed too: it need not lie inside the members' extent.
    cameras = [view_camera(lmk, list(recons.values()) + [with_points(template, template_points)], view)
               for view in VIEWS]
    frontal = cameras[0]

    def reference_tile(titles, reference_points):
        return [(camera, (title, with_points(template, reference_points), MESH_COLOR, []))
                for camera, title in zip(cameras, titles)]

    tiles = [reference_tile(("Template",) + VIEWS[1:], template_points),
             reference_tile(("Code mean (rank 0)",) + VIEWS[1:], mean_points)]
    # Skipped when the codes are centred, as the zero code is then the mean. With a shared
    # flow the zero code is the atlas, the shared flow's output: always shown, and named so.
    shared = has_shared_flow(run)
    if shared or zero_distance > 1e-3 * np.median(distance):
        zero_shift = np.linalg.norm(points["zero"] - mean_points, axis=1).mean()
        zero_title = "Atlas (code 0)" if shared else "Code 0 (zero vector)"
        tiles.append(reference_tile((zero_title, f"left  code d={zero_distance:.2f}",
                                     f"right  {zero_shift:.2f} mm"), points["zero"]))
    tiles += [[]] * (-len(tiles) % args.members_per_row)  # the members start on a new row
    for rank, i in enumerate(order, start=1):
        shift = np.linalg.norm(vtk_to_numpy(recons[i].GetPoints().GetData()) - mean_points, axis=1).mean()
        # The titles spread over the three views, as one would not fit a panel.
        titles = (f"#{rank} {names[i].replace('_semanas', ' wk')}", f"left  code d={distance[i]:.2f}",
                  f"right  {shift:.2f} mm")
        tile = [(camera, (title, recons[i], MESH_COLOR, [])) for camera, title in zip(cameras, titles)]
        target = load_vtp(target_path(run, names[i]))
        tile.append((frontal, ("target", target, SCAN_COLOR, [])))
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
               f"{os.path.basename(run)}: row 1 template, code mean, "
               f"{'atlas (code 0, shared-flow output)' if shared else 'code 0'}; then recon front/left/right "
               f"+ target, ranked by code d = L2 distance of "
               f"the {codes.shape[1]}-d code to the code mean (median {np.median(distance):.2f}); mm = mean vertex "
               f"distance to the code-mean face (rank 0)",
               out_path, args.panel_px)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
        main()
