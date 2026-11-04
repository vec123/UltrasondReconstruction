"""Show every subject as one row, its sessions ordered left to right by
gestational age - a trajectory view. A subject with only one fused target gets
a one-panel row (a lone data point, not a trajectory, but kept so nothing is
left out silently).

Each cell is that session's fused target, frontal view only. One camera per
ROW (built from the subject's earliest session's own landmarks, framing every
one of its sessions' meshes), so a subject's own growth across weeks stays
visually comparable; cameras differ between rows/subjects.

Run with:
    python scripts/inspection/fused_target_trajectories.py \
        --root /home/victor/Escritorio/Data/fused_scan_processing \
        --output /home/victor/Escritorio/Data/fused_scan_processing/fused_target_trajectories.png
"""

import argparse
import re
from collections import defaultdict

import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from scripts.inspection.fused_target_ranking import (  # noqa: E402
    find_targets, reference_landmarks, session_scale, session_scan_count,
)
from src.shared.vtk.render import SCAN_COLOR, view_camera, write_grid  # noqa: E402
from src.shared.vtk.io import load_vtp  # noqa: E402

DATASET_ORDER = ("Casos_Mar", "Estudio_Dexeus", "Casos_Maternitat")
SHORT_NAME = {"Casos_Mar": "Mar", "Estudio_Dexeus": "Dex", "Casos_Maternitat": "Mat"}


def parse_week(week):
    """Gestational age in weeks as a float, for sorting; weeks that aren't a plain
    number (e.g. Casos_Mar's stray "o") sort last, after every real week."""
    match = re.match(r"(\d+(\.\d+)?)", week)
    return float(match.group(1)) if match else float("inf")


def rescale_about_centroid(mesh, factor):
    """mesh's points scaled by `factor` about their own centroid, in place - changes
    size only, not position or orientation. See session_scale for why/when."""
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64)
    centroid = points.mean(axis=0)
    new_points = centroid + (points - centroid) * factor
    mesh.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(new_points, dtype=np.float32), deep=True))
    return mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", nargs="+", required=True,
                        help="One or more fused_scan_processing dataset roots to search (recursively).")
    parser.add_argument("--output", required=True, help="PNG to write.")
    parser.add_argument("--panel-px", type=int, default=220, help="Width and height of each panel in pixels.")
    parser.add_argument("--orig-scale", action="store_true",
                        help="Undo each session's own align_to_template scale factor (about each mesh's "
                             "centroid; rotation/translation untouched), so sizes reflect the original scan's "
                             "mm scale relative to the template instead of the template's own fixed size - "
                             "see session_scale. A session with no saved transform is left at template scale.")
    args = parser.parse_args()

    targets = find_targets(args.root)
    all_subjects = defaultdict(list)  # (dataset, case) -> [(week, fused_path, ref_path), ...]
    for label, dataset, case, week, fused_path, ref_path in targets:
        all_subjects[(dataset, case)].append((week, fused_path, ref_path))
    subjects = {key: sorted(sessions, key=lambda s: parse_week(s[0]))
               for key, sessions in all_subjects.items()}

    def row_order(key):
        dataset, case = key
        order = DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else len(DATASET_ORDER)
        return order, int(case) if case.isdigit() else case
    rows_keys = sorted(subjects, key=row_order)
    max_cols = max(len(sessions) for sessions in subjects.values())
    n_multi = sum(1 for s in subjects.values() if len(s) > 1)
    print(f"{len(rows_keys)} subjects total ({n_multi} with more than one fused target, "
         f"{len(rows_keys) - n_multi} as a single point); up to {max_cols} sessions each")

    grid_rows = []
    for dataset, case in rows_keys:
        sessions = subjects[(dataset, case)]
        meshes = [load_vtp(fused_path) for _, fused_path, _ in sessions]
        ref_mesh = load_vtp(sessions[0][2])
        if args.orig_scale:
            for (week, fused_path, _), mesh in zip(sessions, meshes):
                scale = session_scale(fused_path)
                if scale:
                    rescale_about_centroid(mesh, 1.0 / scale)
            ref_scale = session_scale(sessions[0][1])
            if ref_scale:
                rescale_about_centroid(ref_mesh, 1.0 / ref_scale)
        lmk = reference_landmarks(ref_mesh)
        camera = view_camera(lmk, meshes, "frontal")
        panels = []
        for i, (week, _, _) in enumerate(sessions):
            title = f"{SHORT_NAME.get(dataset, dataset)}/{case}" if i == 0 else ""
            title += f" {week.replace(' semanas', 'w')}" if week else ""
            n_scans = session_scan_count(dataset, case, week)
            panels.append((f"{title.strip()}\n{n_scans} scans" if n_scans is not None else title.strip(),
                           meshes[i], SCAN_COLOR, []))
        grid_rows.append(([camera] * max_cols, panels))

    scale_note = "; each session's own align_to_template scale factor undone (real relative size, not " \
                "the template's fixed size)" if args.orig_scale else ""
    write_grid(grid_rows, max_cols,
               f"{len(rows_keys)} subjects ({n_multi} trajectories, {len(rows_keys) - n_multi} single points), "
               f"ordered by gestational age "
               f"(one camera per row, framing all of that subject's sessions){scale_note}",
               args.output, args.panel_px)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
