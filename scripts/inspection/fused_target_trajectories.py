"""Show every subject with more than one fused scan target as one row, its
sessions ordered left to right by gestational age - a trajectory view.

Each cell is that session's fused target, frontal view only. One camera per
ROW (built from the subject's earliest session's own landmarks, framing every
one of its sessions' meshes), so a subject's own growth across weeks stays
visually comparable; cameras differ between rows/subjects.

A subject with only one fused target has no trajectory and is left out - see
fused_target_ranking.py for a view of every single target instead.

Run with:
    python scripts/inspection/fused_target_trajectories.py \
        --root /home/victor/Escritorio/Data/fused_scan_processing \
        --output /home/victor/Escritorio/Data/fused_scan_processing/fused_target_trajectories.png
"""

import argparse
import re
from collections import defaultdict

from scripts.inspection.fused_target_ranking import find_targets, reference_landmarks  # noqa: E402
from scripts.inspection.render_frontal_views import SCAN_COLOR, view_camera, write_grid  # noqa: E402
from src.vtk.io import load_vtp  # noqa: E402

DATASET_ORDER = ("Casos_Mar", "Estudio_Dexeus", "Casos_Maternitat")
SHORT_NAME = {"Casos_Mar": "Mar", "Estudio_Dexeus": "Dex", "Casos_Maternitat": "Mat"}


def parse_week(week):
    """Gestational age in weeks as a float, for sorting; weeks that aren't a plain
    number (e.g. Casos_Mar's stray "o") sort last, after every real week."""
    match = re.match(r"(\d+(\.\d+)?)", week)
    return float(match.group(1)) if match else float("inf")


def dataset_of(fused_path, roots):
    for root in roots:
        if fused_path.startswith(root):
            return fused_path[len(root):].strip("/").split("/")[0]
    return "?"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", nargs="+", required=True,
                        help="One or more fused_scan_processing dataset roots to search (recursively).")
    parser.add_argument("--output", required=True, help="PNG to write.")
    parser.add_argument("--panel-px", type=int, default=220, help="Width and height of each panel in pixels.")
    args = parser.parse_args()

    targets = find_targets(args.root)
    all_subjects = defaultdict(list)  # (dataset, case) -> [(week, fused_path, ref_path), ...]
    for label, fused_path, ref_path in targets:
        dataset = dataset_of(fused_path, args.root)
        parts = label.split("/")
        case, week = (parts[1], parts[2]) if len(parts) == 3 else (parts[1], "")
        all_subjects[(dataset, case)].append((week, fused_path, ref_path))
    subjects = {key: sorted(sessions, key=lambda s: parse_week(s[0]))
               for key, sessions in all_subjects.items() if len(sessions) > 1}

    def row_order(key):
        dataset, case = key
        order = DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else len(DATASET_ORDER)
        return order, int(case) if case.isdigit() else case
    rows_keys = sorted(subjects, key=row_order)
    max_cols = max(len(sessions) for sessions in subjects.values())
    print(f"{len(rows_keys)} of {len(all_subjects)} subjects have more than one fused target; "
         f"up to {max_cols} sessions each")

    grid_rows = []
    for dataset, case in rows_keys:
        sessions = subjects[(dataset, case)]
        meshes = [load_vtp(fused_path) for _, fused_path, _ in sessions]
        lmk = reference_landmarks(load_vtp(sessions[0][2]))
        camera = view_camera(lmk, meshes, "frontal")
        panels = []
        for i, (week, _, _) in enumerate(sessions):
            title = f"{SHORT_NAME.get(dataset, dataset)}/{case}" if i == 0 else ""
            title += f" {week.replace(' semanas', 'w')}" if week else ""
            panels.append((title.strip(), meshes[i], SCAN_COLOR, []))
        grid_rows.append(([camera] * max_cols, panels))

    write_grid(grid_rows, max_cols,
               f"{len(rows_keys)} subjects with more than one fused target, ordered by gestational age "
               f"(one camera per row, framing all of that subject's sessions)",
               args.output, args.panel_px)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
