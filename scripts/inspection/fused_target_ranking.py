"""Show every fused scan target found under one or more fused_scan_processing
dataset roots, ranked by mean fusion support (highest/best first), then by
point count (most first) as a tie-break.

Each tile is one target, frontal/left/right, coloured as scan data (not a
model mesh). No trained run or reconstruction is involved - this looks at
the fused targets themselves, before any training.

The camera for each target is built from its OWN reference.vtp (scan_fusion's
per-session copy of the reference mesh, same frame as fused_scan.vtp, same
heatmap_{label} point fields as the SSM templates), so no external landmarks
file is needed.

Run with:
    python scripts/inspection/fused_target_ranking.py \
        --root /home/victor/Escritorio/Data/fused_scan_processing \
        --output /home/victor/Escritorio/Data/fused_scan_processing/fused_target_ranking.png
"""

import argparse
import glob
import os

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from scripts.inspection.render_frontal_views import SCAN_COLOR, face_frame, view_camera, write_grid  # noqa: E402
from src.vtk.io import load_vtp  # noqa: E402

HEATMAP_LABELS = ("n", "enR", "enL", "exR", "exL", "aR", "aL", "acR", "acL", "sn", "ls", "pg", "prn")
VIEWS = ("frontal", "left", "right")

# The raw per-session .nrrd scans each fused target was built from - not recorded
# anywhere under fused_scan_processing itself, so session_scan_count looks them up
# here (same dataset/case/week layout gen_fused_scan_targets.yaml reads CASES_ROOT from).
INFERENCE_ROOT = "/home/victor/Projects/UltrasoundLandmarkPredictor/results/inference"

# Casos_Mar's fused targets were moved to fused_scan_processing (see the move to
# /home/victor/Escritorio/Data/fused_scan_processing), but only the fused_scan_ssm/
# folder itself - the rest of each session's tree (aligned_to_template/, ...) was
# never moved and still lives here. session_scale falls back to this root when a
# Casos_Mar session's transforms aren't found alongside its (moved) fused target.
CASOS_MAR_OUTPUTS_ROOT = "/home/victor/Projects/UltrasondReconstruction/outputs/Casos_Mar"


def session_scan_count(dataset, case, week):
    """Number of raw .nrrd scans the (dataset, case, week) session was fused from, or
    None when the session folder isn't found under INFERENCE_ROOT (e.g. moved/renamed)."""
    session_dir = os.path.join(INFERENCE_ROOT, dataset, case, week) if week \
        else os.path.join(INFERENCE_ROOT, dataset, case)
    if not os.path.isdir(session_dir):
        return None
    return len(glob.glob(os.path.join(session_dir, "*.nrrd")))


def session_scale(fused_path):
    """Median align_to_template scale factor (s of x_template = s*R*x_raw + t, composed
    with align_to_reference's own scale - see SimilarityTransform.save) over every scan
    of this session, from its own aligned_to_template/transforms/*.npz files.

    align_to_template fits each scan's landmarks onto ONE FIXED template
    (assets/templates/fetus_mean.vtp), so every session - at every gestational week - gets
    pulled toward that template's own size: fused_scan.vtp is in the TEMPLATE's scale,
    not the original scan's mm scale. Dividing a session's points by this factor (about
    their own centroid, so only size changes) undoes that and restores the session's
    size relative to the template, letting different weeks be compared on a true scale.
    Median over the session's scans, not just one, since each scan gets its own
    (slightly different) landmark-based fit. None when no transform is found.
    """
    session_dir = os.path.dirname(os.path.dirname(fused_path))
    transform_files = glob.glob(os.path.join(session_dir, "aligned_to_template", "transforms", "*.npz"))
    if not transform_files and "Casos_Mar" in fused_path.split(os.sep):
        # case/week are the two path components right before fused_scan_ssm/.
        case, week = fused_path.split(os.sep)[-4:-2]
        transform_files = glob.glob(os.path.join(CASOS_MAR_OUTPUTS_ROOT, case, week,
                                                  "aligned_to_template", "transforms", "*.npz"))
    if not transform_files:
        return None
    return float(np.median([np.load(f)["s"] for f in transform_files]))


def find_targets(roots):
    """[(label, dataset, case, week, fused_scan.vtp path, reference.vtp path), ...] under
    the given roots.

    label is the path relative to its root, e.g. "Estudio_Dexeus/Dex-76/26 semanas";
    dataset/case/week are its parts, week "" for a dataset without week folders
    (e.g. Casos_Maternitat) - the same triple session_scan_count takes.
    """
    targets = []
    for root in roots:
        for fused_path in sorted(glob.glob(os.path.join(root, "**", "fused_scan_ssm", "fused_scan.vtp"),
                                           recursive=True)):
            session_dir = os.path.dirname(os.path.dirname(fused_path))
            label = os.path.relpath(session_dir, root)
            parts = label.split(os.sep)
            dataset, case, week = (parts[0], parts[1], parts[2]) if len(parts) == 3 else (parts[0], parts[1], "")
            ref_path = os.path.join(os.path.dirname(fused_path), "reference.vtp")
            if os.path.isfile(ref_path):
                targets.append((label, dataset, case, week, fused_path, ref_path))
    return targets


def reference_landmarks(reference_mesh):
    """{label: point} from reference.vtp's own heatmap_{label} point fields (argmax vertex)."""
    points = vtk_to_numpy(reference_mesh.GetPoints().GetData())
    point_data = reference_mesh.GetPointData()
    lmk = {}
    for label in HEATMAP_LABELS:
        array = point_data.GetArray(f"heatmap_{label}")
        if array is not None:
            lmk[label] = points[int(np.argmax(vtk_to_numpy(array)))]
    return lmk


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", nargs="+", required=True,
                        help="One or more fused_scan_processing dataset roots to search (recursively).")
    parser.add_argument("--output", required=True, help="PNG to write.")
    parser.add_argument("--per-row", type=int, default=5, help="Targets side by side.")
    parser.add_argument("--panel-px", type=int, default=140, help="Width and height of each panel in pixels.")
    args = parser.parse_args()

    targets = find_targets(args.root)
    if not targets:
        raise FileNotFoundError(f"No fused_scan_ssm/fused_scan.vtp found under {args.root}")

    loaded = []
    for label, dataset, case, week, fused_path, ref_path in targets:
        fused = load_vtp(fused_path)
        support = vtk_to_numpy(fused.GetPointData().GetArray("support"))
        loaded.append({
            "label": label, "fused": fused, "reference": load_vtp(ref_path),
            "mean_support": float(support.mean()), "n_points": fused.GetNumberOfPoints(),
            "n_scans": session_scan_count(dataset, case, week),
        })
    # Highest support first, most points first among ties.
    loaded.sort(key=lambda t: (-t["mean_support"], -t["n_points"]))

    rows = []
    for rank, t in enumerate(loaded, start=1):
        lmk = reference_landmarks(t["reference"])
        cameras = [view_camera(lmk, [t["fused"]], view) for view in VIEWS]
        short_label = t["label"].replace("Estudio_Dexeus", "Dex").replace("Casos_Maternitat", "Mat") \
            .replace("Casos_Mar", "Mar").replace(" semanas", "w")
        n_label = f"n={t['n_points']} ({t['n_scans']} scans)" if t["n_scans"] is not None else f"n={t['n_points']}"
        titles = (f"#{rank} {short_label}", f"support={t['mean_support']:.2f}", n_label)
        row = [(title, t["fused"], SCAN_COLOR, []) for title in titles]
        rows.append((cameras, row))

    per_row = args.per_row
    grid_rows = []
    for start in range(0, len(rows), per_row):
        cameras_row, panels_row = [], []
        for cameras, panels in rows[start:start + per_row]:
            cameras_row += cameras
            panels_row += panels
        grid_rows.append((cameras_row, panels_row))

    write_grid(grid_rows, per_row * 3,
               f"{len(loaded)} fused targets, ranked by mean support (then point count); "
               f"support range {loaded[-1]['mean_support']:.2f}-{loaded[0]['mean_support']:.2f}",
               args.output, args.panel_px)
    print(f"Saved {args.output} ({len(loaded)} targets)")


if __name__ == "__main__":
    main()
