"""Render one inspection PNG per finished case/week of a batch run, into <root>/views/.

Each PNG has, top to bottom:
  * every scan of that case/week, frontal (wrapped over as many rows as needed);
  * one frontal row of results in processing order:
        high-confidence union (with landmarks) | high-confidence union | fetus PCA
        | fetus PCA + Matern32 | fused target | baby PCA | baby PCA + Matern32
        | LDDMM (kinetic 2000, landmarks) | LDDMM (kinetic 5000, no landmarks) | cohort LDDMM
        | cohort LDDMM (field-weighted point-to-plane, as the pair runs) | the same with a 4³ code
        grid | with an 8³ code grid | with a position-independent code (no grid);
  * the fitted models (everything after the two unions) seen from the patient's
    left, then from the right, each model in the same column as its frontal panel.
(both fetus fits are fitted to the high-confidence union, and the fused target
is built from the fetus PCA fit; the baby fits and LDDMM are fitted to it; the
cohort LDDMMs are auto-decoder runs over all fused targets, applied with this
case's code, and are left "(missing)" for a case/week they were not trained on).
Every row of one view shares that view's camera, so panels compare at the same
scale. Red spheres are predicted landmarks: each scan's own, and the reference
scan's on the first high-confidence union.

Every mesh is read from the reference-scan frame: the scans from
align_to_reference's output, everything else from the `*_w_distance.vtp` the
eval stages write (the per-model fit outputs themselves are in a normalized
frame). The scans are the uncut volume surfaces, so the camera's near plane
cuts away the tissue between it and the face: just in front of the nose tip
for the frontal view, just outside the reconstructions for the side views.

The views are set by the reference landmarks' face: left-right from exR -> exL,
up from pg -> n, and the facing sign from prn.

Usage:
    python scripts/inspection/render_frontal_views.py --root outputs/Casos_Mar
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from src.shared.vtk.io import load_vtp  # noqa: E402

FUSED_EVAL = "fetus_ssm/{case}/{week}/end_to_end/fused_scan_ssm/eval"
# scan_consistency's union of the scans' high-confidence (>= 0.8) parts, as the fused eval's target
HIGH_CONFIDENCE = FUSED_EVAL + "/target_w_distance.vtp"
RECONSTRUCTIONS = [
    ("Fetus PCA", "fetus_ssm/{case}/{week}/end_to_end/init_pca/eval_map/recon_w_distance.vtp"),
    ("Fetus PCA + Matern32", "fetus_ssm/{case}/{week}/end_to_end/init_pca_sum_matern32/eval_map/recon_w_distance.vtp"),
    ("Fused target", FUSED_EVAL + "/recon_w_distance.vtp"),
    ("Baby PCA", "baby_ssm/{case}/{week}/end_to_end/pca/eval_map/recon_w_distance.vtp"),
    ("Baby PCA + Matern32", "baby_ssm/{case}/{week}/end_to_end/pca_sum_matern32/eval_map/recon_w_distance.vtp"),
    ("LDDMM recon", "resnet_lddmm_baby_mean/{case}/{week}/apply/eval/recon_w_distance.vtp"),
    ("LDDMM kin5000 no lmk", "resnet_lddmm_baby_mean_kin5000_nolmk_gpu/{case}/{week}/apply/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw", "resnet_lddmm_cohort_autodecoder_fieldweighted/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw g4", "resnet_lddmm_cohort_autodecoder_fieldweighted_g4/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw g8", "resnet_lddmm_cohort_autodecoder_fieldweighted_g8/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw no grid", "resnet_lddmm_cohort_autodecoder_fieldweighted_nopos/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw no grid kin8000", "resnet_lddmm_cohort_autodecoder_fieldweighted_nopos_kin8000/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw direct g4 C8", "resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g4c8/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw direct g4 C8 location weighted", "resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g4c8_dist/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw direct g8 C2 location weighted", "resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g8c2_dist/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw direct g8 C2 location weighted kin8000", "resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g8c2_dist_kin8000/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw direct g8 C2 location weighted far x100", "resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g8c2_dist_far100/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw direct g8 C2 location weighted kin20000", "resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g8c2_dist_kin20000/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw direct g8 C2 location weighted kin8000 cherry-tree", "resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g8c2_dist_kin8000_cherry/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw direct g8 C2 location weighted kin10000 cherry-tree", "resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g8c2_dist_kin10000_cherry/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw direct g8 C2 location weighted kinetic field", "resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g8c2_dist_kinfield/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw direct g8 C2 location weighted stationary field", "resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g8c2_dist_svf/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("SSM posterior (localized PCA) of the stationary field LDDMM", "resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g8c2_dist_svf/apply/{case}_{week_}/posterior_baby-pca-times-gaussian_pose_t2r_cutoff1.5mm_spacing3mm_sigma1mm/eval/recon_w_distance.vtp"),
    ("SSM posterior (PCA + Matern32) of the stationary field LDDMM", "resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g8c2_dist_svf/apply/{case}_{week_}/posterior_baby-pca-sum-matern32_pose_t2r_cutoff1.5mm_spacing3mm_sigma1mm/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw direct g8 C2 location weighted kin8000 cherry-tree stationary field", "resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g8c2_dist_kin8000_cherry_svf/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw direct g8 C2 location weighted stationary field kinetic field", "resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g8c2_dist_svf_kinfield/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw direct g8 C2 location weighted kin8000 cherry-tree stationary field kinetic field", "resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g8c2_dist_kin8000_cherry_svf_kinfield/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("Cohort LDDMM fw direct g16 C2 location weighted kin2000 cherry-tree stationary field kinetic field 14000 6x256", "resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g16c2_dist_kin2000_cherry_svf_kinfield14000_mlp6x256/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
    ("BabySSM flow LayerNorm, codes + shared LayerNorm fitted, cherry-tree", "../stage2_flow/babyssm-sobol1000-mlp6x256-layernorm-kin200_free-layernorm-shared_fieldchange200_anchor0.01_code128_codereg1.67_edgeweight3sp-template2scan_cherry30/apply/{case}_{week_}/eval/recon_w_distance.vtp"),
]
REFERENCE_LANDMARKS = "resnet_lddmm_baby_mean/{case}/{week}/apply/eval/reference_landmarks.vtp"
ALIGNED_LANDMARKS = "{case}/{week}/aligned/aligned_landmarks/*_landmarks.csv"
ALIGNED_SCAN = "{case}/{week}/aligned/aligned_shapes_vtp/{scan}.vtp"

MESH_COLOR = (0.93, 0.80, 0.70)
SCAN_COLOR = (0.55, 0.70, 0.85)
LANDMARK_COLOR = (0.84, 0.15, 0.16)
NEAR_PLANE_MM = 10.0  # how far in front of the nose tip the camera starts seeing


def finished_pairs(root):
    """(case, week) pairs whose pair LDDMM eval exists, i.e. whose batch run finished."""
    pattern = os.path.join(root, dict(RECONSTRUCTIONS)["LDDMM recon"].format(case="*", week="*"))
    pairs = []
    for path in glob.glob(pattern):
        rel = os.path.relpath(path, os.path.join(root, "resnet_lddmm_baby_mean")).split(os.sep)
        pairs.append((rel[0], rel[1]))
    return sorted(pairs, key=lambda p: (int(p[0]) if p[0].isdigit() else p[0], p[1]))


def read_landmarks(path):
    poly = load_vtp(path)
    points = vtk_to_numpy(poly.GetPoints().GetData())
    labels = poly.GetPointData().GetAbstractArray("label")
    return {labels.GetValue(i): points[i] for i in range(poly.GetNumberOfPoints())}


def read_csv_landmarks(path):
    df = pd.read_csv(path)
    return {label: df.loc[i, ["x", "y", "z"]].to_numpy(float) for i, label in enumerate(df["label"])}


def scan_landmarks(root, fmt):
    """{scan name: its landmarks after align_to_reference}, in the reference-scan frame."""
    return {os.path.basename(path)[: -len("_landmarks.csv")]: read_csv_landmarks(path)
            for path in glob.glob(os.path.join(root, ALIGNED_LANDMARKS.format(**fmt)))}


def reference_scan_name(scan_lmks, ref_lmk):
    """The scan whose aligned landmarks are the eval's reference landmarks (it is aligned onto itself)."""
    def dist(points):
        return np.mean([np.linalg.norm(points[k] - v) for k, v in ref_lmk.items() if k in points])
    return min(scan_lmks, key=lambda name: dist(scan_lmks[name]))


def face_frame(lmk):
    """Landmark centroid and orthonormal (right, up, forward) axes of the face."""
    center = np.mean(list(lmk.values()), axis=0)
    lr = lmk["exL"] - lmk["exR"]
    up = lmk["n"] - lmk["pg"]
    forward = np.cross(lr, up)
    forward /= np.linalg.norm(forward)
    if np.dot(lmk["prn"] - center, forward) < 0:
        forward = -forward
    up -= np.dot(up, forward) * forward
    up /= np.linalg.norm(up)
    right = np.cross(up, forward)
    return center, np.stack([right, up, forward])


def view_camera(lmk, meshes, view):
    """Parallel-projection camera on `view` ("frontal", "left" or "right"), framing
    `meshes` and seeing nothing more than NEAR_PLANE_MM in front of the face: the
    nose tip for the frontal view, the outermost mesh point on that side otherwise."""
    center, axes = face_frame(lmk)
    right, up, forward = axes
    coords = np.concatenate([(vtk_to_numpy(m.GetPoints().GetData()) - center) @ axes.T for m in meshes])
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    mid = (lo + hi) / 2
    # "right" is the screen-right axis of the frontal view, i.e. the patient's left.
    if view == "frontal":
        toward, across = forward, 0
        near = np.dot(lmk["prn"] - center, forward)
    else:
        toward, across = (right, 2) if view == "left" else (-right, 2)
        near = coords[:, 0].max() if view == "left" else -coords[:, 0].min()
    focal = center + mid[across] * axes[across] + mid[1] * up
    distance = 500.0
    camera = vtk.vtkCamera()
    camera.ParallelProjectionOn()
    camera.SetFocalPoint(*focal)
    camera.SetPosition(*(focal + toward * distance))
    camera.SetViewUp(*up)
    camera.SetParallelScale(0.55 * max(hi[across] - lo[across], hi[1] - lo[1]))
    camera.SetClippingRange(distance - near - NEAR_PLANE_MM, distance + 1000.0)
    return camera


def mesh_actor(poly, color):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetSpecular(0.2)
    if poly.GetNumberOfPolys() == 0:
        actor.GetProperty().SetRepresentationToPoints()
        actor.GetProperty().SetPointSize(3)
    return actor


def landmarks_actor(points, color, radius=1.2):
    """Spheres at `points` (an (N, 3) array)."""
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk.vtkPoints())
    for p in points:
        poly.GetPoints().InsertNextPoint(*p)
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    glyphs = vtk.vtkGlyph3D()
    glyphs.SetInputData(poly)
    glyphs.SetSourceConnection(sphere.GetOutputPort())
    glyphs.ScalingOff()
    glyphs.Update()
    return mesh_actor(glyphs.GetOutput(), color)


def text_actor(text, size, x, y):
    """Text at (x, y) in its renderer's normalized viewport coordinates."""
    actor = vtk.vtkTextActor()
    actor.SetInput(text)
    actor.GetTextProperty().SetFontSize(size)
    actor.GetTextProperty().SetColor(0.1, 0.1, 0.1)
    actor.GetTextProperty().SetVerticalJustificationToTop()
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.GetPositionCoordinate().SetValue(x, y)
    return actor


def load_if_exists(path):
    return load_vtp(path) if os.path.exists(path) else None


def render_pair(root, case, week, out_path, panel_px):
    fmt = dict(case=case, week=week, week_=week.replace(" ", "_"))
    ref_lmk = read_landmarks(os.path.join(root, REFERENCE_LANDMARKS.format(**fmt)))
    scan_lmks = scan_landmarks(root, fmt)
    ref_scan = reference_scan_name(scan_lmks, ref_lmk)

    recons = {title: load_if_exists(os.path.join(root, rel.format(**fmt))) for title, rel in RECONSTRUCTIONS}
    meshes = [m for m in recons.values() if m is not None]
    frontal, left, right = (view_camera(ref_lmk, meshes, view) for view in ("frontal", "left", "right"))

    # Panels are (title, mesh, mesh colour, [landmark points, ...]); None is a blank cell.
    scan_paths = sorted(glob.glob(os.path.join(root, ALIGNED_SCAN.format(scan="*", **fmt))))
    scan_panels = []
    for path in scan_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        lmk = [np.array(list(scan_lmks[name].values()))] if name in scan_lmks else []
        scan_panels.append((f"{name} (reference)" if name == ref_scan else name, load_vtp(path), SCAN_COLOR, lmk))

    high_conf = load_if_exists(os.path.join(root, HIGH_CONFIDENCE.format(**fmt)))
    unions = [
        ("High-confidence union", high_conf, SCAN_COLOR, [np.array(list(ref_lmk.values()))]),
        ("High-confidence union", high_conf, SCAN_COLOR, []),
    ]
    result_panels = unions + [(title, mesh, MESH_COLOR, []) for title, mesh in recons.items()]

    # The scans wrap over as many rows as they need, above the frontal results and
    # the two side rows, whose models stay in their frontal columns.
    columns = len(result_panels)
    scan_rows = [scan_panels[i:i + columns] for i in range(0, len(scan_panels), columns)]
    side_rows = [(camera, [None] * len(unions)
                  + [(side, mesh, MESH_COLOR, []) for mesh in recons.values()])
                 for camera, side in ((left, "left"), (right, "right"))]
    rows = [(frontal, row) for row in scan_rows] + [(frontal, result_panels)] + side_rows
    write_grid(rows, columns, f"Case {case} - {week} - {len(scan_panels)} scans", out_path, panel_px)


# A vtkRenderWindow taller than this renders as solid BLACK with no error or
# exception (an offscreen-context/GPU renderbuffer size cap - measured on this
# machine: fine up to 32640 px tall, black from 32840 px). Comfortably under
# that, so one write_grid call never silently produces a black PNG for a plot
# with many rows (e.g. fused_target_trajectories.py's 241-subject grid); render
# in row-chunks under this height instead and stack them with PIL.
MAX_WINDOW_HEIGHT = 16000


def _render_rows(rows, columns, panel_px, width):
    """One chunk's worth of `rows` (see write_grid), rendered to an (H, W, 3) uint8 array."""
    height = panel_px * len(rows)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(width, height)

    for r, (camera, row) in enumerate(rows):
        top = 1 - (r * panel_px) / height
        bottom = top - panel_px / height
        for i in range(columns):
            viewport = (i / columns, bottom, (i + 1) / columns, top)
            panel = row[i] if i < len(row) else None
            if panel is not None:
                add_panel(window, camera[i] if isinstance(camera, list) else camera, viewport, *panel)
            else:
                filler = vtk.vtkRenderer()
                filler.SetViewport(*viewport)
                filler.SetBackground(1, 1, 1)
                window.AddRenderer(filler)

    window.Render()
    grabber = vtk.vtkWindowToImageFilter()
    grabber.SetInput(window)
    grabber.Update()
    array = vtk_to_numpy(grabber.GetOutput().GetPointData().GetScalars())
    dims = grabber.GetOutput().GetDimensions()
    image = array.reshape(dims[1], dims[0], -1)[::-1]  # vtkImageData is bottom-up
    window.Finalize()
    return image


def write_grid(rows, columns, header_text, out_path, panel_px):
    """Render rows of panels under a one-line header into one PNG.

    Args:
        rows: [(camera, [(title, mesh, mesh colour, [landmark points, ...]) or None, ...]), ...],
            each row drawn with its own camera, or with one camera per column when
            it is a list; None, and the cells after a row shorter than `columns`,
            are left blank
        columns: panels per row
        header_text: the line above the panels
        out_path: the PNG to write
        panel_px: width and height of each panel in pixels
    """
    from PIL import Image

    header_px = 40
    width = panel_px * columns
    rows_per_chunk = max(1, MAX_WINDOW_HEIGHT // panel_px)

    header_window = vtk.vtkRenderWindow()
    header_window.SetOffScreenRendering(1)
    header_window.SetSize(width, header_px)
    header = vtk.vtkRenderer()
    header.SetBackground(1, 1, 1)
    header.AddViewProp(text_actor(header_text, 20, 0.005, 0.85))
    header_window.AddRenderer(header)
    header_window.Render()
    grabber = vtk.vtkWindowToImageFilter()
    grabber.SetInput(header_window)
    grabber.Update()
    dims = grabber.GetOutput().GetDimensions()
    header_image = vtk_to_numpy(grabber.GetOutput().GetPointData().GetScalars()).reshape(
        dims[1], dims[0], -1)[::-1]
    header_window.Finalize()

    chunks = [header_image]
    for start in range(0, len(rows), rows_per_chunk):
        chunks.append(_render_rows(rows[start:start + rows_per_chunk], columns, panel_px, width))

    Image.fromarray(np.concatenate(chunks, axis=0)).save(out_path)


def add_panel(window, camera, viewport, title, poly, color, landmark_sets):
    renderer = vtk.vtkRenderer()
    renderer.SetViewport(*viewport)
    renderer.SetBackground(0.97, 0.97, 0.97)
    renderer.SetActiveCamera(camera)
    if poly is not None:
        renderer.AddActor(mesh_actor(poly, color))
        for points in landmark_sets:
            renderer.AddActor(landmarks_actor(points, LANDMARK_COLOR))
    renderer.AddViewProp(text_actor(title if poly is not None else f"{title} (missing)", 16, 0.02, 0.98))
    window.AddRenderer(renderer)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="outputs/Casos_Mar", help="Batch output root (holds <case>/, fetus_ssm/, ...).")
    parser.add_argument("--output", default=None, help="Folder for the PNGs (default: <root>/views).")
    parser.add_argument("--panel-px", type=int, default=400, help="Width and height of each panel in pixels.")
    parser.add_argument("--overwrite", action="store_true", help="Re-render PNGs that already exist.")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Render only these case/weeks, named as the PNGs (e.g. 12_26_semanas).")
    args = parser.parse_args()

    output = args.output or os.path.join(args.root, "views")
    os.makedirs(output, exist_ok=True)

    pairs = finished_pairs(args.root)
    print(f"Found {len(pairs)} finished case/week pairs under '{args.root}'.")
    for case, week in pairs:
        name = f"{case}_{week.replace(' ', '_')}"
        if args.cases is not None and name not in args.cases:
            continue
        out_path = os.path.join(output, f"{name}.png")
        if os.path.exists(out_path) and not args.overwrite:
            continue
        try:
            render_pair(args.root, case, week, out_path, args.panel_px)
            print(f"Saved {out_path}")
        except Exception as exc:
            print(f"Case {case} / {week} FAILED: {exc}")

if __name__ == "__main__":
    main()
