"""Stage: PNG views of the meshes of many cases, one column per mesh source.

    stage: render_views
    columns:
      - {title: Fused target, mesh: "${OUT}/fetus_ssm/${SESSION}/end_to_end/fused_scan_ssm/eval/recon_w_distance.vtp"}
      - {title: Cohort g8, mesh: "${OUT}/resnet_lddmm_cohort_autodecoder_fieldweighted_g8/apply/${CASE_KEY}/eval/recon_w_distance.vtp"}
    landmarks: ${OUT}/resnet_lddmm_baby_mean/${SESSION}/apply/eval/reference_landmarks.vtp
    output_dir: ${OUT}/views

`columns`, `scans` and `landmarks` are patterns: the engine leaves their
variables without a value in (StageSpec.patterns), and the stage finds the
cases on the disk (targets.find_case_paths), as compare does. With every case
variable given (a pipeline run of one case), there is one case, `case_name`.

layout per_case writes <output_dir>/<case>.png: the scans of the case, frontal
and wrapped over as many rows as needed, then one row per view (frontal, left,
right) of the columns, each column in the same place in every row. A column
without a mesh for the case shows "(missing)". layout grid writes
<output_dir>/views.png: one row per case, in the first view.

The cameras come from the face's landmarks (src/shared/vtk/render.py): the
`landmarks` file of the case, or else the heatmap fields of its first mesh.
They frame the meshes of the columns with `camera: true`. All meshes of a case
must be in one frame, e.g. the *_w_distance.vtp of the evaluations. With
`color`, each column mesh that has the point field is coloured by it.
"""

import os

import numpy as np
import pandas as pd
from vtk.util.numpy_support import vtk_to_numpy

from src.evaluation.config import RenderViewsConfig as Config
from src.shared.core.config import expand_known, find_paths
from src.shared.core.pipeline.cases import SUBJECT_ID, WEEK, parse_case_key
from src.shared.core.pipeline.targets import find_case_paths
from src.shared.landmarks.vtk_landmarks import heatmap_landmark_vertex_indices
from src.shared.vtk.io import load_vtp
from src.shared.vtk.render import (
    MESH_COLOR,
    SCAN_COLOR,
    ramp_colors,
    read_landmarks,
    view_camera,
    with_colors,
    write_grid,
)

LAYOUTS = ("per_case", "grid")
GRID_FILE = "views.png"
TINTS = {"mesh": MESH_COLOR, "scan": SCAN_COLOR}


def read_any_landmarks(path):
    """{label: point} of a landmark file: a .vtp with a 'label' array, or a CSV with label, x, y, z."""
    if path.endswith(".vtp"):
        return read_landmarks(path)
    df = pd.read_csv(path)
    return {label: df.loc[i, ["x", "y", "z"]].to_numpy(float) for i, label in enumerate(df["label"])}


def heatmap_landmarks(path):
    """{label: point} at the peaks of the heatmap fields of a mesh."""
    points = vtk_to_numpy(load_vtp(path).GetPoints().GetData())
    return {label: points[index] for label, index in heatmap_landmark_vertex_indices(path).items()}


def fill(pattern, values):
    """`pattern` with the case's values; None when it still has a variable without a value."""
    text = expand_known(pattern, values)
    return None if "${" in text else text


def find_cases(cfg):
    """{case name: its values}: the cases of the landmarks pattern, or else of all columns."""
    patterns = [cfg.landmarks] if cfg.landmarks else [column.mesh for column in cfg.columns]
    cases = {}
    for pattern in patterns:
        for name, (values, _) in find_case_paths(pattern, "render_views").items():
            if name == "case" and cfg.case_name:  # every case variable has a value
                name, values = cfg.case_name, parse_case_key(cfg.case_name)
            cases.setdefault(name, values)
    return cases


def header(name, values):
    if SUBJECT_ID in values:
        return f"Case {values[SUBJECT_ID]}" + (f" - {values[WEEK]}" if values.get(WEEK) else "")
    return name


def column_meshes(cfg, values):
    """[the mesh of each column for the case, coloured by cfg.color when it has the field; None if missing]."""
    meshes = []
    for column in cfg.columns:
        path = fill(column.mesh, values)
        mesh = load_vtp(path) if path and os.path.exists(path) else None
        if mesh is not None and cfg.color is not None and mesh.GetPointData().GetArray(cfg.color.field) is not None:
            field = vtk_to_numpy(mesh.GetPointData().GetArray(cfg.color.field))
            mesh = with_colors(mesh, ramp_colors(field, cfg.color.max))
        meshes.append(mesh)
    return meshes


def camera_landmarks(cfg, values, meshes):
    if cfg.landmarks:
        return read_any_landmarks(fill(cfg.landmarks, values))
    for column, mesh in zip(cfg.columns, meshes):
        if mesh is not None:
            return heatmap_landmarks(fill(column.mesh, values))
    raise ValueError("no mesh of the case to take landmarks from")


def scan_panels(cfg, values, landmarks):
    """The frontal panels of the case's scans; the scan nearest to the camera landmarks is the reference."""
    if cfg.scans is None or not cfg.scans.mesh:
        return []
    scans = []
    for scan_values, path in find_paths(cfg.scans.mesh, values):
        lmk_path = fill(cfg.scans.landmarks, {**values, **scan_values}) if cfg.scans.landmarks else None
        lmk = read_any_landmarks(lmk_path) if lmk_path and os.path.exists(lmk_path) else {}
        scans.append((os.path.splitext(os.path.basename(path))[0], load_vtp(path), lmk))

    def distance(lmk):
        shared = [label for label in landmarks if label in lmk]
        return np.mean([np.linalg.norm(lmk[label] - landmarks[label]) for label in shared]) if shared else np.inf

    reference = min(scans, key=lambda scan: distance(scan[2]))[0] if scans else None
    return [(f"{name} (reference)" if name == reference else name, mesh, SCAN_COLOR,
             [np.array(list(lmk.values()))] if lmk else []) for name, mesh, lmk in scans]


def column_panel(cfg, column, mesh, landmarks, title=None):
    """The panel of a column; with a field colour, a mesh without the field keeps the plain colour."""
    if column.tint not in TINTS:
        raise ValueError(f"column '{column.title}': tint must be one of {list(TINTS)}, got {column.tint!r}")
    color = None if mesh is not None and cfg.color is not None and mesh.GetPointData().GetArray("rgb") is not None \
        else TINTS[column.tint]
    lmk = [np.array(list(landmarks.values()))] if column.landmarks else []
    return (title or column.title, mesh, color, lmk)


def cameras(cfg, landmarks, meshes):
    framing = [mesh for column, mesh in zip(cfg.columns, meshes) if column.camera and mesh is not None]
    if not framing:
        framing = [mesh for mesh in meshes if mesh is not None]
    return {view: view_camera(landmarks, framing, view) for view in cfg.views}


def render_case(cfg, name, values, out_path):
    meshes = column_meshes(cfg, values)
    landmarks = camera_landmarks(cfg, values, meshes)
    camera = cameras(cfg, landmarks, meshes)
    columns = len(cfg.columns)
    scans = scan_panels(cfg, values, landmarks)
    rows = [(camera[cfg.views[0]], scans[i:i + columns]) for i in range(0, len(scans), columns)]
    for view in cfg.views:
        rows.append((camera[view], [
            column_panel(cfg, column, mesh, landmarks, title=None if view == cfg.views[0] else view)
            if not column.views or view in column.views else None
            for column, mesh in zip(cfg.columns, meshes)]))
    text = header(name, values) + (f" - {len(scans)} scans" if cfg.scans is not None and cfg.scans.mesh else "")
    write_grid(rows, columns, text, out_path, cfg.panel_px)


def render_grid(cfg, cases, out_path):
    view = cfg.views[0]
    rows = []
    for name, values in cases.items():
        meshes = column_meshes(cfg, values)
        camera = cameras(cfg, camera_landmarks(cfg, values, meshes), meshes)[view]
        rows.append((camera, [column_panel(cfg, column, mesh, {}, title=f"{name}: {column.title}")
                              for column, mesh in zip(cfg.columns, meshes)]))
    write_grid(rows, len(cfg.columns), f"{len(cases)} cases - {view}", out_path, cfg.panel_px)


def run(cfg: Config):
    if not cfg.columns:
        raise ValueError("render_views needs at least one entry under 'columns'.")
    if cfg.layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {cfg.layout!r}")
    if not cfg.output_dir:
        raise ValueError("render_views requires output_dir.")
    os.makedirs(cfg.output_dir, exist_ok=True)
    cases = find_cases(cfg)
    print(f"render_views: {len(cases)} case(s), {len(cfg.columns)} column(s), views {cfg.views}")

    if cfg.layout == "grid":
        out_path = os.path.join(cfg.output_dir, GRID_FILE)
        render_grid(cfg, cases, out_path)
        print(f"Saved {out_path}")
        return

    failed = []
    for name, values in cases.items():
        out_path = os.path.join(cfg.output_dir, f"{name}.png")
        if os.path.exists(out_path) and not cfg.overwrite:
            continue
        try:
            render_case(cfg, name, values, out_path)
            print(f"Saved {out_path}")
        except Exception as error:  # one case without its files must not stop the others
            failed.append(name)
            print(f"Case {name} FAILED: {error}")
    if failed:
        print(f"render_views: {len(failed)} of {len(cases)} case(s) failed: {failed}")
