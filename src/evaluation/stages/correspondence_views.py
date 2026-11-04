"""Stage: draw a correspondence field on the surfaces, in one PNG.

scripts/SSM/correspondence_field.py writes 'correspondence_id' onto a recon and
a target point cloud. This stage draws it as a surface field: the fused scan
(a mesh, in mm), the recon and the full identity field of the recon (every
vertex coloured by its own index), side by side, with frontal, left and right
views in rows. A last row shows the landmark field (heatmap_landmarks) of the
recon carried to the scan through the correspondence (each matched scan point
takes the value of its recon partner), and the field of the recon itself. A target vertex and a recon vertex with the
same colour are partners. Points with no partner are grey.

The target of a cohort run is a point cloud, a subset of the vertices of the
fused scan. Its field is moved onto the fused scan by vertex identity (the
points must agree to ``surface_tolerance``). A scan vertex outside the subset
takes the value of its nearest subset vertex, only so that the surface has no
holes. This is for the picture. The files are not changed.

The recon is moved from its own frame into the scan frame (mm) by
``similarity_path``, so both panels share the cameras. The cameras are set by
the heatmap landmarks of the scan (src/shared/vtk/render.py).
"""

import json
import os

import numpy as np
import vtk
from matplotlib import colormaps
from scipy.spatial import cKDTree
from vtk.util.numpy_support import vtk_to_numpy

from src.evaluation.config import CorrespondenceOverviewConfig as OverviewConfig
from src.evaluation.config import CorrespondenceViewsConfig as Config
from src.shared.landmarks.vtk_landmarks import heatmap_landmark_vertex_indices
from src.shared.transforms.similarity import SimilarityTransform
from src.shared.vtk import render
from src.shared.vtk.fields import add_point_field
from src.shared.vtk.io import load_vtp

VIEWS = ("frontal", "left", "right")
FIELD = "correspondence_id"
NO_PARTNER = (0.78, 0.78, 0.78, 1.0)
COLORMAP = "turbo"
LANDMARK_FIELD = "heatmap_landmarks"
CARRIED_FIELD = "landmark_from_recon"
LANDMARK_COLORMAP = "viridis"


def field_color(array_name, high, colormap):
    """The ``color`` of a panel (see src/shared/vtk/render.py): the point array ``array_name``
    of the mesh through a lookup table over 0 .. high, NaN grey."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    for i, (r, g, b, _) in enumerate(colormaps[colormap](np.linspace(0, 1, 256))):
        lut.SetTableValue(i, r, g, b, 1.0)
    lut.SetNanColor(*NO_PARTNER)
    lut.SetRange(0, high)
    lut.Build()

    def actor(poly):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(0, high)
        mapper.ScalarVisibilityOn()
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(array_name)
        mesh = vtk.vtkActor()
        mesh.SetMapper(mapper)
        mesh.GetProperty().SetSpecular(0.2)
        return mesh

    return actor


def field_on_surface(surface_points, target_points, target_values, tolerance):
    """(V,) field of the surface: the values of the target points, moved onto the vertices they are.

    Vertices that are not a target point take the value of the nearest one.
    """
    distance, vertex = cKDTree(surface_points).query(target_points)
    if distance.max() > tolerance:
        raise ValueError(
            f"the target points are not vertices of the surface: largest gap {distance.max():.3g} "
            f"(surface_tolerance {tolerance})"
        )
    field = np.full(len(surface_points), np.nan)
    field[vertex] = target_values
    others = np.setdiff1d(np.arange(len(surface_points)), vertex)
    if others.size:
        _, nearest = cKDTree(surface_points[vertex]).query(surface_points[others])
        field[others] = target_values[nearest]
    return field


def surface_path(cfg, case_id):
    """The fused scan of the case: ``cfg.surface_path``, else its entry in the original_scans.json."""
    if cfg.surface_path:
        return cfg.surface_path
    with open(cfg.original_scans_path) as f:
        scans = json.load(f)
    if case_id not in scans:
        raise KeyError(f"'{cfg.original_scans_path}' has no scan for case '{case_id}'")
    return scans[case_id]


def matched_ids(poly):
    """(N,) correspondence_id of a written field file, NaN where the point has no partner."""
    ids = vtk_to_numpy(poly.GetPointData().GetArray(FIELD)).astype(np.float64)
    weight = vtk_to_numpy(poly.GetPointData().GetArray("correspondence_weight"))
    ids[weight <= 0] = np.nan
    return ids


def point_array(poly, name):
    """The point array ``name`` of ``poly``, as float64."""
    return vtk_to_numpy(poly.GetPointData().GetArray(name)).astype(np.float64)


def carried_landmarks(recon_landmarks, target_ids):
    """(N,) landmark field at the recon partner of each target point, NaN where it has none.

    ``recon_landmarks``: the field of the recon, one value per recon vertex.
    """
    carried = np.full(len(target_ids), np.nan)
    has_partner = np.isfinite(target_ids)
    carried[has_partner] = recon_landmarks[target_ids[has_partner].astype(np.int64)]
    return carried


def reference_tile(poly, cfg, to_mm, landmark_indices, landmark_values):
    """(camera, mesh) of a shape with the vertices of the template, moved into the scan frame (mm).

    The mesh carries the tile field: its own vertex index, or ``landmark_values``.
    """
    shape = to_mm.apply_polydata(poly)
    points = vtk_to_numpy(shape.GetPoints().GetData())
    if len(landmark_values) != len(points):
        raise ValueError(f"{len(points)} vertices, but {len(landmark_values)} landmark values")
    add_point_field(
        shape, np.arange(len(points), dtype=np.float64) if cfg.field == FIELD else landmark_values, cfg.field,
    )
    landmarks = {label: points[index] for label, index in landmark_indices.items()}
    return render.view_camera(landmarks, [shape], "frontal"), shape


def run(cfg: Config):
    folder = cfg.correspondence_dir
    output_path = cfg.output_path or os.path.join(folder, "correspondence_field.png")
    with open(os.path.join(folder, "correspondence_summary.json")) as f:
        summary = json.load(f)
    n_recon = summary["recon"]["total"]

    to_mm = SimilarityTransform.from_json(cfg.similarity_path)
    recon = to_mm.apply_polydata(load_vtp(os.path.join(folder, "recon_w_correspondence.vtp")))
    recon_ids = matched_ids(recon)
    add_point_field(recon, recon_ids, FIELD)

    target = load_vtp(os.path.join(folder, "target_w_correspondence.vtp"))
    target_points = to_mm.apply_points(vtk_to_numpy(target.GetPoints().GetData()))
    target_ids = matched_ids(target)

    scan_path = surface_path(cfg, summary["case_id"])
    surface = load_vtp(scan_path)
    surface_points = vtk_to_numpy(surface.GetPoints().GetData())
    add_point_field(
        surface,
        field_on_surface(surface_points, target_points, target_ids, cfg.surface_tolerance),
        FIELD,
    )

    add_point_field(
        surface,
        field_on_surface(surface_points, target_points, carried_landmarks(point_array(recon, LANDMARK_FIELD), target_ids), cfg.surface_tolerance),
        CARRIED_FIELD,
    )

    full = vtk.vtkPolyData()
    full.DeepCopy(recon)
    add_point_field(full, np.arange(n_recon, dtype=np.float64), FIELD)

    landmarks = {
        label: surface_points[index]
        for label, index in heatmap_landmark_vertex_indices(scan_path).items()
    }
    cameras = {view: render.view_camera(landmarks, [surface, recon], view) for view in VIEWS}
    color = field_color(FIELD, n_recon - 1, COLORMAP)
    landmark_color = field_color(LANDMARK_FIELD, 1.0, LANDMARK_COLORMAP)
    carried_color = field_color(CARRIED_FIELD, 1.0, LANDMARK_COLORMAP)
    panels = [
        (f"scan: {int(np.isfinite(target_ids).sum())} of {len(target_ids)} points matched", surface, color, []),
        (f"recon: {int(np.isfinite(recon_ids).sum())} of {n_recon} vertices matched", recon, color, []),
        (f"recon: all {n_recon} vertices, by index", full, color, []),
    ]
    rows = [(cameras[view], panels) for view in VIEWS]
    rows.append((cameras["frontal"], [
        ("scan: landmark field carried from recon", surface, carried_color, []),
        ("recon: landmark field", recon, landmark_color, []),
    ]))
    render.write_grid(
        rows, len(panels), f"{summary['case_id']}: same colour = same recon vertex, grey = no partner",
        output_path, cfg.panel_px,
    )
    print(f"Saved correspondence views to: {output_path}")


def run_overview(cfg: OverviewConfig):
    """One PNG with a tile per case: the fused scan, frontal view, coloured by ``cfg.field``:
    the correspondence_id, or the landmark field carried from the recon."""
    if cfg.field not in (FIELD, CARRIED_FIELD):
        raise ValueError(f"field must be '{FIELD}' or '{CARRIED_FIELD}', got '{cfg.field}'")
    folder = cfg.cases_dir
    output_path = cfg.output_path or os.path.join(folder, "correspondence_overview.png")
    with open(cfg.original_scans_path) as f:
        scans = json.load(f)
    to_mm = SimilarityTransform.from_json(cfg.similarity_path)
    shared_landmarks = point_array(load_vtp(cfg.landmark_path), cfg.landmark_field) if cfg.landmark_path else None

    panels, cameras, n_recon = [], [], None
    for case in sorted(os.listdir(folder)):
        case_dir = os.path.join(folder, case)
        if not os.path.isdir(case_dir):
            continue
        with open(os.path.join(case_dir, "correspondence_summary.json")) as f:
            summary = json.load(f)
        if n_recon not in (None, summary["recon"]["total"]):
            raise ValueError(f"{case}: {summary['recon']['total']} recon vertices, the cases before had {n_recon}")
        n_recon = summary["recon"]["total"]

        target = load_vtp(os.path.join(case_dir, "target_w_correspondence.vtp"))
        target_points = to_mm.apply_points(vtk_to_numpy(target.GetPoints().GetData()))
        values = matched_ids(target)
        if cfg.field == CARRIED_FIELD:
            recon_landmarks = shared_landmarks
            if recon_landmarks is None:
                recon_landmarks = point_array(
                    load_vtp(os.path.join(case_dir, "recon_w_correspondence.vtp")), LANDMARK_FIELD)
            values = carried_landmarks(recon_landmarks, values)
        scan_path = scans[summary["case_id"]]
        surface = load_vtp(scan_path)
        surface_points = vtk_to_numpy(surface.GetPoints().GetData())
        add_point_field(
            surface,
            field_on_surface(surface_points, target_points, values, cfg.surface_tolerance),
            cfg.field,
        )
        landmarks = {
            label: surface_points[index]
            for label, index in heatmap_landmark_vertex_indices(scan_path).items()
        }
        cameras.append(render.view_camera(landmarks, [surface], "frontal"))
        panels.append((case, surface))

    n_cases = len(panels)
    if cfg.reference_path:
        template = load_vtp(cfg.reference_path)
        indices = heatmap_landmark_vertex_indices(cfg.reference_path)
        values = shared_landmarks if shared_landmarks is not None else point_array(template, LANDMARK_FIELD)
        tiles = [("template", template)]
        if cfg.mean_recon_path:
            tiles.append(("mean_recon", load_vtp(cfg.mean_recon_path)))
        for title, poly in reversed(tiles):
            camera, shape = reference_tile(poly, cfg, to_mm, indices, values)
            cameras.insert(0, camera)
            panels.insert(0, (title, shape))

    color = (field_color(FIELD, n_recon - 1, COLORMAP) if cfg.field == FIELD
             else field_color(CARRIED_FIELD, 1.0, LANDMARK_COLORMAP))
    legend = ("same colour = same recon vertex" if cfg.field == FIELD
              else f"landmark field carried from the recon ({cfg.landmark_field})")
    rows = [
        (cameras[start:start + cfg.columns],
         [(title, surface, color, []) for title, surface in panels[start:start + cfg.columns]])
        for start in range(0, len(panels), cfg.columns)
    ]
    render.write_grid(
        rows, cfg.columns, f"{os.path.basename(os.path.normpath(folder))}: {n_cases} cases, "
        f"{legend}, grey = no partner", output_path, cfg.tile_px,
    )
    print(f"Saved correspondence overview to: {output_path}")
