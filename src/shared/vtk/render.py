"""Offscreen rendering of meshes into one PNG: a grid of panels under a header.

A panel is ``(title, mesh, colour, [landmark points, ...])``, or None for a
blank cell; write_grid draws rows of them, each row with its own camera.
The cameras are set by a face's landmarks (face_frame): left-right from
exR -> exL, up from pg -> n, and the facing sign from prn. They work in
millimetres.

The colour of a panel selects how mesh_actor draws its mesh:
  * an (r, g, b) tuple: one colour;
  * None: the mesh's own RGB point scalars (with_colors, e.g. of ramp_colors);
  * a function poly -> vtkActor: that function builds the actor (e.g. a
    scalar field through a lookup table).
"""

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from src.shared.vtk.io import load_vtp

MESH_COLOR = (0.93, 0.80, 0.70)
SCAN_COLOR = (0.55, 0.70, 0.85)
LANDMARK_COLOR = (0.84, 0.15, 0.16)
NEAR_PLANE_MM = 10.0  # how far in front of the nose tip the camera starts seeing
# Sequential ramp, light (0) -> dark (max_value and above).
SEQUENTIAL_RAMP = ("#fff5d6", "#fdd98b", "#f9a94b", "#e8712c", "#c44620", "#8f2618", "#561410")


def read_landmarks(path):
    poly = load_vtp(path)
    points = vtk_to_numpy(poly.GetPoints().GetData())
    labels = poly.GetPointData().GetAbstractArray("label")
    return {labels.GetValue(i): points[i] for i in range(poly.GetNumberOfPoints())}


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


def ramp_colors(values, max_value, ramp=SEQUENTIAL_RAMP):
    """[N] values -> [N, 3] uint8 RGB on `ramp`, clipped at max_value."""
    stops = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in ramp], dtype=float)
    t = np.clip(np.asarray(values, dtype=float) / max_value, 0, 1) * (len(ramp) - 1)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, len(ramp) - 1)
    frac = (t - lo)[:, None]
    return ((1 - frac) * stops[lo] + frac * stops[hi]).round().astype(np.uint8)


def with_colors(poly, rgb):
    """A copy of `poly` with [N, 3] uint8 `rgb` as its RGB point scalars (a panel with colour None)."""
    colored = vtk.vtkPolyData()
    colored.DeepCopy(poly)
    array = numpy_to_vtk(rgb, deep=True)
    array.SetName("rgb")
    colored.GetPointData().SetScalars(array)
    return colored


def mesh_actor(poly, color):
    """The actor of `poly`; `color` is a colour, None or a function (see the module docstring)."""
    if callable(color):
        return color(poly)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetSpecular(0.2)
    if color is None:
        mapper.ScalarVisibilityOn()
        mapper.SetColorModeToDirectScalars()
        return actor
    mapper.ScalarVisibilityOff()
    actor.GetProperty().SetColor(color)
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
