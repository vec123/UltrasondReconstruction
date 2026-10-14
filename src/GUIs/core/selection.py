"""Selecting whole regions of a mesh: a vertex-set model, a brush that edits it, and its overlay.

Three separable pieces, so each can be reused or replaced alone:

    VertexSelection   the set itself (a boolean mask) with undo and change listeners - no VTK
    BrushTool         Ctrl+click adds the region around the clicked vertex, Ctrl+Shift+click
                      removes it; *which* region is a callback (e.g. a geodesic ball)
    SelectionOverlay  draws the selected vertices on top of the mesh, following it as it moves
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import vtk
from vtk.util import numpy_support

from src.GUIs.core.input import VertexPicker
from src.GUIs.core.mesh_view import MeshView
from src.GUIs.core.theme import Theme
from src.GUIs.core.viewer import Viewer

#: ``region(seed_vertex, radius) -> vertex indices`` - e.g. TemplateMesh.geodesic_ball.
RegionFn = Callable[[int, float], np.ndarray]


@dataclass(frozen=True)
class Stroke:
    """One brush application, kept so a selection can later be stored as how it was made."""

    seed: int
    radius: float
    subtract: bool


class VertexSelection:
    """A set of mesh vertices with undo; listeners are called after every change."""

    def __init__(self, n_vertices: int, max_undo: int = 100):
        self.mask = np.zeros(int(n_vertices), dtype=bool)
        self.strokes: List[Stroke] = []
        self._history: List[tuple] = []
        self._max_undo = int(max_undo)
        self._listeners: List[Callable[["VertexSelection"], None]] = []

    def __len__(self) -> int:
        return int(self.mask.sum())

    @property
    def indices(self) -> np.ndarray:
        return np.flatnonzero(self.mask)

    def on_change(self, callback: Callable[["VertexSelection"], None]) -> None:
        self._listeners.append(callback)

    # -- edits (each one undoable) ------------------------------------------------

    def add(self, indices: np.ndarray, stroke: Optional[Stroke] = None) -> None:
        self._edit(lambda mask: mask.__setitem__(np.asarray(indices, dtype=np.int64), True), stroke)

    def subtract(self, indices: np.ndarray, stroke: Optional[Stroke] = None) -> None:
        self._edit(lambda mask: mask.__setitem__(np.asarray(indices, dtype=np.int64), False), stroke)

    def set(self, indices: np.ndarray) -> None:
        """Replace the selection."""
        def replace(mask):
            mask[:] = False
            mask[np.asarray(indices, dtype=np.int64)] = True
        self._edit(replace)

    def mirror(self, mirror_index: np.ndarray, keep_original: bool = True) -> None:
        """Add (or switch to) the mirror image, given ``mirror_index[i]`` = vertex mirroring i."""
        def reflect(mask):
            mirrored = np.asarray(mirror_index, dtype=np.int64)[np.flatnonzero(mask)]
            if not keep_original:
                mask[:] = False
            mask[mirrored] = True
        self._edit(reflect)

    def clear(self) -> None:
        self._edit(lambda mask: mask.__setitem__(slice(None), False))

    def undo(self) -> bool:
        """Revert the last edit; False when there is nothing to undo."""
        if not self._history:
            return False
        self.mask, self.strokes = self._history.pop()
        self._notify()
        return True

    def _edit(self, change: Callable[[np.ndarray], None], stroke: Optional[Stroke] = None) -> None:
        self._history.append((self.mask.copy(), list(self.strokes)))
        del self._history[:-self._max_undo]
        change(self.mask)
        if stroke is not None:
            self.strokes.append(stroke)
        self._notify()

    def _notify(self) -> None:
        for callback in self._listeners:
            callback(self)


class BrushTool:
    """Ctrl+click selects the region around a vertex; Ctrl+Shift+click deselects it.

    The modifier keeps the brush out of the way of plain clicks (which other
    tools, like a landmark picker, may own) and of the camera: a handled
    click is consumed, so Ctrl+drag does not also spin the view.
    """

    def __init__(self, viewer: Viewer, selection: Optional[VertexSelection], region: RegionFn,
                 radius: float, tolerance: float = 0.01,
                 min_radius: float = 0.0, max_radius: float = np.inf,
                 on_stroke: Optional[Callable[[Stroke, int], None]] = None,
                 on_miss: Optional[Callable[[int, int], None]] = None,
                 priority: float = 11.0):
        """Attach the brush.

        Args:
            viewer: the window.
            selection: the set strokes edit, or None to only *report* strokes
                through ``on_stroke`` (when the caller records them elsewhere,
                e.g. as steps of a selection recipe).
            region: ``region(seed, radius) -> indices``.
            radius: initial brush radius, in mesh units.
            tolerance: picker tolerance (fraction of the window diagonal).
            min_radius, max_radius: limits for :meth:`scale_radius`.
            on_stroke: called with the stroke and the number of vertices it touched.
            on_miss: called with (x, y) when a brush click hit no vertex.
            priority: above a plain picker's, so a Ctrl+click is the brush's alone.
        """
        self.viewer = viewer
        self.selection = selection
        self.region = region
        self.radius = float(radius)
        self.min_radius, self.max_radius = float(min_radius), float(max_radius)
        self.on_stroke = on_stroke
        # Surface picking: a brush seeds on the visible surface, never behind it.
        self.picker = VertexPicker(viewer, tolerance, self._on_pick, on_miss or (lambda x, y: None),
                                   enabled=lambda: bool(viewer.interactor.GetControlKey()),
                                   priority=priority, consume=True, mode="surface")

    def scale_radius(self, factor: float) -> float:
        """Multiply the radius by ``factor`` (within its limits); returns the new radius."""
        self.radius = float(np.clip(self.radius * factor, self.min_radius, self.max_radius))
        return self.radius

    def stroke(self, seed: int, subtract: bool = False) -> int:
        """Apply the brush at ``seed`` as if clicked; returns the number of vertices touched."""
        region = np.asarray(self.region(int(seed), self.radius), dtype=np.int64)
        stroke = Stroke(int(seed), self.radius, subtract)
        if self.selection is not None:
            if subtract:
                self.selection.subtract(region, stroke)
            else:
                self.selection.add(region, stroke)
        if self.on_stroke is not None:
            self.on_stroke(stroke, len(region))
        return len(region)

    def _on_pick(self, vertex: int) -> None:
        self.stroke(vertex, subtract=bool(self.viewer.interactor.GetShiftKey()))


class SelectionOverlay:
    """The selected vertices drawn as points over the mesh, in a theme colour role (``selection``)."""

    def __init__(self, renderer: vtk.vtkRenderer, mesh: MeshView, selection: VertexSelection,
                 theme: Theme, point_size: float = 6.0, role: str = "selection"):
        self.mesh = mesh
        self.role = role
        self.selection = selection
        self.polydata = vtk.vtkPolyData()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.polydata)
        mapper.ScalarVisibilityOff()
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        prop = self.actor.GetProperty()
        prop.SetRepresentationToPoints()
        prop.SetPointSize(point_size)
        prop.RenderPointsAsSpheresOn()
        prop.LightingOff()
        self.actor.PickableOff()
        renderer.AddActor(self.actor)
        self.apply_theme(theme)
        selection.on_change(lambda _: self.refresh())
        self.refresh()

    def refresh(self) -> None:
        """Rebuild from the selection and the mesh's current shape."""
        indices = self.selection.indices
        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(
            np.ascontiguousarray(self.mesh.vertices[indices], dtype=np.float32), deep=True))
        verts = vtk.vtkCellArray()
        verts.SetCells(len(indices), numpy_support.numpy_to_vtk(
            np.column_stack([np.ones(len(indices), np.int64), np.arange(len(indices), dtype=np.int64)])
            .reshape(-1), deep=True, array_type=vtk.VTK_ID_TYPE))
        self.polydata.SetPoints(points)
        self.polydata.SetVerts(verts)
        self.polydata.Modified()

    def set_visible(self, visible: bool) -> None:
        self.actor.SetVisibility(int(visible))

    def apply_theme(self, theme: Theme) -> None:
        self.actor.GetProperty().SetColor(*theme.color(self.role))
