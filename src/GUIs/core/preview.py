"""Live preview of displacement fields: deform the mesh by amplitude, colour and arrow the active field.

:class:`FieldPreview` keeps the mesh's rest shape and a list of named fields
``u_k`` (one (N, 3) displacement per vertex - typically a generated mode at
unit strength), and shows ``rest + sum_k alpha_k u_k``. It is also the
:class:`~src.GUIs.core.slider_panel.SliderSource` for amplitude sliders:
``alpha`` holds the amplitudes and ``sigma`` one *reference* amplitude per
field, chosen so that a slider at 1 moves that field's peak by
``reference_fraction`` of the mesh radius - so sliders read the same whatever
units the field came in.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import vtk

from src.GUIs.core.coloring import SEQUENTIAL, ScalarColoring, SurfaceColoring
from src.GUIs.core.glyphs import VectorGlyphs
from src.GUIs.core.mesh_view import MeshView
from src.GUIs.core.theme import Theme

MAGNITUDE_FIELD = "field_magnitude"


def spread_subset(points: np.ndarray, candidates: np.ndarray, count: int) -> np.ndarray:
    """At most ``count`` of ``candidates`` spread over space, earlier candidates preferred.

    One candidate per grid cell, with the cell sized so a surface patch holds
    about ``count`` cells; pass candidates sorted by priority (e.g. magnitude,
    descending) so each cell keeps its most important one.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    if len(candidates) <= count:
        return candidates
    p = points[candidates]
    extent = float((p.max(0) - p.min(0)).max())
    cell = extent / max(np.sqrt(count), 1.0)
    keys = np.floor((p - p.min(0)) / max(cell, 1e-12)).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    chosen = candidates[np.sort(first)]
    if len(chosen) > count:
        chosen = chosen[np.linspace(0, len(chosen) - 1, count).astype(np.int64)]
    return chosen


class FieldPreview:
    """Named displacement fields on one mesh, previewed live."""

    def __init__(self, mesh: MeshView, renderer: vtk.vtkRenderer, theme: Theme,
                 faces: Optional[np.ndarray] = None, arrow_count: int = 250,
                 arrow_length: float = 0.06, reference_fraction: float = 0.05):
        """Take the mesh's current shape as the rest shape.

        Args:
            mesh: the mesh to deform (its vertex buffer is written to).
            renderer: where the arrows go.
            theme: initial theme.
            faces: (F, 3) triangles, for :meth:`flipped_faces`; None disables it.
            arrow_count: at most this many arrows for the active field.
            arrow_length: longest arrow, as a fraction of the mesh radius.
            reference_fraction: a slider at 1 moves a field's peak by this
                fraction of the mesh radius.
        """
        self.mesh = mesh
        self.faces = faces
        self.rest = mesh.vertices.astype(np.float64).copy()
        self.radius = float(np.linalg.norm(self.rest - self.rest.mean(0), axis=1).max())
        self.arrow_count = int(arrow_count)
        self.arrow_length = float(arrow_length)
        self.reference_fraction = float(reference_fraction)

        self.names: List[str] = []
        self.fields = np.zeros((0, len(self.rest), 3))
        self.alpha = np.zeros(0)
        self.sigma = np.zeros(0)
        self.active: Optional[int] = None

        mesh.add_point_field(MAGNITUDE_FIELD)
        self.coloring = ScalarColoring("field", MAGNITUDE_FIELD, (0.0, 1.0), "field\nmagnitude",
                                       SEQUENTIAL, label="field")
        self.arrows = VectorGlyphs(renderer, theme, role="vector")
        self._surface_coloring: Optional[SurfaceColoring] = None
        self._listeners: List[Callable[[], None]] = []
        self._field_listeners: List[Callable[[], None]] = []

    # -- wiring -----------------------------------------------------------------

    def attach_coloring(self, coloring: SurfaceColoring) -> None:
        """Offer the field-magnitude colouring in ``coloring``, kept in range as fields change."""
        coloring.add_mode(self.coloring)
        self._surface_coloring = coloring

    def on_shape_changed(self, callback: Callable[[], None]) -> None:
        """Call ``callback()`` after every redraw of the shape (e.g. to move markers with it)."""
        self._listeners.append(callback)

    def on_fields_changed(self, callback: Callable[[], None]) -> None:
        """Call ``callback()`` when fields are added, replaced or removed, or the active one
        changes - e.g. ``panel.refresh`` to update slider rows and titles. Not called while
        amplitudes change, so a slider being dragged is never rewritten under the mouse."""
        self._field_listeners.append(callback)

    def _fields_changed(self) -> None:
        self.update()
        for callback in self._field_listeners:
            callback()

    # -- fields -----------------------------------------------------------------

    def set_field(self, name: str, field: np.ndarray) -> int:
        """Add field ``name`` (or replace it, keeping its amplitude); returns its index.

        Raises:
            ValueError: the field does not have one 3-vector per mesh vertex.
        """
        field = np.asarray(field, np.float64)
        if field.shape != self.rest.shape:
            raise ValueError(f"field {name!r} has shape {field.shape}, expected {self.rest.shape}")
        peak = float(np.linalg.norm(field, axis=1).max())
        reference = self.reference_fraction * self.radius / peak if peak > 0 else 1.0
        if name in self.names:
            index = self.names.index(name)
            self.fields[index] = field
            self.sigma[index] = reference
        else:
            index = len(self.names)
            self.names.append(name)
            self.fields = np.concatenate([self.fields, field[None]], axis=0)
            self.alpha = np.append(self.alpha, 0.0)
            self.sigma = np.append(self.sigma, reference)
        if self.active is None:
            self.active = index
        self._fields_changed()
        return index

    def remove_field(self, name: str) -> None:
        index = self.names.index(name)
        del self.names[index]
        keep = np.arange(len(self.alpha)) != index
        self.fields, self.alpha, self.sigma = self.fields[keep], self.alpha[keep], self.sigma[keep]
        if self.active is not None:
            self.active = None if not self.names else min(self.active, len(self.names) - 1)
        self._fields_changed()

    def set_active(self, which: Union[int, str]) -> None:
        """Which field is coloured and drawn as arrows (all of them deform the mesh)."""
        index = self.names.index(which) if isinstance(which, str) else int(which)
        if not 0 <= index < len(self.names):
            raise IndexError(f"no field {which!r}; have {self.names}")
        self.active = index
        self._fields_changed()

    def set_amplitudes(self, amplitudes: Sequence[float]) -> None:
        self.alpha[:] = amplitudes
        self.update()

    def reset_amplitudes(self) -> None:
        self.alpha[:] = 0.0
        self.update()

    def row_title(self, index: int) -> str:
        """Slider row title: the field's name, starred when it is the active one."""
        return f"{self.names[index]}{'  *' if index == self.active else ''}"

    # -- what is shown --------------------------------------------------------

    def displacement(self) -> np.ndarray:
        """(N, 3) current total displacement ``sum_k alpha_k u_k``."""
        if not len(self.alpha):
            return np.zeros_like(self.rest)
        return np.einsum("k,knd->nd", self.alpha, self.fields)

    def deformed(self) -> np.ndarray:
        return self.rest + self.displacement()

    def flipped_faces(self) -> Optional[int]:
        """Triangles the current deformation turns over - the folding check; None without faces."""
        if self.faces is None:
            return None
        from src.geometry.geometry import face_normals
        before = face_normals(self.rest, self.faces, normalized=False)
        after = face_normals(self.deformed(), self.faces, normalized=False)
        return int(((before * after).sum(-1) < 0).sum())

    def update(self) -> None:
        """Redraw: deformed shape, active field's magnitude colouring and arrows."""
        shape = self.deformed()
        self.mesh.set_vertices(shape)

        if self.active is None:
            magnitude = np.zeros(len(self.rest))
            self.arrows.clear()
        else:
            field = self.fields[self.active]
            magnitude = np.linalg.norm(field, axis=1)
            self._draw_arrows(shape, field, magnitude)
        self.mesh.set_point_field(MAGNITUDE_FIELD, magnitude)
        self.coloring.value_range = (0.0, max(float(magnitude.max()), 1e-12))
        if self._surface_coloring is not None and self._surface_coloring.active is self.coloring:
            self._surface_coloring.refresh()
        for callback in self._listeners:
            callback()

    def _draw_arrows(self, shape: np.ndarray, field: np.ndarray, magnitude: np.ndarray) -> None:
        peak = float(magnitude.max())
        if peak <= 0:
            self.arrows.clear()
            return
        candidates = np.flatnonzero(magnitude > 0.05 * peak)
        candidates = candidates[np.argsort(-magnitude[candidates])]
        chosen = spread_subset(shape, candidates, self.arrow_count)
        self.arrows.set_data(shape[chosen], field[chosen] * (self.arrow_length * self.radius / peak))

    def set_arrows_visible(self, visible: bool) -> None:
        self.arrows.set_visible(visible)

    def apply_theme(self, theme: Theme) -> None:
        self.arrows.apply_theme(theme)
