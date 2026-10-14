"""Sphere markers pinned to mesh vertices, optionally labelled, that follow the mesh as it moves."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import vtk

from src.GUIs.core.mesh_view import MeshView
from src.GUIs.core.theme import Theme


class Marker:
    """One sphere, plus an optional billboard label, in a renderer.

    Neither is pickable: otherwise a click near a marker could hand
    ``vtkPointPicker`` the sphere glyph's own point id instead of a mesh vertex.
    """

    def __init__(self, renderer: vtk.vtkRenderer, radius: float, label: Optional[str] = None):
        self._renderer = renderer
        self.sphere = vtk.vtkSphereSource()
        self.sphere.SetRadius(radius)
        self.sphere.SetThetaResolution(16)
        self.sphere.SetPhiResolution(16)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.sphere.GetOutputPort())
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().LightingOff()
        self.actor.PickableOff()
        renderer.AddActor(self.actor)

        self.label: Optional[vtk.vtkBillboardTextActor3D] = None
        if label is not None:
            self.label = vtk.vtkBillboardTextActor3D()
            self.label.SetInput(label)
            prop = self.label.GetTextProperty()
            prop.SetFontSize(14)
            prop.SetJustificationToCentered()
            prop.ShadowOff()
            prop.SetBackgroundOpacity(0.55)
            self.label.PickableOff()
            renderer.AddActor(self.label)

    def move_to(self, position: np.ndarray, label_position: Optional[np.ndarray] = None) -> None:
        self.sphere.SetCenter(*position)
        if self.label is not None:
            self.label.SetPosition(*(position if label_position is None else label_position))

    def set_style(self, color, text_color=None, text_background=None) -> None:
        self.actor.GetProperty().SetColor(*color)
        if self.label is not None:
            prop = self.label.GetTextProperty()
            if text_color is not None:
                prop.SetColor(*text_color)
            if text_background is not None:
                prop.SetBackgroundColor(*text_background)

    def remove(self) -> None:
        self._renderer.RemoveActor(self.actor)
        if self.label is not None:
            self._renderer.RemoveActor(self.label)


class VertexMarkers:
    """Named markers, each pinned to a vertex of one :class:`MeshView`.

    A marker stores the vertex *index*, not a position, so after the mesh
    moves :meth:`reposition` puts every marker back on its vertex. Each marker
    also carries a theme colour *role*, so a theme switch restyles it without
    the caller remembering which marker was highlighted.
    """

    def __init__(self, renderer: vtk.vtkRenderer, mesh: MeshView, radius: float,
                 theme: Theme, labelled: bool = True, label_offset: float = 4.0):
        """Set up an empty marker set.

        Args:
            renderer: where the markers are drawn.
            mesh: the mesh whose vertices markers are pinned to.
            radius: sphere radius, in mesh units.
            theme: initial theme.
            labelled: give each marker a billboard label showing its name.
            label_offset: label distance from its vertex, in sphere radii,
                along the vertex's outward direction from the mesh centroid -
                taken from the shape at construction, which keeps labels
                stable while the mesh moves.
        """
        self.renderer = renderer
        self.mesh = mesh
        self.radius = float(radius)
        self.theme = theme
        self.labelled = labelled
        self.label_offset = float(label_offset)

        outward = mesh.vertices.astype(np.float64) - mesh.vertices.mean(axis=0)
        norm = np.linalg.norm(outward, axis=1, keepdims=True)
        norm[norm < 1e-12] = 1.0
        self._outward = outward / norm

        self._markers: Dict[str, Marker] = {}
        self._vertex: Dict[str, int] = {}
        self._role: Dict[str, str] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._markers

    def __len__(self) -> int:
        return len(self._markers)

    def place(self, name: str, vertex: int, role: str = "marker") -> None:
        """Create or move marker ``name`` onto ``vertex``, coloured by ``role``."""
        if name not in self._markers:
            self._markers[name] = Marker(self.renderer, self.radius,
                                         label=name if self.labelled else None)
        self._vertex[name] = int(vertex)
        self._role[name] = role
        self._move(name)
        self._style(name)

    def set_role(self, name: str, role: str) -> None:
        """Recolour marker ``name``; a no-op when it does not exist."""
        if name in self._markers:
            self._role[name] = role
            self._style(name)

    def remove(self, name: str) -> None:
        marker = self._markers.pop(name, None)
        if marker is not None:
            marker.remove()
            del self._vertex[name], self._role[name]

    def clear(self) -> None:
        for name in list(self._markers):
            self.remove(name)

    def reposition(self) -> None:
        """Put every marker back on its vertex - after the mesh moved."""
        for name in self._markers:
            self._move(name)

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        for name in self._markers:
            self._style(name)

    def _move(self, name: str) -> None:
        vertex = self._vertex[name]
        position = self.mesh.vertices[vertex]
        label_position = position + self._outward[vertex] * (self.radius * self.label_offset)
        self._markers[name].move_to(position, label_position)

    def _style(self, name: str) -> None:
        self._markers[name].set_style(self.theme.color(self._role[name]),
                                      text_color=self.theme.text,
                                      text_background=self.theme.background)
