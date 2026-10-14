"""Glyph layers: arrows at points, local frames drawn as their three axes, and reference planes.

Each layer owns its actors, takes plain arrays, and is recoloured by theme
role - so a direction rule's vectors, a frame field and the facial midline are
all drawn the same way whatever computed them. None of them is pickable, so
clicks always reach the mesh underneath.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import vtk
from vtk.util import numpy_support

from src.GUIs.core.theme import Theme


def _vtk_points(points: np.ndarray) -> vtk.vtkPoints:
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(
        np.ascontiguousarray(np.asarray(points, np.float32).reshape(-1, 3)), deep=True))
    return vtk_points


class VectorGlyphs:
    """One arrow per point, ``|vector| * scale`` long, in a theme colour role."""

    def __init__(self, renderer: vtk.vtkRenderer, theme: Theme, role: str = "vector",
                 scale: float = 1.0, shaft_radius: float = 0.03, tip_radius: float = 0.09,
                 tip_length: float = 0.3):
        self._renderer = renderer
        self.role = role
        arrow = vtk.vtkArrowSource()
        arrow.SetShaftResolution(10)
        arrow.SetTipResolution(14)
        arrow.SetShaftRadius(shaft_radius)
        arrow.SetTipRadius(tip_radius)
        arrow.SetTipLength(tip_length)

        self.polydata = vtk.vtkPolyData()
        self.glyph = vtk.vtkGlyph3D()
        self.glyph.SetSourceConnection(arrow.GetOutputPort())
        self.glyph.SetInputData(self.polydata)
        self.glyph.SetVectorModeToUseVector()
        self.glyph.SetScaleModeToScaleByVector()
        self.glyph.OrientOn()
        self.glyph.SetScaleFactor(scale)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.glyph.GetOutputPort())
        mapper.ScalarVisibilityOff()
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.PickableOff()
        renderer.AddActor(self.actor)
        self.apply_theme(theme)
        self.clear()

    def set_data(self, points: np.ndarray, vectors: np.ndarray) -> None:
        """Arrows starting at ``points`` (n, 3) along ``vectors`` (n, 3)."""
        points = np.asarray(points, np.float32).reshape(-1, 3)
        vectors = np.asarray(vectors, np.float32).reshape(-1, 3)
        if len(points) != len(vectors):
            raise ValueError(f"{len(points)} points but {len(vectors)} vectors")
        array = numpy_support.numpy_to_vtk(np.ascontiguousarray(vectors), deep=True)
        array.SetName("vectors")
        self.polydata.SetPoints(_vtk_points(points))
        self.polydata.GetPointData().SetVectors(array)
        self.polydata.Modified()

    def clear(self) -> None:
        self.set_data(np.zeros((0, 3)), np.zeros((0, 3)))

    @property
    def scale(self) -> float:
        return float(self.glyph.GetScaleFactor())

    def set_scale(self, scale: float) -> None:
        self.glyph.SetScaleFactor(float(scale))

    def set_visible(self, visible: bool) -> None:
        self.actor.SetVisibility(int(visible))

    def apply_theme(self, theme: Theme) -> None:
        self.actor.GetProperty().SetColor(*theme.color(self.role))

    def remove(self) -> None:
        self._renderer.RemoveActor(self.actor)


class FrameGlyphs:
    """Local frames drawn as their three axes (tangent, bitangent, normal), one colour each.

    An axis with positive magnitude is drawn as a double arrow pointing out
    (<-->: stretch), a negative one pointing in (>--<: compression), and 0
    hides it - so the same glyphs show a bare frame (magnitudes 1) or a
    prescribed strain (magnitudes = stretch per axis).
    """

    ROLES = ("axis_t", "axis_b", "axis_n")

    def __init__(self, renderer: vtk.vtkRenderer, theme: Theme, length: float):
        """``length``: half-length of a unit axis, in mesh units."""
        self.length = float(length)
        self.axes: List[VectorGlyphs] = [VectorGlyphs(renderer, theme, role) for role in self.ROLES]

    def set_frames(self, points: np.ndarray, frames: np.ndarray,
                   magnitudes: Optional[np.ndarray] = None) -> None:
        """Draw ``frames`` (n, 3, 3), columns = axes, at ``points`` (n, 3).

        Args:
            magnitudes: (3,) or (n, 3) signed length per axis, in units of
                ``length``; default 1 everywhere.
        """
        points = np.asarray(points, float).reshape(-1, 3)
        frames = np.asarray(frames, float).reshape(-1, 3, 3)
        magnitudes = np.broadcast_to(np.asarray(1.0 if magnitudes is None else magnitudes, float),
                                     (len(points), 3))
        for i, glyphs in enumerate(self.axes):
            axis = frames[:, :, i] * self.length
            m = magnitudes[:, i]
            out, inward = m > 0, m < 0
            a_out = axis[out] * m[out, None]
            a_in = axis[inward] * (-m[inward])[:, None]
            starts = np.concatenate([points[out], points[out],
                                     points[inward] + a_in, points[inward] - a_in])
            vectors = np.concatenate([a_out, -a_out, -a_in, a_in])
            glyphs.set_data(starts, vectors)

    def set_length(self, length: float) -> None:
        """Takes effect on the next :meth:`set_frames`."""
        self.length = float(length)

    def clear(self) -> None:
        for glyphs in self.axes:
            glyphs.clear()

    def set_visible(self, visible: bool) -> None:
        for glyphs in self.axes:
            glyphs.set_visible(visible)

    def apply_theme(self, theme: Theme) -> None:
        for glyphs in self.axes:
            glyphs.apply_theme(theme)


class PlaneGlyph:
    """A translucent square marking a plane, e.g. the facial midline mirroring reflects through."""

    def __init__(self, renderer: vtk.vtkRenderer, theme: Theme, role: str = "plane",
                 opacity: float = 0.18):
        self.role = role
        self.source = vtk.vtkPlaneSource()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.source.GetOutputPort())
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().SetOpacity(opacity)
        self.actor.GetProperty().LightingOff()
        self.actor.PickableOff()
        renderer.AddActor(self.actor)
        self.apply_theme(theme)
        self.set_visible(False)

    def set_plane(self, point: Sequence[float], normal: Sequence[float], size: float) -> None:
        """A ``size`` x ``size`` square centred on ``point``, perpendicular to ``normal``."""
        normal = np.asarray(normal, float)
        normal /= np.linalg.norm(normal)
        helper = np.eye(3)[np.argmin(np.abs(normal))]
        u = np.cross(normal, helper)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        origin = np.asarray(point, float) - 0.5 * size * (u + v)
        self.source.SetOrigin(*origin)
        self.source.SetPoint1(*(origin + size * u))
        self.source.SetPoint2(*(origin + size * v))
        self.source.Update()

    def set_visible(self, visible: bool) -> None:
        self.actor.SetVisibility(int(visible))

    @property
    def visible(self) -> bool:
        return bool(self.actor.GetVisibility())

    def apply_theme(self, theme: Theme) -> None:
        self.actor.GetProperty().SetColor(*theme.color(self.role))
