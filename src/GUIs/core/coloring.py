"""How the mesh is coloured: interchangeable colour modes plus the scalar bar that explains them.

A colour mode is a small strategy object - :class:`FlatColoring` (one theme
colour) or :class:`ScalarColoring` (a point field through a ramp of theme
roles). :class:`SurfaceColoring` owns the list of modes for one mesh and
switches between them, so a tool decides *which* modes exist and when to
switch, never how a lookup table is built.
"""

from __future__ import annotations

from typing import Dict, Optional, Protocol, Sequence, Tuple

import numpy as np
import vtk

from src.GUIs.core.mesh_view import MeshView
from src.GUIs.core.theme import Theme

# Ramp stop: (fraction of the value range, theme colour role).
Stop = Tuple[float, str]

DIVERGING: Tuple[Stop, ...] = ((0.0, "concave"), (0.5, "neutral"), (1.0, "convex"))
SEQUENTIAL: Tuple[Stop, ...] = ((0.0, "neutral"), (1.0, "convex"))


class ColorMode(Protocol):
    """One way of colouring a mesh."""

    name: str      # key the controller selects it by
    label: str     # short human name, for status lines

    def apply(self, mesh: MeshView, bar: vtk.vtkScalarBarActor, theme: Theme) -> None:
        """Point the mesh's mapper (and the scalar bar) at this mode."""


class FlatColoring:
    """The whole surface in the theme's ``surface`` colour; no scalar bar."""

    def __init__(self, name: str = "flat", label: str = "flat"):
        self.name = name
        self.label = label

    def apply(self, mesh: MeshView, bar: vtk.vtkScalarBarActor, theme: Theme) -> None:
        mesh.mapper.ScalarVisibilityOff()
        mesh.actor.GetProperty().SetColor(*theme.surface)
        bar.SetVisibility(0)


class ScalarColoring:
    """A point field of the mesh through a ramp of theme colour roles."""

    def __init__(self, name: str, array: str, value_range: Tuple[float, float],
                 title: str, stops: Sequence[Stop] = SEQUENTIAL, label: Optional[str] = None):
        """Configure the mode.

        Args:
            name: key the controller selects it by.
            array: point-field name on the mesh to colour by.
            value_range: (low, high) mapped onto the ramp's 0..1.
            title: scalar bar title.
            stops: (fraction, theme role) pairs, e.g. DIVERGING or SEQUENTIAL.
            label: short name for status lines; defaults to ``name``.
        """
        self.name = name
        self.label = label or name
        self.array = array
        self.value_range = (float(value_range[0]), float(value_range[1]))
        self.title = title
        self.stops = tuple(stops)
        self.lookup_table = vtk.vtkColorTransferFunction()

    def apply(self, mesh: MeshView, bar: vtk.vtkScalarBarActor, theme: Theme) -> None:
        low, high = self.value_range
        self.lookup_table.RemoveAllPoints()
        for fraction, role in self.stops:
            self.lookup_table.AddRGBPoint(low + fraction * (high - low), *theme.color(role))

        mesh.mapper.SetLookupTable(self.lookup_table)
        mesh.mapper.SetScalarModeToUsePointFieldData()
        mesh.mapper.SelectColorArray(self.array)
        mesh.mapper.SetScalarRange(low, high)
        mesh.mapper.ScalarVisibilityOn()
        bar.SetLookupTable(self.lookup_table)
        bar.SetTitle(self.title)
        bar.SetVisibility(1)


def symmetric_range(values: np.ndarray, percentile: float = 97.0) -> Tuple[float, float]:
    """(-r, r) with r the ``percentile`` of |values| - a diverging range a few outliers can't wash out."""
    r = float(max(np.percentile(np.abs(values), percentile), 1e-12))
    return -r, r


def make_scalar_bar(position: Tuple[float, float] = (0.915, 0.06),
                    width: float = 0.07, height: float = 0.26) -> vtk.vtkScalarBarActor:
    """A compact scalar bar, by default in the lower-right corner."""
    bar = vtk.vtkScalarBarActor()
    bar.SetNumberOfLabels(3)
    bar.SetMaximumWidthInPixels(60)
    bar.SetMaximumHeightInPixels(200)
    bar.GetPositionCoordinate().SetValue(*position)
    bar.SetWidth(width)
    bar.SetHeight(height)
    bar.UnconstrainedFontSizeOn()
    bar.GetTitleTextProperty().SetFontSize(13)
    bar.GetLabelTextProperty().SetFontSize(11)
    for prop in (bar.GetTitleTextProperty(), bar.GetLabelTextProperty()):
        prop.SetBold(False)
        prop.ShadowOff()
        prop.ItalicOff()
    return bar


class SurfaceColoring:
    """The colour modes available for one mesh, which one is active, and its scalar bar."""

    def __init__(self, mesh: MeshView, renderer: vtk.vtkRenderer,
                 modes: Sequence[ColorMode], active: str, theme: Theme):
        """Register the modes and show ``active``.

        Raises:
            ValueError: no modes, duplicate names, or ``active`` is not one of them.
        """
        if not modes:
            raise ValueError("SurfaceColoring needs at least one colour mode")
        self._modes: Dict[str, ColorMode] = {}
        for mode in modes:
            if mode.name in self._modes:
                raise ValueError(f"duplicate colour mode {mode.name!r}")
            self._modes[mode.name] = mode
        self.mesh = mesh
        self.theme = theme
        self.scalar_bar = make_scalar_bar()
        renderer.AddActor2D(self.scalar_bar)
        self._active = ""
        self.set_active(active)

    @property
    def active(self) -> ColorMode:
        return self._modes[self._active]

    @property
    def names(self) -> Sequence[str]:
        return tuple(self._modes)

    def add_mode(self, mode: ColorMode) -> None:
        """Register another mode after construction (e.g. one a preview component owns).

        Raises:
            ValueError: a mode of that name exists.
        """
        if mode.name in self._modes:
            raise ValueError(f"duplicate colour mode {mode.name!r}")
        self._modes[mode.name] = mode

    def set_active(self, name: str) -> None:
        if name not in self._modes:
            raise ValueError(f"unknown colour mode {name!r}; available: {list(self._modes)}")
        self._active = name
        self.refresh()

    def cycle(self, step: int = 1) -> ColorMode:
        """Activate the next mode in registration order; returns it."""
        names = list(self._modes)
        self.set_active(names[(names.index(self._active) + step) % len(names)])
        return self.active

    def refresh(self) -> None:
        """Re-apply the active mode, e.g. after its range or theme changed."""
        self.active.apply(self.mesh, self.scalar_bar, self.theme)

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.refresh()
        self.scalar_bar.GetTitleTextProperty().SetColor(*theme.muted)
        self.scalar_bar.GetLabelTextProperty().SetColor(*theme.muted)
