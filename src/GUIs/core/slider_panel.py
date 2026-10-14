"""A scrolling column of mode sliders: a fixed pool of rows re-bound to a window of modes.

The panel never owns the coefficients. It reads and writes them through a
:class:`SliderSource` - anything with ``alpha``, ``sigma`` and
``variance_share`` arrays - and reports each change through a callback, so
what a coefficient *means* (a PCA weight, an LDDMM momentum weight, ...) stays
with the caller.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Tuple

import numpy as np
import vtk

from src.GUIs.core.hud import make_text_actor
from src.GUIs.core.theme import Theme
from src.GUIs.core.viewer import Viewer


class SliderSource(Protocol):
    """Coefficients the panel edits, read fresh on every use (so they may be rebound).

    A source may also carry ``variance_share`` (K,), which the default row
    title shows; pass ``row_title`` to the panel for any other kind of source.
    """

    alpha: np.ndarray            # (K,) coefficients
    sigma: np.ndarray            # (K,) per-row unit; a row shows alpha / sigma


@dataclass(frozen=True)
class PanelGeometry:
    """Panel placement, in normalised display coordinates."""

    x: Tuple[float, float] = (0.030, 0.205)             # slider left/right ends
    y: Tuple[float, float] = (0.880, 0.155)             # centres of the top and bottom rows
    scrollbar_x: Tuple[float, float] = (0.216, 0.224)
    header_from_top: int = 30                           # header text, pixels below the top edge
    track_margin: float = 0.03                          # scrollbar overhang beyond the rows


class Throttle:
    """Lets an action through at most every ``min_interval`` seconds (0 = always)."""

    def __init__(self, min_interval: float = 0.0):
        self.min_interval = float(min_interval)
        self._last = -np.inf

    def ready(self) -> bool:
        return self.min_interval <= 0.0 or time.monotonic() - self._last >= self.min_interval

    def mark(self) -> None:
        self._last = time.monotonic()


class ModeSliderPanel:
    """``visible_rows`` slider widgets scrolled over every mode of a :class:`SliderSource`.

    The rows are a fixed pool chosen once at construction, so the widget count
    does not grow with a model that has hundreds of modes; scrolling re-binds
    the pool to a different window of them. A row past the end of the
    source's *current* modes is disabled rather than indexed, which keeps the
    panel valid when the source switches to a basis with fewer modes.
    """

    def __init__(self, viewer: Viewer, source: SliderSource, visible_rows: int,
                 sigma_range: float, on_change: Callable[[int, bool], None],
                 geometry: PanelGeometry = PanelGeometry(), priority: float = 10.0,
                 row_title: Optional[Callable[[int], str]] = None,
                 fixed_pool: bool = False, noun: str = "modes"):
        """Build the rows, scrollbar and header, and take the mouse wheel over the panel.

        Args:
            viewer: the window the panel lives in.
            source: the coefficients edited.
            visible_rows: rows drawn at once.
            sigma_range: slider travel, in units of ``source.sigma`` either side of 0.
            on_change: called as ``on_change(mode, final)`` after a drag wrote
                ``source.alpha[mode]``; ``final`` is True on release.
            geometry: placement.
            priority: wheel-observer priority, above the camera style's.
            row_title: ``row_title(mode) -> str``; by default "mode i  x% var"
                from ``source.variance_share``.
            fixed_pool: always build ``visible_rows`` rows, for a source whose
                rows appear later (otherwise the pool is capped at the source's
                current size).
            noun: what the rows are called in the header ("modes 0-9 of 40").
        """
        self.viewer = viewer
        self.source = source
        self.on_change = on_change
        self.geometry = geometry
        self.row_title = row_title or self._variance_title
        self.noun = noun
        self.n_rows = int(visible_rows) if fixed_pool else min(int(visible_rows), len(source.alpha))
        self.first_row = 0
        self.sliders: List[vtk.vtkSliderWidget] = []

        self._build_rows(float(sigma_range))
        self._build_scrollbar()
        self.header = make_text_actor(12)
        viewer.renderer.AddActor2D(self.header)

        self._wheel_tags = {
            event: viewer.interactor.AddObserver(event, self._on_wheel, priority)
            for event in ("MouseWheelForwardEvent", "MouseWheelBackwardEvent")
        }

    # -- construction --------------------------------------------------------

    def _build_rows(self, limit: float) -> None:
        top, bottom = self.geometry.y
        spacing = (top - bottom) / max(self.n_rows - 1, 1) if self.n_rows > 1 else 0.0
        x0, x1 = self.geometry.x

        for row in range(self.n_rows):
            y = top - row * spacing
            representation = vtk.vtkSliderRepresentation2D()
            representation.SetMinimumValue(-limit)
            representation.SetMaximumValue(limit)
            representation.SetValue(0.0)
            representation.SetLabelFormat("%+.2f")
            for coordinate, x in ((representation.GetPoint1Coordinate(), x0),
                                  (representation.GetPoint2Coordinate(), x1)):
                coordinate.SetCoordinateSystemToNormalizedDisplay()
                coordinate.SetValue(x, y)
            representation.SetSliderLength(0.016)
            representation.SetSliderWidth(0.020)
            representation.SetEndCapLength(0.003)
            representation.SetEndCapWidth(0.018)
            representation.SetTubeWidth(0.005)
            representation.SetTitleHeight(0.016)
            representation.SetLabelHeight(0.014)

            widget = vtk.vtkSliderWidget()
            widget.SetInteractor(self.viewer.interactor)
            widget.SetRepresentation(representation)
            widget.SetAnimationModeToJump()
            widget.EnabledOn()
            callback = self._row_callback(row)
            widget.AddObserver("InteractionEvent", callback)
            widget.AddObserver("EndInteractionEvent", callback)
            self.sliders.append(widget)

    def _build_scrollbar(self) -> None:
        self.scroll_track, self._track_points = _quad()
        self.scroll_thumb, self._thumb_points = _quad()
        self.viewer.renderer.AddActor2D(self.scroll_track)
        self.viewer.renderer.AddActor2D(self.scroll_thumb)
        _set_quad(self._track_points, self.geometry.scrollbar_x, self._track_range())

    def _track_range(self) -> Tuple[float, float]:
        top, bottom = self.geometry.y
        return bottom - self.geometry.track_margin, top + self.geometry.track_margin

    # -- values ---------------------------------------------------------------

    def _row_callback(self, row: int):
        def callback(widget, event):
            mode = self.first_row + row
            if mode >= len(self.source.alpha):
                return
            self.source.alpha[mode] = widget.GetRepresentation().GetValue() * self.source.sigma[mode]
            self.on_change(mode, event == "EndInteractionEvent")
        return callback

    def reset_values(self) -> None:
        """Put every row's knob back at 0 (the source's own alpha is the caller's to reset)."""
        for widget in self.sliders:
            widget.GetRepresentation().SetValue(0.0)

    # -- scrolling --------------------------------------------------------------

    def _variance_title(self, mode: int) -> str:
        return f"mode {mode}    {self.source.variance_share[mode]:.2%} var"

    def scroll_to(self, first_row: int) -> None:
        """Re-bind the row pool to the window of modes starting at ``first_row``."""
        alpha, sigma = self.source.alpha, self.source.sigma
        n_modes = len(alpha)
        last_start = max(0, n_modes - self.n_rows)
        self.first_row = int(np.clip(first_row, 0, last_start))

        for row, widget in enumerate(self.sliders):
            mode = self.first_row + row
            representation = widget.GetRepresentation()
            if mode >= n_modes:
                representation.SetTitleText("")
                representation.SetValue(0.0)
                widget.SetEnabled(0)
                continue
            widget.SetEnabled(1)
            representation.SetTitleText(self.row_title(mode))
            representation.SetValue(float(alpha[mode] / sigma[mode]))

        track_low, track_high = self._track_range()
        span = track_high - track_low
        visible_rows = min(self.n_rows, n_modes) if n_modes else self.n_rows
        height = min(span, max(span * visible_rows / float(max(n_modes, 1)), 0.02))
        offset = 0.0 if last_start == 0 else self.first_row / float(last_start)
        thumb_top = track_high - offset * (span - height)
        _set_quad(self._thumb_points, self.geometry.scrollbar_x, (thumb_top - height, thumb_top))

        self.header.SetInput(
            f"{self.noun} {self.first_row}-{self.first_row + visible_rows - 1} of {n_modes}"
            if n_modes else f"no {self.noun}"
        )

    def scroll_by(self, rows: int) -> None:
        self.scroll_to(self.first_row + rows)

    def refresh(self) -> None:
        """Re-read the source at the current scroll position - after it changed underneath."""
        self.scroll_to(self.first_row)

    # -- events, layout, theme ---------------------------------------------

    def contains_x(self, x_pixels: int) -> bool:
        """Whether a horizontal screen position falls on the panel (sliders or scrollbar)."""
        return x_pixels <= (self.geometry.scrollbar_x[1] + 0.01) * self.viewer.size[0]

    def _on_wheel(self, interactor, event) -> None:
        x, _ = interactor.GetEventPosition()
        if not self.contains_x(x):
            return
        self.scroll_by(-1 if event == "MouseWheelForwardEvent" else 1)
        self.viewer.render()
        # Over the panel the wheel scrolls modes; it must not also zoom the camera.
        interactor.GetCommand(self._wheel_tags[event]).SetAbortFlag(1)

    def layout(self, width: int, height: int) -> None:
        self.header.SetDisplayPosition(int(self.geometry.x[0] * width),
                                       height - self.geometry.header_from_top)

    def apply_theme(self, theme: Theme) -> None:
        for widget in self.sliders:
            representation = widget.GetRepresentation()
            representation.GetSliderProperty().SetColor(*theme.knob)
            representation.GetSelectedProperty().SetColor(*theme.knob_active)
            representation.GetTubeProperty().SetColor(*theme.rail)
            representation.GetCapProperty().SetColor(*theme.rail)
            for prop in (representation.GetTitleProperty(), representation.GetLabelProperty()):
                prop.SetColor(*theme.muted)
        self.scroll_track.GetProperty().SetColor(*theme.track)
        self.scroll_thumb.GetProperty().SetColor(*theme.thumb)
        self.header.GetTextProperty().SetColor(*theme.muted)


def _quad() -> Tuple[vtk.vtkActor2D, vtk.vtkPoints]:
    """A filled 2D quad in normalised viewport coordinates, and its (movable) corner points."""
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(4)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(4)
    for i in range(4):
        cells.InsertCellPoint(i)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)

    coordinate = vtk.vtkCoordinate()
    coordinate.SetCoordinateSystemToNormalizedViewport()
    mapper = vtk.vtkPolyDataMapper2D()
    mapper.SetInputData(polydata)
    mapper.SetTransformCoordinate(coordinate)

    actor = vtk.vtkActor2D()
    actor.SetMapper(mapper)
    return actor, points


def _set_quad(points: vtk.vtkPoints, x_range: Tuple[float, float],
              y_range: Tuple[float, float]) -> None:
    x0, x1 = x_range
    y0, y1 = y_range
    for i, (x, y) in enumerate(((x0, y0), (x1, y0), (x1, y1), (x0, y1))):
        points.SetPoint(i, x, y, 0.0)
    points.Modified()
