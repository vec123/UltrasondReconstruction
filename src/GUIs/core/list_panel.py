"""A clickable text panel: rows of cells, each optionally a button, laid out from the window's top.

Raw VTK has no list or form widgets, so this is one: the caller describes the
panel as plain :class:`Row` / :class:`Cell` values every time something
changes, and the panel turns them into text actors, measures them, and maps
mouse events back to the ``key`` of whatever cell is under the pointer:

    click on a keyed cell    -> on_click(key)
    wheel over a row         -> on_wheel(row.wheel, +1 / -1)
    pointer over a keyed cell   that cell is drawn in the ``hover`` role

Keys are any hashable values the caller chooses - the panel never interprets
them - so the same widget serves an outline, a property editor and a history.
Clicks and wheel turns over the panel are consumed, so they never also spin or
zoom the camera; pointer motion is not, so a drag that starts on the mesh
keeps rotating it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import vtk

from src.GUIs.core.hud import make_text_actor
from src.GUIs.core.theme import Theme
from src.GUIs.core.viewer import Viewer


@dataclass(frozen=True)
class Cell:
    """One run of text in a row; a button when it has a ``key``."""

    text: str
    key: Optional[Hashable] = None
    role: str = "text"                  # theme colour role
    bold: bool = False
    x: Optional[int] = None             # px from the panel's left edge; None = after the previous cell
    size: Optional[int] = None          # font size; None = the panel's


@dataclass(frozen=True)
class Row:
    """One line of the panel."""

    cells: Tuple[Cell, ...] = ()
    wheel: Optional[Hashable] = None    # what the mouse wheel steps while over this row

    @classmethod
    def text(cls, text: str, role: str = "text", bold: bool = False, key: Optional[Hashable] = None,
             x: int = 0, wheel: Optional[Hashable] = None) -> "Row":
        return cls((Cell(text, key, role, bold, x),), wheel)


BLANK = Row()


@dataclass
class _Placed:
    """A cell as laid out: its actor and its clickable span, in px from the panel's left edge."""

    cell: Cell
    actor: vtk.vtkTextActor
    left: int
    right: int


class ListPanel:
    """Rows of clickable text in one column of the window."""

    def __init__(self, viewer: Viewer, theme: Theme, x: float, top: int, width: float,
                 on_click: Callable[[Hashable], None],
                 on_wheel: Optional[Callable[[Hashable, int], None]] = None,
                 pitch: int = 17, size: int = 12, gap: int = 8, bottom: int = 110,
                 hover_role: str = "hover", priority: float = 12.0):
        """Attach an empty panel.

        Args:
            viewer: the window.
            theme: initial theme.
            x: left edge, as a fraction of the window width.
            top: px from the window's top edge to the first row's top.
            width: as a fraction of the window width; clicks and wheel turns
                inside [x, x + width] over the rows are the panel's.
            on_click: called with the key of a clicked cell.
            on_wheel: called with a row's ``wheel`` key and +1 (away) / -1 (toward).
            pitch: px from one row's top to the next.
            size: font size.
            gap: px between cells placed one after another.
            bottom: px kept free below the last row (status lines live there);
                rows that do not fit are cut and replaced by an ellipsis.
            hover_role: colour of a keyed cell under the pointer.
            priority: above the brush and the pickers, so a click on the
                panel is the panel's alone.
        """
        self.viewer = viewer
        self.theme = theme
        self.x, self.top, self.width = float(x), int(top), float(width)
        self.on_click = on_click
        self.on_wheel = on_wheel
        self.pitch, self.size, self.gap, self.bottom = int(pitch), int(size), int(gap), int(bottom)
        self.hover_role = hover_role
        self.rows: List[Row] = []
        self._placed: List[List[_Placed]] = []
        self._pool: List[vtk.vtkTextActor] = []
        self._hovered: Optional[_Placed] = None
        self._widths: Dict[Tuple[str, bool, int], int] = {}
        self._measure_prop = vtk.vtkTextProperty()
        self._measure_prop.SetFontSize(self.size)
        self._text_renderer = vtk.vtkTextRenderer()
        interactor = viewer.interactor
        self._tags = {
            event: interactor.AddObserver(event, handler, priority)
            for event, handler in (("LeftButtonPressEvent", self._on_press),
                                   ("MouseWheelForwardEvent", self._on_wheel),
                                   ("MouseWheelBackwardEvent", self._on_wheel),
                                   ("MouseMoveEvent", self._on_move))
        }

    # -- content ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """How many rows fit in the window as it is now."""
        height = self.viewer.size[1]
        return max(1, (height - self.top - self.bottom) // self.pitch)

    def measure(self, text: str, bold: bool = False, size: Optional[int] = None) -> int:
        """Width of ``text`` in px, at font ``size`` (default: the panel's)."""
        key = (text, bold, size or self.size)
        if key not in self._widths:
            if not text:
                self._widths[key] = 0
            else:
                self._measure_prop.SetBold(bold)
                self._measure_prop.SetFontSize(size or self.size)
                bbox = [0, 0, 0, 0]
                self._text_renderer.GetBoundingBox(self._measure_prop, text, bbox,
                                                   self.viewer.render_window.GetDPI())
                self._widths[key] = bbox[1] - bbox[0] + 1
        return self._widths[key]

    def fit(self, text: str, width: int, bold: bool = False) -> str:
        """``text``, cut with an ellipsis so it is at most ``width`` px wide."""
        if self.measure(text, bold) <= width:
            return text
        low, high = 0, len(text)
        while low < high:                       # longest prefix that fits with the ellipsis
            mid = (low + high + 1) // 2
            if self.measure(text[:mid].rstrip() + "…", bold) <= width:
                low = mid
            else:
                high = mid - 1
        return text[:low].rstrip() + "…"

    def wrap(self, text: str, width: int, max_lines: int = 3, bold: bool = False) -> List[str]:
        """``text`` broken at spaces into lines at most ``width`` px wide (the last one cut)."""
        lines: List[str] = []
        current = ""
        for word in text.split():
            trial = f"{current} {word}".strip()
            if current and self.measure(trial, bold) > width:
                lines.append(current)
                current = word
            else:
                current = trial
        if current:
            lines.append(current)
        if len(lines) > max_lines:
            lines = lines[:max_lines - 1] + [" ".join(lines[max_lines - 1:])]
        return [self.fit(line, width, bold) for line in lines]

    @property
    def pixel_width(self) -> int:
        return int(self.width * self.viewer.size[0])

    def set_rows(self, rows: Sequence[Row]) -> None:
        """Replace the content; rows beyond :attr:`capacity` are cut and marked with an ellipsis."""
        rows = list(rows)
        if len(rows) > self.capacity:
            rows = rows[:self.capacity - 1] + [Row.text("…", role="muted")]
        self.rows = rows
        self._hovered = None
        used = 0
        self._placed = []
        for row in rows:
            placed_row: List[_Placed] = []
            cursor = 0
            for cell in row.cells:
                actor = self._actor(used)
                used += 1
                left = cursor if cell.x is None else int(cell.x)
                width = self.measure(cell.text, cell.bold, cell.size)
                actor.SetInput(cell.text)
                actor.GetTextProperty().SetBold(cell.bold)
                actor.GetTextProperty().SetFontSize(cell.size or self.size)
                actor.SetVisibility(1)
                placed_row.append(_Placed(cell, actor, left, left + width))
                cursor = left + width + self.gap
            # A keyed cell is clickable up to where the next cell starts, so a
            # short label still has a comfortable target.
            for i, placed in enumerate(placed_row[:-1]):
                if placed.cell.key is not None:
                    placed.right = max(placed.right, placed_row[i + 1].left - 2)
            self._placed.append(placed_row)
        for actor in self._pool[used:]:
            actor.SetVisibility(0)
        self._recolor()
        self.layout(*self.viewer.size)

    def _actor(self, index: int) -> vtk.vtkTextActor:
        while len(self._pool) <= index:
            actor = make_text_actor(self.size)
            actor.GetTextProperty().SetVerticalJustificationToTop()
            actor.PickableOff()
            self.viewer.renderer.AddViewProp(actor)
            self._pool.append(actor)
        return self._pool[index]

    # -- geometry -------------------------------------------------------------------

    def layout(self, width: int, height: int) -> None:
        left = int(self.x * width)
        for i, placed_row in enumerate(self._placed):
            y = height - self.top - i * self.pitch
            for placed in placed_row:
                placed.actor.SetDisplayPosition(left + placed.left, y)

    def _hit(self, x: int, y: int) -> Tuple[Optional[int], Optional[_Placed], bool]:
        """(row index or None, keyed cell under the pointer or None, inside the panel?)."""
        width, height = self.viewer.size
        local_x = x - int(self.x * width)
        if not 0 <= local_x <= self.pixel_width:
            return None, None, False
        index = (height - self.top - y) // self.pitch
        if not 0 <= index < len(self._placed):
            return None, None, False
        for placed in self._placed[index]:
            if placed.cell.key is not None and placed.left - 4 <= local_x <= placed.right + 4:
                return index, placed, True
        return index, None, True

    # -- events ----------------------------------------------------------------------

    def _abort(self, event: str) -> None:
        self.viewer.interactor.GetCommand(self._tags[event]).SetAbortFlag(1)

    def _on_press(self, interactor, event) -> None:
        _, placed, inside = self._hit(*interactor.GetEventPosition())
        if not inside:
            return
        self._abort(event)
        if placed is not None:
            self.on_click(placed.cell.key)

    def _on_wheel(self, interactor, event) -> None:
        index, _, inside = self._hit(*interactor.GetEventPosition())
        if not inside:
            return
        self._abort(event)
        wheel = self.rows[index].wheel
        if wheel is not None and self.on_wheel is not None:
            self.on_wheel(wheel, 1 if event == "MouseWheelForwardEvent" else -1)

    def _on_move(self, interactor, event) -> None:
        _, placed, _ = self._hit(*interactor.GetEventPosition())
        if placed is not self._hovered:
            self._hovered = placed
            self._recolor()
            self.viewer.render()

    # -- colour ------------------------------------------------------------------------

    def _recolor(self) -> None:
        for placed_row in self._placed:
            for placed in placed_row:
                role = self.hover_role if placed is self._hovered else placed.cell.role
                placed.actor.GetTextProperty().SetColor(*self.theme.color(role))

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self._recolor()
