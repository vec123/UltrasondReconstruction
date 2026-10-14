"""A keyboard-driven property list: a cursor over named values, each stepped by a key.

Raw VTK has no form widgets, so every editable value is a :class:`Property`
with a ``step(+1 / -1)`` - cycling a choice, toggling a flag, nudging a number.
The :class:`Inspector` only keeps the cursor and renders text; what a step
*does* is the property's closure, so the same inspector edits any kind of item.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence


@dataclass(frozen=True)
class Property:
    label: str
    value: Callable[[], str]                          # current value, as text
    step: Optional[Callable[[int], None]] = None      # None: read-only


class Inspector:
    """The properties of one item, and which of them is focused."""

    def __init__(self):
        self.title = ""
        self.properties: List[Property] = []
        self.cursor = 0

    def show(self, title: str, properties: Sequence[Property]) -> None:
        """Show another item's properties, keeping focus on the same label when it exists."""
        focused = self.focused.label if self.focused is not None else None
        self.title = title
        self.properties = list(properties)
        labels = [p.label for p in self.properties]
        if focused in labels:
            self.cursor = labels.index(focused)
        else:
            self.cursor = min(self.cursor, max(len(self.properties) - 1, 0))
        self._skip_read_only(+1)

    @property
    def focused(self) -> Optional[Property]:
        return self.properties[self.cursor] if self.properties else None

    def focus(self, index: int) -> bool:
        """Focus property ``index`` if it is editable; False otherwise."""
        if 0 <= index < len(self.properties) and self.properties[index].step is not None:
            self.cursor = index
            return True
        return False

    def move(self, delta: int) -> None:
        """Move focus to the next editable property in direction ``delta`` (wrapping)."""
        if not self.properties:
            return
        self.cursor = (self.cursor + delta) % len(self.properties)
        self._skip_read_only(delta)

    def step(self, delta: int) -> bool:
        """Step the focused property; False if there is none or it is read-only."""
        prop = self.focused
        if prop is None or prop.step is None:
            return False
        prop.step(delta)
        return True

    def text(self) -> str:
        lines = [self.title]
        width = max((len(p.label) for p in self.properties), default=0)
        for i, prop in enumerate(self.properties):
            cursor = ">" if i == self.cursor and prop.step is not None else " "
            lines.append(f"{cursor} {prop.label.ljust(width)}  {prop.value()}")
        return "\n".join(lines)

    def _skip_read_only(self, direction: int) -> None:
        if not any(p.step is not None for p in self.properties):
            return
        step = 1 if direction >= 0 else -1
        while self.properties[self.cursor].step is None:
            self.cursor = (self.cursor + step) % len(self.properties)
