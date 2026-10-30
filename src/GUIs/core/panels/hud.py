"""Screen-space text: named slots laid out by window edge, recoloured by theme role."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import vtk

from src.GUIs.core.theme import Theme


@dataclass(frozen=True)
class TextSlot:
    """Where one line (or block) of overlay text sits and how it looks.

    Horizontal position is a fraction of the window width, so columns keep
    their proportions; vertical position is a pixel offset from the top or
    bottom edge, so text never drifts off-screen as the window is resized.
    """

    name: str
    x: float                 # fraction of the window width
    y: int                   # pixels from the bottom edge to the text's bottom line, or
                             # (from_top) from the top edge to the text's top line
    from_top: bool = False
    size: int = 12
    bold: bool = False
    role: str = "muted"      # theme colour role


def make_text_actor(size: int, bold: bool = False) -> vtk.vtkTextActor:
    """A plain (no shadow, no italics) 2D text actor, not yet added to a renderer."""
    actor = vtk.vtkTextActor()
    prop = actor.GetTextProperty()
    prop.SetFontSize(size)
    prop.SetBold(bold)
    prop.ShadowOff()
    prop.ItalicOff()
    return actor


class Hud:
    """A set of named text slots on one renderer."""

    def __init__(self, renderer: vtk.vtkRenderer, slots: Sequence[TextSlot]):
        """Create one text actor per slot.

        Raises:
            ValueError: two slots share a name.
        """
        self._slots: Dict[str, TextSlot] = {}
        self._actors: Dict[str, vtk.vtkTextActor] = {}
        for slot in slots:
            if slot.name in self._slots:
                raise ValueError(f"duplicate text slot {slot.name!r}")
            actor = make_text_actor(slot.size, slot.bold)
            if slot.from_top:
                # Hang multi-line text down from its anchor; VTK's default
                # bottom anchoring would grow it upward, off the window.
                actor.GetTextProperty().SetVerticalJustificationToTop()
            renderer.AddActor2D(actor)
            self._slots[slot.name] = slot
            self._actors[slot.name] = actor

    def __getitem__(self, name: str) -> vtk.vtkTextActor:
        return self._actors[name]

    def set_text(self, name: str, text: str) -> None:
        self._actors[name].SetInput(text)

    def text(self, name: str) -> str:
        return self._actors[name].GetInput() or ""

    def layout(self, width: int, height: int) -> None:
        """Position every slot for a window of ``width`` x ``height`` pixels."""
        for name, slot in self._slots.items():
            y = height - slot.y if slot.from_top else slot.y
            self._actors[name].SetDisplayPosition(int(slot.x * width), y)

    def apply_theme(self, theme: Theme) -> None:
        for name, slot in self._slots.items():
            self._actors[name].GetTextProperty().SetColor(*theme.color(slot.role))
