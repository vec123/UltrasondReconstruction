"""The editor's bottom text: the status line, the prompt line and the key hints - pure functions.

(The side panel's rows are ``panel.py``.)
"""

from __future__ import annotations

from typing import Optional

from src.deformation_modes import AwayFrom, Axis, Between, ConstraintSet, Mode
from src.GUIs.deformation_modes.document import GROUP_NOUNS, SETTINGS, EditorDocument, ItemRef
from src.GUIs.deformation_modes.properties import split_rule

HINTS = (
    "Mouse: click a name or a [button] on the right; < > or the wheel over a value changes it.   "
    "Ctrl+click the face: brush (Ctrl+Shift+click: erase).   Sliders on the left deform the face.",
    "Keys: Up/Down item  Left/Right value  , . change  a/c/n new  Enter rename  Del delete  "
    "k landmarks  f field  m mirror  p/e compile  0 amplitudes  u/y undo/redo  s save  "
    "+/- brush  h colour  l landmarks  b midline  t theme  r camera  q quit",
)


def ctrl_click_effect(doc: EditorDocument, active: ItemRef, focused_label: Optional[str]) -> str:
    """What a Ctrl+click on the face does right now - so it never does something unseen."""
    item = doc.get(active)
    if isinstance(item, Mode):
        return f"Ctrl+click adds to {item.name}'s sites (what moves)"
    if isinstance(item, ConstraintSet):
        return f"Ctrl+click adds to {item.name}'s region ({item.kind})"
    if isinstance(item, Axis):
        base = split_rule(item.rule)[0]
        if isinstance(base, Between):
            end = "start" if focused_label == "start" else "end"
            return f"Ctrl+click sets {item.name}'s {end} to the nearest landmark"
        if isinstance(base, AwayFrom):
            return f"Ctrl+click adds / removes the nearest landmark as {item.name}'s centre"
        return f"Ctrl+click does nothing ({item.name} uses no landmark)"
    return "Ctrl+click does nothing - select a mode or a constraint set to brush"


def status_text(doc: EditorDocument, active: ItemRef, brush_radius: float, color_label: str,
                focused_label: Optional[str]) -> str:
    reference, reference_name = doc.reference_length
    noun = "settings" if active == SETTINGS else f"{GROUP_NOUNS[active[0]]} {active[1]}"
    return (f"[{noun}]   {ctrl_click_effect(doc, active, focused_label)}   ·   brush "
            f"{brush_radius / reference:.0%} of {reference_name} (+/-)   ·   colour: {color_label} (h)")


def prompt_line(label: str, buffer: str) -> str:
    return f"{label}: {buffer}_      (Enter = confirm, Esc = cancel)"
