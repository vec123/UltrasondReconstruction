"""The picker's text: checklist panel, status line and key hint, as pure functions of its state."""

from __future__ import annotations

from src.GUIs.set_landmarks.checklist import LandmarkChecklist

HINT = (
    "click = pending pick      Up/Down = choose name      Enter = link them      "
    "a = custom name      Delete = clear current      m = reset shape      "
    "l = linear/LDDMM      n/p = next/prev real subject      h = heatmap coloring      "
    "v = surface/points      s = save      x = export mean.vtp      "
    "t = theme      c = reset camera      q = quit (autosaves)"
)

START_MESSAGE = (
    "window open - click a vertex (it becomes the pending pick), Up/Down chooses a "
    "landmark name, Enter links the two - either order works. 'a' adds a custom name, "
    "Delete clears the current one, 'm' resets the shape to the mean, 'l' toggles "
    "linear/LDDMM, 'n'/'p' step through real subjects, 'h' toggles heatmap coloring, "
    "'s' saves, 'x' exports mean.vtp, 'q' quits (autosaves on close)"
)


def checklist_text(checklist: LandmarkChecklist) -> str:
    """The right-hand panel: one row per name, cursor and placed box."""
    lines = [f"landmarks  ({checklist.n_placed}/{len(checklist.names)} placed)", ""]
    for i, name in enumerate(checklist.names):
        cursor = ">" if i == checklist.current_index else " "
        box = "x" if checklist.is_placed(name) else " "
        lines.append(f"{cursor} [{box}] {name}")
    return "\n".join(lines)


def status_text(checklist: LandmarkChecklist, shape_label: str, color_label: str) -> str:
    """The status line: what is shown, the current target, and unsaved/pending flags."""
    current = checklist.current_name
    state = "placed" if checklist.is_placed(current) else "not yet placed"
    pending = "   pending pick ready to link" if checklist.pending_vertex is not None else ""
    dirty = "   *unsaved changes*" if checklist.dirty else ""
    return (f"[{shape_label}][{color_label}]  current target: {current}  ({state})      "
            f"{checklist.n_placed}/{len(checklist.names)} placed{pending}{dirty}")


def prompt_text(buffer: str) -> str:
    """The status line while a custom name is being typed."""
    return f"new landmark name: {buffer}_      (Enter = confirm, Esc = cancel)"
