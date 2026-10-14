"""The side panel's content: what the editor shows about the document, as clickable rows.

Pure description - no VTK, no event handling. :func:`panel_rows` turns the
document, the selected item and the inspector into :class:`~src.GUIs.core.Row`
values whose cell keys say what a click means; the app dispatches them:

    ("select", ref)        select an item                ("new", group)    add one
    ("focus", i)           focus property i              ("step", i, d)    change it by d (< >)
    ("drop", start, stop)  remove region steps start..stop-1
    ("action", name)       rename, delete, compile, mirror, landmarks, field, undo, redo, ...

and a row's ``wheel`` key ("prop", i) means the wheel over it changes property i.

Three sections, top to bottom - what exists, what the selected item is, what
has been done:

    outline    every mode, constraint set and axis, one line each, with its state
    inspector  the selected item's values (with steppers), its region as the
               steps that built it, a mode's compiled field and warnings, and
               what to do next
    history    the last edits, newest first, with undo / redo
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np

from src.deformation_modes import Axis, Brush, ConstraintSet, FieldThreshold, Mirror, Mode, NearLandmarks
from src.GUIs.core import BLANK, Cell, Inspector, Row
from src.GUIs.deformation_modes.document import GROUP_NOUNS, SETTINGS, EditorDocument, ItemRef

if TYPE_CHECKING:
    from src.GUIs.core import ListPanel

# Columns, px from the panel's left edge.
_NAME_X = 12
_VALUE_X = 150          # outline summaries, read-only values, the steppers
_ARROW_SIZE = 16        # font size of the < > steppers, larger than the text so they are easy to hit

GROUP_HEADINGS = {
    "modes": ("MODES", "what moves"),
    "constraint_sets": ("CONSTRAINT SETS", "what stays or slides"),
    "axes": ("AXES", "directions modes move in"),
}
_SET_VERBS = {"anchor": "holds", "slide": "slides", "rigid": "keeps rigid"}
_HISTORY_ROWS = 5


def panel_rows(doc: EditorDocument, active: ItemRef, inspector: Inspector, text: "ListPanel",
               selected: int) -> List[Row]:
    """Every row of the side panel, top to bottom.

    Args:
        doc: the document.
        active: the selected item.
        inspector: its properties (as built by PropertyBuilder) and the focus.
        text: the panel, for measuring - long text is cut or wrapped to fit.
        selected: vertices in the selected item's region (as drawn on the face).
    """
    return (outline_rows(doc, active, text) + [BLANK]
            + inspector_rows(doc, active, inspector, text, selected) + [BLANK]
            + history_rows(doc, text))


# -- outline ------------------------------------------------------------------


def outline_rows(doc: EditorDocument, active: ItemRef, text: "ListPanel") -> List[Row]:
    rows: List[Row] = []
    for group in ("modes", "constraint_sets", "axes"):
        title, meaning = GROUP_HEADINGS[group]
        rows.append(Row((Cell(title, role="text", bold=True), Cell(meaning, role="muted"),
                         Cell(f"[+ new {GROUP_NOUNS[group]}]", key=("new", group), role="accent"))))
        items = getattr(doc.modeset, group)
        if not items:
            rows.append(Row.text("none yet", role="muted", x=_NAME_X))
        for item in items:
            rows.append(_item_row(doc, (group, item.name), active, text))
    rows.append(Row.text("SETTINGS", bold=True))
    rows.append(_item_row(doc, SETTINGS, active, text))
    return rows


def _item_row(doc: EditorDocument, ref: ItemRef, active: ItemRef, text: "ListPanel") -> Row:
    chosen = ref == active
    name = "defaults" if ref == SETTINGS else ref[1]
    summary, badge, badge_role = item_summary(doc, ref)
    badge_width = text.measure(badge) + 8 if badge else 0
    room = text.pixel_width - _VALUE_X - badge_width
    cells = [Cell("•" if chosen else "", role="accent"),
             Cell(text.fit(name, _VALUE_X - _NAME_X - 6, bold=chosen), key=("select", ref),
                  role="accent" if chosen else "text", bold=chosen, x=_NAME_X),
             Cell(text.fit(summary, room), key=("select", ref), role="text" if chosen else "muted",
                  x=_VALUE_X)]
    if badge:
        cells.append(Cell(badge, key=("select", ref), role=badge_role))
    return Row(tuple(cells))


def item_summary(doc: EditorDocument, ref: ItemRef) -> Tuple[str, str, str]:
    """(one-line description, status badge, badge colour role) of an item."""
    ms = doc.modeset
    if ref == SETTINGS:
        reference, name = doc.reference_length
        return (f"kernel width {ms.sigma / reference:.0%} of {name} · spacing {ms.spacing:.2f} "
                f"· softness {ms.nu:.0e}", "", "muted")
    item = doc.get(ref)
    if isinstance(item, Axis):
        users = [m.name for m in ms.modes if m.axis == item.name]
        return item.rule.describe(), ("used" if users else ""), "muted"
    if isinstance(item, ConstraintSet):
        users = [m.name for m in ms.modes if item.name in m.constraints]
        count = _count(doc, item.selection)
        summary = f"{item.kind} · {count} vertices"
        if users:
            return summary + f" · used by {', '.join(users)}", "", "muted"
        return summary, "unused", "muted"
    return mode_summary(doc, item), *mode_badge(doc, item)


def mode_summary(doc: EditorDocument, mode: Mode) -> str:
    if mode.kind == "displacement":
        what = f"along {mode.axis or '(no axis)'}"
    else:
        what = (f"stretch t{mode.stretch[0]:+g} b{mode.stretch[1]:+g} n{mode.stretch[2]:+g} "
                f"on {mode.axis or '(no axis)'}")
    sets = []
    for name in mode.constraints:
        if doc.modeset.has("constraint_sets", name):
            sets.append(f"{_SET_VERBS[doc.modeset.constraint_set(name).kind]} {name}")
    return " · ".join([what] + sets)


def mode_badge(doc: EditorDocument, mode: Mode) -> Tuple[str, str]:
    compiled = doc.compiled(mode.name)
    if compiled is None:
        return "not compiled", "muted"
    if doc.is_stale(mode.name):
        return "out of date", "warning"
    if compiled.warnings:
        return f"{len(compiled.warnings)} warning{'s' * (len(compiled.warnings) > 1)}", "warning"
    return "ok", "accent"


def _count(doc: EditorDocument, selection) -> str:
    try:
        return str(len(doc.resolve(selection)))
    except (KeyError, ValueError):
        return "?"


# -- inspector ----------------------------------------------------------------


def inspector_rows(doc: EditorDocument, active: ItemRef, inspector: Inspector, text: "ListPanel",
                   selected: int) -> List[Row]:
    item = doc.get(active)
    noun = "SETTINGS" if active == SETTINGS else GROUP_NOUNS[active[0]].upper()
    header = [Cell(noun, role="accent", bold=True),
              Cell("defaults for every mode" if active == SETTINGS else active[1], role="accent", bold=True)]
    if active != SETTINGS:
        header += [Cell("[rename]", key=("action", "rename"), role="accent"),
                   Cell("[delete]", key=("action", "delete"), role="accent")]
    rows = [Row(tuple(header))]
    if isinstance(item, (Mode, ConstraintSet)):
        rows.append(Row((Cell("on the face:", role="muted", x=_NAME_X),
                         Cell("• pink = its region", role="selection"),
                         Cell("• purple = the sets it uses", role="marker") if isinstance(item, Mode)
                         else Cell(""))))
    rows += property_rows(inspector, text)
    if isinstance(item, (Mode, ConstraintSet)):
        rows += region_rows(doc, item, text, selected)
    if isinstance(item, Mode):
        rows += field_rows(doc, item, text)
    hint = next_step(doc, active)
    if hint:
        for i, line in enumerate(text.wrap(f"next: {hint}", text.pixel_width - _NAME_X, max_lines=2)):
            rows.append(Row.text(line, role="accent", x=_NAME_X + (0 if i == 0 else 30)))
    return rows


def property_rows(inspector: Inspector, text: "ListPanel") -> List[Row]:
    rows = []
    arrow = text.measure("<", True, _ARROW_SIZE) + 8
    value_room = text.pixel_width - _VALUE_X - 2 * arrow
    for i, prop in enumerate(inspector.properties):
        focused = i == inspector.cursor and prop.step is not None
        label = text.fit(prop.label, _VALUE_X - _NAME_X - 6, bold=focused)
        if prop.step is None:
            rows.append(Row((Cell(label, role="muted", x=_NAME_X),
                             Cell(text.fit(prop.value(), text.pixel_width - _VALUE_X), role="muted",
                                  x=_VALUE_X))))
            continue
        rows.append(Row((
            Cell("•" if focused else "", role="accent"),
            Cell(label, key=("focus", i), role="accent" if focused else "text", bold=focused, x=_NAME_X),
            Cell("<", key=("step", i, -1), role="accent", bold=True, x=_VALUE_X, size=_ARROW_SIZE),
            Cell(text.fit(prop.value(), value_room), key=("focus", i),
                 role="accent" if focused else "text", bold=focused, x=_VALUE_X + arrow),
            Cell(">", key=("step", i, +1), role="accent", bold=True, size=_ARROW_SIZE),
        ), wheel=("prop", i)))
    return rows


def region_rows(doc: EditorDocument, item, text: "ListPanel", selected: int) -> List[Row]:
    """The region, as the steps that built it - each removable on its own."""
    selection = item.sites if isinstance(item, Mode) else item.selection
    title = "SITES" if isinstance(item, Mode) else "REGION"
    meaning = "what moves" if isinstance(item, Mode) else f"what it {_SET_VERBS[item.kind]}"
    rows = [Row((Cell(title, bold=True, x=_NAME_X), Cell(f"{meaning} · {selected} vertices", role="muted"),
                 Cell("[landmarks…]", key=("action", "landmarks"), role="accent"),
                 Cell("[field…]", key=("action", "field"), role="accent"),
                 Cell("[mirror]", key=("action", "mirror"), role="accent")))]
    groups = recipe_groups(selection.ops, doc.reference_length)
    if not groups:
        rows.append(Row.text("empty - Ctrl+click on the face to brush it, or use the buttons above",
                             role="muted", x=_NAME_X + 18))
    drop_x = text.pixel_width - text.measure("[remove]")
    for number, (start, stop, description) in enumerate(groups, 1):
        rows.append(Row((Cell(f"{number}.", role="muted", x=_NAME_X + 18),
                         Cell(text.fit(description, drop_x - _NAME_X - 48), x=_NAME_X + 40),
                         Cell("[remove]", key=("drop", start, stop), role="accent", x=drop_x))))
    return rows


def recipe_groups(ops: Sequence, reference: Tuple[float, str]) -> List[Tuple[int, int, str]]:
    """(start, stop, description) per run of steps: consecutive strokes of one brush are one run."""
    length, _ = reference
    groups: List[Tuple[int, int, str]] = []
    i = 0
    while i < len(ops):
        op = ops[i]
        sign = "-" if getattr(op, "subtract", False) else "+"
        if isinstance(op, Brush):
            j = i
            while (j < len(ops) and isinstance(ops[j], Brush) and ops[j].subtract == op.subtract
                   and np.isclose(ops[j].radius, op.radius)):
                j += 1
            n = j - i
            verb = "erase" if op.subtract else "brush"
            groups.append((i, j, f"{sign} {n} {verb} stroke{'s' * (n > 1)}, size {op.radius / length:.0%}"))
            i = j
            continue
        if isinstance(op, NearLandmarks):
            description = f"{sign} around {', '.join(op.labels)}, size {op.radius / length:.0%}"
        elif isinstance(op, FieldThreshold):
            description = f"{sign} where {op.describe().lstrip('+-')}"
        elif isinstance(op, Mirror):
            description = "+ mirrored to the other side"
        else:
            description = op.describe()
        groups.append((i, i + 1, description))
        i += 1
    return groups


def field_rows(doc: EditorDocument, mode: Mode, text: "ListPanel") -> List[Row]:
    """A mode's generated field: whether it is current, and what the solver warned about."""
    rows = [Row((Cell("FIELD", bold=True, x=_NAME_X),
                 Cell("[compile]", key=("action", "compile"), role="accent")))]
    compiled = doc.compiled(mode.name)
    if compiled is None:
        rows.append(Row.text("not compiled yet", role="muted", x=_NAME_X + 18))
        return rows
    peak = float(np.linalg.norm(compiled.field, axis=1).max())
    state = "out of date - changed since it was compiled" if doc.is_stale(mode.name) else "up to date"
    rows.append(Row.text(f"{state} · peak {peak:.3g} · {compiled.kernel.n_rows} constraint rows · "
                         f"{compiled.seconds:.1f} s",
                         role="warning" if doc.is_stale(mode.name) else "text", x=_NAME_X + 18))
    width = text.pixel_width - _NAME_X - 30
    for warning in compiled.warnings:
        for i, line in enumerate(text.wrap(warning, width, max_lines=2)):
            rows.append(Row.text(("! " if i == 0 else "   ") + line, role="warning", x=_NAME_X + 18))
    return rows


def next_step(doc: EditorDocument, active: ItemRef) -> Optional[str]:
    """What to do next with the selected item, or None when it is complete."""
    ms = doc.modeset
    item = doc.get(active)
    if active == SETTINGS:
        return "these apply to every mode that does not set its own; most specs never change them"
    if isinstance(item, Axis):
        return ("click start or end below, then Ctrl+click near a landmark on the face to set it; "
                "the arrows show the direction")
    if isinstance(item, ConstraintSet):
        if not item.selection.ops:
            return "Ctrl+click on the face to paint where it applies; [mirror] copies it to the other side"
        if not any(item.name in m.constraints for m in ms.modes):
            return f"select a mode and switch 'uses {item.name}' to yes - a set does nothing on its own"
        return None
    if not item.sites.ops:
        return "Ctrl+click on the face to mark what moves (or [landmarks…] e.g. aL aR)"
    if not item.axis:
        return "pick an axis above - the direction the sites move in"
    if not item.constraints:
        if ms.constraint_sets:
            return "switch 'uses …' to yes for the regions that must stay still or only slide"
        return "[+ new constraint set] for what must stay still - without one the whole face drifts along"
    if doc.compiled(item.name) is None or doc.is_stale(item.name):
        return "[compile] (or p) to solve the field, then drag its slider on the left"
    compiled = doc.compiled(item.name)
    if compiled.warnings:
        return "read the warnings above - move a set away from the sites, or lower sigma, then compile again"
    return "drag its slider on the left to check the shape; the message line reports folding"


# -- history ----------------------------------------------------------------------


def history_rows(doc: EditorDocument, text: "ListPanel") -> List[Row]:
    past, undone = doc.history, doc.redo_history
    rows = [Row((Cell("HISTORY", bold=True),
                 Cell(f"{len(past)} edit{'s' * (len(past) != 1)}{' · unsaved' if doc.dirty else ' · saved'}",
                      role="warning" if doc.dirty else "muted"),
                 Cell("[undo]", key=("action", "undo") if past else None,
                      role="accent" if past else "muted"),
                 Cell("[redo]", key=("action", "redo") if undone else None,
                      role="accent" if undone else "muted"),
                 Cell("[save]", key=("action", "save"), role="accent")))]
    width = text.pixel_width - _NAME_X - 50
    if undone:
        rows.append(Row((Cell("redo:", role="muted", x=_NAME_X),
                         Cell(text.fit(undone[-1], width), role="muted", x=_NAME_X + 44))))
    if not past:
        rows.append(Row.text("nothing yet", role="muted", x=_NAME_X))
    runs = history_runs(past)
    for i, (description, count) in enumerate(runs[:_HISTORY_ROWS]):
        label = description + (f"  (x{count})" if count > 1 else "")
        rows.append(Row((Cell("undo:" if i == 0 else "", role="muted", x=_NAME_X),
                         Cell(text.fit(label, width), role="text" if i == 0 else "muted",
                              x=_NAME_X + 44))))
    if len(runs) > _HISTORY_ROWS:
        earlier = sum(count for _, count in runs[_HISTORY_ROWS:])
        rows.append(Row.text(f"… {earlier} earlier", role="muted", x=_NAME_X + 44))
    return rows


def history_runs(past: Sequence[str]) -> List[Tuple[str, int]]:
    """(description, how many times in a row), newest first - five nudges of one value are one line."""
    runs: List[Tuple[str, int]] = []
    for description in reversed(past):
        if runs and runs[-1][0] == description:
            runs[-1] = (description, runs[-1][1] + 1)
        else:
            runs.append((description, 1))
    return runs
