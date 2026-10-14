"""What the inspector shows and edits for each kind of item - its settable values.

A region's recipe and a mode's compiled field are not properties: the side
panel (``panel.py``) shows them as sections of their own.

Every editable value is a :class:`~src.GUIs.core.Property` whose ``step``
builds a new ModeSet and hands it to the document - so every edit made here is
undoable, and nothing else in the editor needs to know how an axis rule or a
mode's stretch is changed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, List, Tuple

import numpy as np

from src.deformation_modes import (
    CONSTRAINT_KINDS, MODE_KINDS, AwayFrom, AwayFromMidline, Axis, Between, ConstraintSet,
    MidlineNormal, Mode, Reversed, SurfaceNormal, Tangential,
)
from src.GUIs.core import Property
from src.GUIs.deformation_modes.document import GROUP_NOUNS, SETTINGS, EditorDocument, ItemRef

#: Base rules an axis can cycle through; tangential / reversed are separate toggles.
RULE_KINDS = ("between", "away_from_midline", "midline_normal", "surface_normal", "away_from")

OnEdit = Callable[[str], None]       # called with a description after each committed edit


def _cycle(options, current, delta):
    options = list(options)
    index = options.index(current) if current in options else -1
    return options[(index + delta) % len(options)]


# -- axis rules: (base, tangential, reversed) <-> nested rule ---------------------------


def split_rule(rule):
    """(base rule, tangential?, reversed?) - the canonical form is Reversed(Tangential(base))."""
    reversed_ = isinstance(rule, Reversed)
    if reversed_:
        rule = rule.rule
    tangential = isinstance(rule, Tangential)
    if tangential:
        rule = rule.rule
    return rule, tangential, reversed_


def join_rule(base, tangential: bool, reversed_: bool):
    rule = Tangential(base) if tangential else base
    return Reversed(rule) if reversed_ else rule


def rule_kind(base) -> str:
    return {Between: "between", AwayFromMidline: "away_from_midline", MidlineNormal: "midline_normal",
            SurfaceNormal: "surface_normal", AwayFrom: "away_from"}.get(type(base), type(base).__name__)


def make_base(kind: str, landmarks: List[str], previous=None):
    """A base rule of ``kind``, keeping landmark choices from ``previous`` where they fit."""
    if kind == "between":
        if isinstance(previous, Between):
            return previous
        start, end = ("aR", "aL") if {"aR", "aL"} <= set(landmarks) else (landmarks[0], landmarks[-1])
        return Between(start, end)
    if kind == "away_from":
        if isinstance(previous, AwayFrom):
            return previous
        return AwayFrom(("prn",) if "prn" in landmarks else (landmarks[0],))
    return {"away_from_midline": AwayFromMidline, "midline_normal": MidlineNormal,
            "surface_normal": SurfaceNormal}[kind]()


class PropertyBuilder:
    """Builds the inspector's (title, properties) for whichever item is selected."""

    def __init__(self, document: EditorDocument, on_edit: OnEdit):
        self.doc = document
        self.on_edit = on_edit

    @property
    def landmarks(self) -> List[str]:
        return sorted(self.doc.mesh.landmarks)

    def build(self, ref: ItemRef) -> Tuple[str, List[Property]]:
        if ref == SETTINGS:
            return "settings (defaults for every mode)", self._settings()
        item = self.doc.get(ref)
        title = f"{GROUP_NOUNS[ref[0]]}: {item.name}"
        if isinstance(item, Axis):
            return title, self._axis(item)
        if isinstance(item, ConstraintSet):
            return title, self._constraint_set(item)
        return title, self._mode(item)

    # -- shared helpers ---------------------------------------------------------------

    def _commit(self, modeset, description: str) -> None:
        if self.doc.edit(modeset, description):
            self.on_edit(description)

    def _replace(self, item, description: str) -> None:
        self._commit(self.doc.modeset.with_item(item), description)

    def _length(self, value: float) -> str:
        reference, name = self.doc.reference_length
        return f"{value / reference:.1%} of {name}"

    # -- settings ------------------------------------------------------------------------

    def _settings(self) -> List[Property]:
        ms = self.doc.modeset

        def sigma(d):
            self._commit(replace(self.doc.modeset, sigma=self.doc.modeset.sigma * 1.25 ** d),
                         "default sigma")

        def spacing(d):
            value = float(np.clip(round(self.doc.modeset.spacing + 0.1 * d, 2), 0.2, 2.0))
            self._commit(replace(self.doc.modeset, spacing=value), "constraint spacing")

        def nu(d):
            value = float(np.clip(self.doc.modeset.nu * 10.0 ** d, 1e-9, 1e-1))
            self._commit(replace(self.doc.modeset, nu=value), "constraint softness")

        return [
            Property("sigma", lambda: self._length(self.doc.modeset.sigma), sigma),
            Property("spacing", lambda: f"{self.doc.modeset.spacing:.2f} x sigma", spacing),
            Property("softness nu", lambda: f"{self.doc.modeset.nu:.0e}", nu),
            Property("template", lambda: f"{ms.template}  ({ms.n_vertices} vertices)"),
        ]

    # -- axes --------------------------------------------------------------------------

    def _axis(self, axis: Axis) -> List[Property]:
        name = axis.name

        def current():
            return split_rule(self.doc.modeset.axis(name).rule)

        def set_rule(rule, description):
            self._replace(replace(self.doc.modeset.axis(name), rule=rule), f"{name}: {description}")

        def kind(d):
            base, tangential, reversed_ = current()
            new_kind = _cycle(RULE_KINDS, rule_kind(base), d)
            set_rule(join_rule(make_base(new_kind, self.landmarks, base), tangential, reversed_), new_kind)

        def landmark(attribute):
            def step(d):
                base, tangential, reversed_ = current()
                value = _cycle(self.landmarks, getattr(base, attribute), d)
                set_rule(join_rule(replace(base, **{attribute: value}), tangential, reversed_),
                         f"{attribute} = {value}")
            return step

        def center(d):
            base, tangential, reversed_ = current()
            value = _cycle(self.landmarks, base.center[0], d)
            set_rule(join_rule(AwayFrom((value,)), tangential, reversed_), f"center = {value}")

        def toggle(which):
            def step(d):
                base, tangential, reversed_ = current()
                if which == "tangential":
                    tangential = not tangential
                else:
                    reversed_ = not reversed_
                set_rule(join_rule(base, tangential, reversed_), f"{which} toggled")
            return step

        base, tangential, reversed_ = current()
        props = [Property("rule", lambda: rule_kind(current()[0]), kind)]
        if isinstance(base, Between):
            props += [Property("start", lambda: current()[0].start, landmark("start")),
                      Property("end", lambda: current()[0].end, landmark("end"))]
        elif isinstance(base, AwayFrom):
            props.append(Property("center", lambda: "+".join(current()[0].center), center))
        props += [
            Property("tangential", lambda: "yes" if current()[1] else "no", toggle("tangential")),
            Property("reversed", lambda: "yes" if current()[2] else "no", toggle("reversed")),
            Property("reads as", lambda: self.doc.modeset.axis(name).rule.describe()),
            Property("used by", lambda: ", ".join(m.name for m in self.doc.modeset.modes
                                                  if m.axis == name) or "-"),
        ]
        return props

    # -- constraint sets ------------------------------------------------------------------

    def _constraint_set(self, cset: ConstraintSet) -> List[Property]:
        name = cset.name
        item = lambda: self.doc.modeset.constraint_set(name)

        def kind(d):
            value = _cycle(CONSTRAINT_KINDS, item().kind, d)
            self._replace(replace(item(), kind=value), f"{name}: {value}")

        def weight(d):
            self._replace(replace(item(), weight=item().weight * 2.0 ** d), f"{name}: weight")

        meaning = {"anchor": "v = 0 (does not move)",
                   "slide": "n . v = 0 (slides, never bulges)",
                   "rigid": "no strain (moves as a piece)"}
        return [
            Property("kind", lambda: f"{item().kind}   {meaning[item().kind]}", kind),
            Property("weight", lambda: f"{item().weight:g}", weight),
            Property("used by", lambda: ", ".join(m.name for m in self.doc.modeset.modes
                                                  if name in m.constraints) or "-"),
        ]

    # -- modes --------------------------------------------------------------------------

    def _mode(self, mode: Mode) -> List[Property]:
        name = mode.name
        item = lambda: self.doc.modeset.mode(name)

        def kind(d):
            value = _cycle(MODE_KINDS, item().kind, d)
            self._replace(replace(item(), kind=value), f"{name}: {value}")

        def axis(d):
            names = [a.name for a in self.doc.modeset.axes]
            if not names:
                return
            value = _cycle(names, item().axis, d)
            self._replace(replace(item(), axis=value), f"{name}: axis {value}")

        def magnitude(d):
            self._replace(replace(item(), magnitude=round(item().magnitude + 0.25 * d, 4)),
                          f"{name}: magnitude")

        def stretch(i):
            def step(d):
                values = list(item().stretch)
                values[i] = round(values[i] + 0.25 * d, 4)
                self._replace(replace(item(), stretch=tuple(values)), f"{name}: stretch")
            return step

        def sigma(d):
            current = item().sigma or self.doc.modeset.sigma
            self._replace(replace(item(), sigma=current * 1.25 ** d), f"{name}: sigma")

        def uses(set_name):
            def step(d):
                current = item().constraints
                value = (tuple(c for c in current if c != set_name) if set_name in current
                         else current + (set_name,))
                self._replace(replace(item(), constraints=value),
                              f"{name}: {'uses' if set_name in value else 'drops'} {set_name}")
            return step

        def sigma_text():
            own = item().sigma
            return self._length(own) if own is not None else f"default ({self._length(self.doc.modeset.sigma)})"

        props = [
            Property("kind", lambda: item().kind, kind),
            Property("axis", lambda: f"{item().axis or '-'}" + (
                f"   ({self.doc.modeset.axis(item().axis).rule.describe()})"
                if self.doc.modeset.has("axes", item().axis) else ""), axis),
        ]
        if mode.kind == "displacement":
            props.append(Property("magnitude", lambda: f"{item().magnitude:+.2f}", magnitude))
        else:
            for i, label in enumerate(("stretch tangent", "stretch bitangent", "stretch normal")):
                props.append(Property(label, lambda i=i: f"{item().stretch[i]:+.2f}", stretch(i)))
        props.append(Property("sigma", sigma_text, sigma))
        for cset in self.doc.modeset.constraint_sets:
            props.append(Property(f"uses {cset.name}",
                                  lambda c=cset.name: f"{'yes' if c in item().constraints else 'no'}"
                                                      f"   ({self.doc.modeset.constraint_set(c).kind})",
                                  uses(cset.name)))
        return props
