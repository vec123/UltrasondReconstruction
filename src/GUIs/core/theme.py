"""Colour themes: one palette per theme, addressed by role rather than by colour.

Components never hard-code a colour; they name a role (``text``, ``knob``,
``marker_current``, ...) and look it up on whichever :class:`Theme` is active,
so a theme switch is one ``apply_theme(theme)`` call per component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

RGB = Tuple[float, float, float]


def rgb(hex_color: str) -> RGB:
    """'#rrggbb' -> (r, g, b) in 0..1, the form VTK colour setters take."""
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


@dataclass(frozen=True)
class Theme:
    """Every colour role a core component may ask for."""

    name: str
    background: RGB
    text: RGB                # primary text: title, status line
    muted: RGB               # secondary text: panels, hints, scalar bar
    rail: RGB                # slider tube and end caps
    track: RGB               # scrollbar trough
    knob: RGB
    knob_active: RGB
    thumb: RGB               # scrollbar thumb
    surface: RGB             # the mesh under flat colouring
    concave: RGB             # diverging ramp: negative end
    neutral: RGB             # ramp midpoint / "nothing here"
    convex: RGB              # diverging ramp: positive end; hot end of a 0..1 field
    marker: RGB
    marker_current: RGB
    marker_pending: RGB
    selection: RGB           # brushed vertices
    vector: RGB              # direction / field arrows
    axis_t: RGB              # frame axes: tangent, bitangent, normal
    axis_b: RGB
    axis_n: RGB
    plane: RGB               # reference planes (e.g. the facial midline)
    accent: RGB              # the selected / focused entry of a list
    warning: RGB             # problems the user should look at
    hover: RGB               # a clickable entry under the pointer

    def color(self, role: str) -> RGB:
        """The colour for ``role``, by name - for components configured with role strings."""
        try:
            return getattr(self, role)
        except AttributeError:
            raise KeyError(f"theme {self.name!r} has no colour role {role!r}") from None


THEMES: Dict[str, Theme] = {
    "light": Theme(
        name="light",
        background=(0.988, 0.988, 0.984),
        text=(0.043, 0.043, 0.043),
        muted=(0.322, 0.318, 0.306),
        rail=(0.847, 0.843, 0.827),
        track=(0.902, 0.898, 0.882),
        knob=rgb("#2a78d6"),
        knob_active=rgb("#1c5cab"),
        thumb=rgb("#3987e5"),
        surface=(0.78, 0.76, 0.72),
        concave=rgb("#3f8fae"),
        neutral=(0.80, 0.78, 0.74),
        convex=rgb("#d9793d"),
        marker=rgb("#8e3b9e"),
        marker_current=rgb("#e0a72c"),
        marker_pending=rgb("#1fb6a6"),
        selection=rgb("#d6409f"),
        vector=(0.16, 0.16, 0.15),
        axis_t=rgb("#d1495b"),
        axis_b=rgb("#3f9c35"),
        axis_n=rgb("#2a78d6"),
        plane=rgb("#8e3b9e"),
        accent=rgb("#1c5cab"),
        warning=rgb("#c0392b"),
        hover=rgb("#d6409f"),
    ),
    "dark": Theme(
        name="dark",
        background=(0.102, 0.102, 0.098),
        text=(0.92, 0.92, 0.90),
        muted=(0.66, 0.65, 0.61),
        rail=(0.26, 0.26, 0.25),
        track=(0.20, 0.20, 0.19),
        knob=rgb("#6da7ec"),
        knob_active=rgb("#cde2fb"),
        thumb=rgb("#3987e5"),
        surface=(0.82, 0.80, 0.76),
        concave=rgb("#6db8d8"),
        neutral=(0.42, 0.41, 0.38),
        convex=rgb("#f0a05c"),
        marker=rgb("#c77dd1"),
        marker_current=rgb("#f0c14b"),
        marker_pending=rgb("#5be0cf"),
        selection=rgb("#f06fbd"),
        vector=(0.92, 0.92, 0.90),
        axis_t=rgb("#f07178"),
        axis_b=rgb("#7fd17a"),
        axis_n=rgb("#6da7ec"),
        plane=rgb("#c77dd1"),
        accent=rgb("#8fc0f5"),
        warning=rgb("#f07178"),
        hover=rgb("#f06fbd"),
    ),
}


def get_theme(name: str) -> Theme:
    """The theme called ``name``.

    Raises:
        ValueError: no theme of that name.
    """
    try:
        return THEMES[name]
    except KeyError:
        raise ValueError(f"unknown theme {name!r}; available: {sorted(THEMES)}") from None


def next_theme(name: str) -> str:
    """The theme after ``name`` in THEMES order - what a toggle key steps to."""
    names = list(THEMES)
    return names[(names.index(name) + 1) % len(names)]
