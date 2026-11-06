"""Colours, themes and the panel geometry of the shape-space explorer."""

def _rgb(hex_color: str):
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


# One blue ramp, ordered near-zero -> maximum. It is reversed between the
# themes on purpose: a sequential encoding wants its low end to recede towards
# the surface, which is the lightest step on white and the darkest on black.
_BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

# Signed displacement needs a *diverging* ramp instead: the quantity has a
# meaningful zero (on the mean surface) and a sign, and a sequential ramp would
# hide both by mapping "denting in" and "on the mean" to adjacent colours.
# Red out / blue in, through an unsaturated middle that reads as "no change" in
# either theme - the neutral step is the only one that changes between them.
_DIVERGING_LIGHT = ("#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7",
                    "#fddbc7", "#ef8a62", "#b2182b")
# On black the neutral step has to be dark rather than white, or every
# unchanged vertex glows brighter than the deformed ones.
_DIVERGING_DARK = ("#4393c3", "#92c5de", "#5a6b7a", "#3a3a38",
                   "#8a5348", "#d6604d", "#f4a582")

_THEMES = {
    "light": {
        "background": (0.988, 0.988, 0.984),
        "text": (0.043, 0.043, 0.043),
        "muted": (0.322, 0.318, 0.306),
        "rail": (0.847, 0.843, 0.827),          # slider tube and end caps
        "track": (0.902, 0.898, 0.882),         # scrollbar trough
        "knob": _rgb("#2a78d6"),
        "knob_active": _rgb("#1c5cab"),
        "thumb": _rgb("#3987e5"),
        "button": (42, 120, 214),               # 0..255: it is drawn into an image
        "button_active": (28, 92, 171),
        "button_text": (1.0, 1.0, 1.0),
        "surface": (0.78, 0.76, 0.72),          # the shape when color_by: none
        "ghost": (0.45, 0.45, 0.43),
        "ramp": _BLUE,
        "diverging": _DIVERGING_LIGHT,
    },
    "dark": {
        "background": (0.102, 0.102, 0.098),
        "text": (0.92, 0.92, 0.90),
        "muted": (0.66, 0.65, 0.61),
        "rail": (0.26, 0.26, 0.25),
        "track": (0.20, 0.20, 0.19),
        "knob": _rgb("#6da7ec"),
        "knob_active": _rgb("#cde2fb"),
        "thumb": _rgb("#3987e5"),
        "button": (52, 80, 128),
        "button_active": (90, 130, 190),
        "button_text": (1.0, 1.0, 1.0),
        "surface": (0.82, 0.80, 0.76),
        "ghost": (0.55, 0.55, 0.52),
        "ramp": tuple(reversed(_BLUE)),
        "diverging": _DIVERGING_DARK,
    },
}

# Panel geometry, in normalised display coordinates: the slider column and the
# scrollbar beside it. _PANEL_Y is the centre of the top and bottom rows.
_PANEL_X = (0.030, 0.205)
_PANEL_Y = (0.880, 0.235)   # bottom clears the three-button stack below it
_SCROLLBAR_X = (0.216, 0.224)
# Where everything right of the panel starts.
_CONTENT_X = 0.26

# The model dropdown, in pixels down from the top of the content area.
_DROPDOWN_TOP = 30          # gap above the closed bar
_DROPDOWN_BAR_H = 28        # height of the closed bar
_DROPDOWN_PAD = 8           # text inset inside the bar and the rows

#: The one point-data array the mapper colours by. Its *contents* change with
#: the colour mode; its name must not, or the mapper would need re-pointing.
_COLOR_ARRAY = "color"

_COLOR_BAR_TITLES = {
    "displacement": "displacement",
    "signed_displacement": "signed displacement\n(+ out / - in)",
    "total_std": "per-vertex sd",
}
_COLOR_LABELS = {
    "displacement": "COLOR: DISPLACEMENT",
    "signed_displacement": "COLOR: SIGNED",
    "total_std": "COLOR: PER-VERTEX SD",
    "none": "COLOR: OFF",
}

#: Keys the window acts on. Consumed rather than passed through, because
#: vtkInteractorStyleTrackballCamera binds most single letters for itself.
_OUR_KEYS = frozenset({"c", "s", "m", "r", "t", "escape",
                       "bracketleft", "bracketright",
                       "prior", "next", "home", "end"})

_AXIS_VECTORS = {
    "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
}
