"""Configuration of the deformation-mode editor (``scripts/GUIs/edit_deformation_modes.py``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.GUIs.core.theme import THEMES


@dataclass
class DeformationModesConfig:
    """What to edit, where to save it, and how the window behaves.

    Lengths that depend on the template's units (brush radius, default
    kernel width) are given as fractions of the template's bounding radius,
    so one config works for templates in metres, millimetres or anything else.
    """

    # The mesh modes are authored on. Its heatmap_<label> point fields provide
    # the landmarks every axis and landmark selection refers to.
    template: Path
    # The mode-set YAML; created on the first save when it does not exist yet.
    spec: Path

    window_size: List[int] = field(default_factory=lambda: [1500, 950])
    theme: str = "light"
    # Initial brush radius, as a fraction of the template's bounding radius.
    brush_radius: float = 0.08
    # Kernel width for a NEW mode set (an existing file keeps its own), as a
    # fraction of the bounding radius: the length over which a mode's field
    # varies. ~0.07 is a few millimetres on a face.
    sigma: float = 0.07
    # Amplitude sliders: rows on screen, and travel either side of 0 in units
    # where 1 moves a mode's peak by reference_fraction of the bounding radius.
    visible_rows: int = 6
    slider_range: float = 3.0
    reference_fraction: float = 0.05
    # At most this many arrows when drawing a field or an axis.
    arrow_count: int = 250
    pick_tolerance: float = 0.005
    verbose: bool = True

    def __post_init__(self):
        if self.theme not in THEMES:
            raise ValueError(f"theme must be one of {sorted(THEMES)}, got {self.theme!r}")
        for name in ("brush_radius", "sigma", "slider_range", "reference_fraction"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.visible_rows < 1:
            raise ValueError(f"visible_rows must be >= 1, got {self.visible_rows}")
