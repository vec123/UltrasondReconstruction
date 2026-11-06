"""Config for scripts/GUIs/place_landmarks.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .model import ModelConfig
from .sample_shapes import SlidersConfig


# --------------------------------------------------------------------------
# scripts/GUIs/place_landmarks.py
# --------------------------------------------------------------------------

# The BabyFM's own 23 anatomical landmarks (outputs/mean_shape/landmark_coordinates.csv),
# used as the default checklist so a hand-placed set on another model - the
# fetus atlas - lines up name for name with the baby's. R/L suffixes are right/
# left as seen on the face (not the viewer's left/right).
DEFAULT_LANDMARK_NAMES: List[str] = [
    "exR", "enR", "n", "enL", "exL",     # eye corners + nasion
    "acR", "aR", "prn", "aL", "acL",     # nose
    "sn",                                 # subnasale
    "chR", "cphR", "ls", "cphL", "chL",  # mouth, upper lip
    "li", "sl", "pg",                     # mouth, lower lip + chin
    "tR", "oiR", "tL", "oiL",             # ears
]


@dataclass
class PlaceLandmarksConfig:
    """Interactive vertex picker: click the mean shape, name what you clicked.

    Built for the Deformetrica atlas (the fetus model), which - unlike the
    .mat - carries no landmark annotation of its own (see the module docstring
    of src.model.deformetrica). Writes name -> vertex-index JSON at
    landmarks_json, the format src.model.landmarks reads back and what
    model.atlas.landmarks in another config (mean_shape_fetus.yaml,
    sample_shapes.yaml) points at to give that atlas model the same
    landmark-driven fields the .mat model has built in.
    """

    # .mat statistical shape model, or a Deformetrica atlas directory - only
    # its mean shape is used here, so atlas options that shape the modes
    # (n_components, variance_threshold, ...) make no difference to this tool.
    shape_model: Path
    landmarks_json: Path
    model: ModelConfig = field(default_factory=ModelConfig)
    # The checklist walked while placing landmarks, in order. Names already in
    # landmarks_json that are not in this list are kept and appended after it,
    # rather than dropped, so re-opening a file never loses a name.
    landmark_names: List[str] = field(default_factory=lambda: list(DEFAULT_LANDMARK_NAMES))
    # The same slider panel sample_shapes.py opens the shape space with, so a
    # landmark can be placed on a deformed instance - useful where an
    # anatomical point is easier to spot away from the mean - rather than only
    # on alpha = 0. A landmark is still just a vertex *index*, so it stays
    # valid at every point in shape space once placed.
    sliders: SlidersConfig = field(default_factory=SlidersConfig)

    camera_axis: str = "+z"
    view_up: str = "+y"
    window_size: List[int] = field(default_factory=lambda: [1400, 900])
    theme: str = "light"

    # curvature shows the anatomy - nose tip, eye sockets, chin - that
    # landmarks sit on, which is what makes them findable by eye. none is a
    # flat surface colour, for a mesh where curvature reads noisy. Computed
    # once from the mean shape and held fixed as the sliders move - it is a
    # guide to where things are, not a live property of the current sample.
    color_by: str = "curvature"
    # Picked-vertex marker radius, in model units. null derives it as 0.6% of
    # the mesh's bounding radius, which reads at a consistent size whether the
    # model spans ~0.013 (the .mat) or ~70 mm (the fetus atlas).
    marker_radius: Optional[float] = None
    # vtkPointPicker tolerance: fraction of the render window's diagonal a
    # click may miss the nearest vertex by and still hit it.
    pick_tolerance: float = 0.01
    # Vertex size used when 'v' switches the mesh to a point cloud - a plain
    # surface can hide a landmark sitting just under a fold or occlude the far
    # side of the shape, where the raw vertices stay visible from anywhere.
    point_size: float = 4.0
    # Geodesic shooting resolution when 'l' toggles the sliders onto the LDDMM
    # parametrization (an atlas only - see src.GUIs.set_landmarks.shape_space).
    # 6 is converged to ~0.14 mm for the fetus atlas, same default as
    # ShapeModelSource.lddmm_time_points in sample_shapes.yaml.
    lddmm_time_points: int = 6

    verbose: bool = True

    COLOR_MODES = ("curvature", "none")
    AXES = ("+x", "-x", "+y", "-y", "+z", "-z")

    def __post_init__(self):
        if self.color_by not in self.COLOR_MODES:
            raise ValueError(
                f"color_by must be one of {list(self.COLOR_MODES)}, got {self.color_by!r}"
            )
        for name in ("camera_axis", "view_up"):
            value = getattr(self, name)
            if value not in self.AXES:
                raise ValueError(f"{name} must be one of {list(self.AXES)}, got {value!r}")
        if self.camera_axis[1] == self.view_up[1]:
            raise ValueError(
                f"camera_axis and view_up cannot share an axis "
                f"({self.camera_axis!r} / {self.view_up!r})"
            )
        if len(self.window_size) != 2:
            raise ValueError(f"window_size must be [width, height], got {self.window_size}")
        if not (0.0 < self.pick_tolerance < 0.5):
            raise ValueError(f"pick_tolerance must be in (0, 0.5), got {self.pick_tolerance}")
        if self.point_size <= 0:
            raise ValueError(f"point_size must be positive, got {self.point_size}")
        if self.lddmm_time_points < 2:
            raise ValueError(f"lddmm_time_points must be at least 2, got {self.lddmm_time_points}")
