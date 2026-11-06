"""Config for the ssm_mean_shape stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .model import ModelConfig


# --------------------------------------------------------------------------
# the ssm_mean_shape stage
# --------------------------------------------------------------------------

@dataclass
class MeanShapeFields:
    """Which per-vertex fields to attach to the mean shape.

    All of them are properties of the *model*, not of any particular fit, so
    this is the reference picture of what the SSM can do and where it is
    confident.
    """

    # sqrt(sum_k lambda_k ||phi_k(v)||^2): the standard deviation of each
    # vertex's position under the model. The heatmap that says where the
    # training population actually varies.
    total_std: bool = True
    # 1-sigma displacement magnitude of each of the leading modes, one field
    # each, so a mode's spatial support is visible. 0 disables.
    mode_count: int = 5
    mode_vectors: bool = False   # the same modes as 3-vectors, for glyphs
    curvature: bool = True       # mean curvature of the mean surface
    landmark_distance: bool = True   # distance to the nearest landmark, and its id
    # One field per landmark, heatmap_<name>: heat released at that landmark and
    # allowed to diffuse over the surface for time t. Unlike a Euclidean radius
    # it follows the geometry, so it does not leak across a fold.
    landmark_heatmaps: bool = True
    # heatmap_landmarks: one field for the whole landmark set.
    combined_heatmap: bool = True
    #   max  element-wise maximum of the per-landmark maps, each normalised to
    #        peak 1, so every landmark reads exactly 1 at its own vertex
    #        whatever its neighbours are doing.
    #   sum  heat from a sum of Dirac deltas. Physically the field of all
    #        landmarks releasing at once, but overlapping bumps add, so a dense
    #        cluster outweighs an isolated landmark.
    combined_mode: str = "max"
    # Spread of the bump, as a length in model units: the diffusion time is
    # sigma^2, so this is directly comparable with the distance thresholds used
    # for matching. null derives it as ten times the mean edge length.
    heat_sigma: Optional[float] = None
    heat_normalize: bool = True      # rescale each heatmap to peak at 1

    COMBINED_MODES = ("max", "sum")

    def __post_init__(self):
        if self.combined_mode not in self.COMBINED_MODES:
            raise ValueError(
                f"fields.combined_mode must be one of {list(self.COMBINED_MODES)}, "
                f"got {self.combined_mode!r}"
            )


@dataclass
class MeanShapeOutputFiles:
    mean_shape: str = "mean_shape.vtp"
    landmarks_csv: str = "landmark_coordinates.csv"
    landmarks_vtp: str = "landmarks.vtp"


@dataclass
class MeanShapeOutputs:
    dir: Path
    files: MeanShapeOutputFiles = field(default_factory=MeanShapeOutputFiles)

    def path(self, key: str) -> Path:
        path = self.dir / getattr(self.files, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class MeanShapeConfig:
    # BabyFaceModel .mat statistical shape model, or the directory of a
    # Deformetrica atlas (see AtlasConfig) - the loader dispatches on which.
    shape_model_mat: Path
    outputs: MeanShapeOutputs
    model: ModelConfig = field(default_factory=ModelConfig)
    fields: MeanShapeFields = field(default_factory=MeanShapeFields)
    verbose: bool = True
