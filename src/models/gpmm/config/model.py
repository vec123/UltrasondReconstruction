"""Shared model configs: the atlas, the model and the GPMM output."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class AtlasConfig:
    """How a Deformetrica atlas is turned into a linear SSM.

    Only read when the model path points at an atlas directory rather than a
    .mat file; ignored otherwise, since a .mat already holds a linear model.
    """

    # Deformation kernel width (sigma). null reads 'kwd' from the atlas params.json.
    kernel_width: Optional[float] = None
    # Geodesic shooting resolution. 11 is Deformetrica's default and is
    # converged here to ~0.03 mm; 2 would be a plain linear approximation and
    # is off by several mm on the larger deformations.
    time_points: int = 11
    # Mode truncation. null/1.0 keeps every non-degenerate mode (at most
    # n_subjects - 1 of them). The stricter of the two limits wins.
    n_components: Optional[int] = None
    variance_threshold: float = 1.0
    # Cache the shot decomposition next to the atlas - shooting the population
    # takes ~90 s, reading it back takes under a second.
    cache: bool = True
    # name -> vertex-index JSON annotating the atlas's mean shape, written by
    # scripts/GUIs/place_landmarks.py (see src/model/landmarks.py). null leaves the
    # atlas with no landmarks, as before - every landmark-driven field then
    # stays off, same as documented in configs/mean_shape_fetus.yaml.
    landmarks: Optional[Path] = None

    def to_options(self) -> Dict[str, Any]:
        """The keyword arguments load_shape_model() hands to the atlas backend."""
        return {
            "kernel_width": self.kernel_width,
            "time_points": self.time_points,
            "n_components": self.n_components,
            "variance_threshold": self.variance_threshold,
            "cache": self.cache,
            "landmarks": self.landmarks,
        }


@dataclass
class ModelConfig:
    normalize: bool = False      # normalize the SSM to unit RMS scale on load
    atlas: AtlasConfig = field(default_factory=AtlasConfig)


@dataclass
class GpmmModelConfig:
    """Configuration for building one Gaussian Process Morphable Model.

    The kernel tree is intentionally untyped internally: composite kernels
    recurse through arbitrary ``children`` and each registered kernel owns its
    own kwargs validation. ``ConfigClient`` still validates the model-level
    fields and interpolates paths before ``build_model`` receives the mapping.
    """

    domain: Dict[str, Any]
    kernel: Dict[str, Any] = field(default_factory=lambda: {"name": "pca"})
    approximation: Optional[Dict[str, Any]] = None
    truncation: Optional[Dict[str, Any]] = None
    inference: Optional[Dict[str, Any]] = None
    normalize: bool = True
    output: "GpmmModelOutputConfig" = field(default_factory=lambda: GpmmModelOutputConfig())


@dataclass
class GpmmModelOutputConfig:
    """Destination for the serialized model archive."""

    dir: Path = Path("outputs/gpmm")
    filename: str = "model.npz"

    @property
    def path(self) -> Path:
        return self.dir / self.filename
