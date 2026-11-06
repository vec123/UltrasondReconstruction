"""Config for the align_landmarks stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# --------------------------------------------------------------------------
# the align_landmarks stage
# --------------------------------------------------------------------------

@dataclass
class AlignLandmarksOutputs:
    dir: Path
    transform: str = "rigid_transform.npz"
    original_mesh: str = "original_mean_mesh.vtp"
    aligned_mesh: str = "aligned_mean_mesh.vtp"

    def path(self, key: str) -> Path:
        return self.dir / getattr(self, key)


@dataclass
class AlignLandmarksConfig:
    shape_model_mat: Path
    output: AlignLandmarksOutputs
    # Either give the targets inline (name -> [x, y, z]) or point at a landmark JSON.
    target_landmarks: Dict[str, List[float]] = field(default_factory=dict)
    target_landmarks_json: Optional[Path] = None
    allow_scale: bool = True
    verbose: bool = True

    def __post_init__(self):
        if bool(self.target_landmarks) == (self.target_landmarks_json is not None):
            raise ValueError(
                "set exactly one of 'target_landmarks' (inline) or 'target_landmarks_json'"
            )
