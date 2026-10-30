"""Config for scripts/sim_tf.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# --------------------------------------------------------------------------
# scripts/sim_tf.py
# --------------------------------------------------------------------------

@dataclass
class SimTransformInputs:
    """Meshes to transform: either a whole directory or an explicit list."""

    dir: Optional[Path] = None
    pattern: str = "*.vtp"
    exclude: List[str] = field(default_factory=list)   # file names skipped when scanning dir
    files: List[Path] = field(default_factory=list)

    def __post_init__(self):
        if (self.dir is None) == (not self.files):
            raise ValueError("set exactly one of inputs.dir or inputs.files")


@dataclass
class SimTransformConfig:
    # .npz holding R (3,3), t (3,) - 'T' is also accepted - and s (scalar).
    # Written by scripts/pipeline.py (gmm_2_template.npz) and by
    # scripts/align_landmarks.py (rigid_transform.npz).
    transform: Path
    inputs: SimTransformInputs
    output_dir: Path
    inverse: bool = False              # apply the inverse of the saved transform
    # Appended to each stem. Omit it or set null to keep the original file name
    # (only allowed when output_dir differs from the source directory).
    suffix: Optional[str] = None
    verbose: bool = True
