"""Config for scripts/vtu_2_vtp.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class VtuToVtpConfig:
    input_vtu: Path
    output_vtp: Path
    filter_largest_component: bool = True
    verbose: bool = True
