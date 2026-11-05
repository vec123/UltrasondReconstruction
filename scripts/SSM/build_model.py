"""Build and save a GPMM from a typed YAML configuration.

The same as ``python scripts/run.py <model config>.yaml`` (the ssm_build stage,
src/models/gpmm/stages/build.py), which this script calls.

Usage:
    python -m scripts.SSM.build_model <model config>.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.gpmm.config import GpmmModelConfig
from src.models.gpmm.stages import build
from src.shared.core.config import ConfigClient


def build_and_save(config_path, *, verbose: bool = True):
    """Build the configured model and save its content-addressed NPZ archive."""
    project_root = Path(__file__).resolve().parents[2]
    config = ConfigClient(project_root=project_root).load(Path(config_path), GpmmModelConfig)
    return build.build_and_save(config, verbose=verbose)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="GPMM YAML configuration path")
    args = parser.parse_args(argv)
    build_and_save(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
