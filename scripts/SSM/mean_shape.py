"""Entry point for the mean-shape export: config -> MeanShapeRunner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from src.config.SSM.client import ConfigClient
from src.config.SSM.schema import MeanShapeConfig
from src.SSM.runners import MeanShapeRunner

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "mean_shape.yaml"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", "-c", type=Path, default=DEFAULT_CONFIG,
        help=f"YAML config file (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args(argv)

    config = ConfigClient().load(args.config, MeanShapeConfig)
    result = MeanShapeRunner(config).run()

    print(f"\n----------Done, {len(result.landmarks)} landmark(s), "
          f"{len(result.fields)} field(s), "
          f"wrote {len(result.written_files)} file(s) to {config.outputs.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
