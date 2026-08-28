"""Entry point for replaying a similarity transform: config -> SimTransformRunner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from src.config import ConfigClient, SimTransformConfig
from src.runners import SimTransformRunner

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sim_tf.yaml"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", "-c", type=Path, default=DEFAULT_CONFIG,
        help=f"YAML config file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--inverse", action="store_true",
        help="apply the inverse transform, overriding the config",
    )
    args = parser.parse_args(argv)

    config = ConfigClient().load(args.config, SimTransformConfig)
    if args.inverse:
        config.inverse = True

    result = SimTransformRunner(config).run()

    print(f"Done, wrote {len(result.written_files)} mesh(es) to {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
