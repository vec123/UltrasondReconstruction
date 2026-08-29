"""Entry point for the shape-space explorer: config -> SampleShapesRunner."""

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
from src.config.SSM.schema import SampleShapesConfig
from src.SSM.runners import SampleShapesRunner

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sample_shapes.yaml"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", "-c", type=Path, default=DEFAULT_CONFIG,
        help=f"YAML config file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="which entry of 'models' to open, overriding the config (e.g. baby, fetus)",
    )
    parser.add_argument(
        "--screenshot", type=Path, default=None,
        help="write a PNG of the opening frame",
    )
    parser.add_argument(
        "--no-window", action="store_true",
        help="render one frame and exit instead of opening a window",
    )
    args = parser.parse_args(argv)

    config = ConfigClient().load(args.config, SampleShapesConfig)
    if args.model is not None:
        if args.model not in config.models:
            parser.error(
                f"--model {args.model!r} is not declared in {args.config}. "
                f"Available: {sorted(config.models)}"
            )
        config.model = args.model
    if args.screenshot is not None:
        config.screenshot = args.screenshot
    if args.no_window:
        config.interactive = False

    result = SampleShapesRunner(config).run()

    coefficients = ", ".join(f"{value:+.2f}" for value in result.sigmas)
    print(f"\n----------Done, left at [{coefficients}] sd "
          f"(Mahalanobis {result.mahalanobis:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
