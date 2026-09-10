"""Entry point for the landmark picker: config -> PlaceLandmarksRunner."""

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
from src.config.SSM.schema import PlaceLandmarksConfig
from src.SSM.runners import PlaceLandmarksRunner

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ssm_gui" / "place_landmarks.yaml"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", "-c", type=Path, default=DEFAULT_CONFIG,
        help=f"YAML config file (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args(argv)

    config = ConfigClient().load(args.config, PlaceLandmarksConfig)
    result = PlaceLandmarksRunner(config).run()

    print(f"\n----------Done, {len(result.landmarks)}/{len(result.names)} landmark(s) placed, "
          f"saved to {config.landmarks_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
