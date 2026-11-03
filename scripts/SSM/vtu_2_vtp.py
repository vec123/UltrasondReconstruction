"""Entry point for the VTU -> VTP conversion: config -> VtuToVtpRunner."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from dotenv import load_dotenv

load_dotenv()

from src.core.config import ConfigClient
from src.config.SSM.schema import VtuToVtpConfig
from src.SSM.runners import VtuToVtpRunner

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "vtu_2_vtp.yaml"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", "-c", type=Path, default=DEFAULT_CONFIG,
        help=f"YAML config file (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args(argv)

    config = ConfigClient().load(args.config, VtuToVtpConfig)
    result = VtuToVtpRunner(config).run()

    print(f"Done, wrote {result.output_vtp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
