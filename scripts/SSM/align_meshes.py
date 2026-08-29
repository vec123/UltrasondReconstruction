"""Entry point for the correspondence-free alignment: config -> AlignMeshesRunner."""

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
from src.config.SSM.schema import AlignMeshesConfig
from src.SSM.runners import AlignMeshesRunner

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "align_fetus_to_baby.yaml"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", "-c", type=Path, default=DEFAULT_CONFIG,
        help=f"YAML config file (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args(argv)

    config = ConfigClient().load(args.config, AlignMeshesConfig)
    result = AlignMeshesRunner(config).run()

    print(f"\n----------Done, residual {result.residual:.6g}, scale {result.s:.6g}, "
          f"wrote {len(result.written_files)} file(s) to {config.outputs.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
