"""Entry point for atlas resampling: config -> UpsampleAtlasRunner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from src.config import ConfigClient, UpsampleAtlasConfig
from src.runners import UpsampleAtlasRunner

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "upsample_fetus_hd.yaml"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", "-c", type=Path, default=DEFAULT_CONFIG,
        help=f"YAML config file (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args(argv)

    config = ConfigClient().load(args.config, UpsampleAtlasConfig)
    result = UpsampleAtlasRunner(config).run()

    print(f"\n----------Done, {result.source_vertices} -> {result.output_vertices} vertices "
          f"({result.output_vertices / result.source_vertices:.2f}x), "
          f"{result.output_faces} triangles, written to {result.output_dir}"
          f"{' (cache warmed)' if result.cache_warmed else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
