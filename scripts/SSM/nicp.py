"""Entry point for the NICP step on its own: config -> NicpRunner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from src.config import ConfigClient, NicpFitConfig
from src.runners import NicpRunner

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "nicp.yaml"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", "-c", type=Path, default=DEFAULT_CONFIG,
        help=f"YAML config file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--run", "-r", action="append", metavar="NAME",
        help="only this run (repeatable); default is every run in the config",
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="print the resolved runs and exit, without fitting",
    )
    args = parser.parse_args(argv)

    client = ConfigClient()
    config = client.load(args.config, NicpFitConfig)
    runner = NicpRunner(config, client=client)

    if args.list:
        # resolving validates every run, so this doubles as a config check
        for spec in runner.resolve_runs():
            print(f"{spec.name}:")
            print(f"  correspondence {spec.correspondence.name} {spec.correspondence.kwargs or ''}")
            print(f"  loss           {spec.loss.name} {spec.loss.kwargs or ''}")
            print(f"  prior          {spec.prior.name} {spec.prior.kwargs or ''}")
            print(f"  solver         {spec.solver.name} {spec.solver.kwargs or ''}")
            print(f"  projection     {spec.projection.name} {spec.projection.kwargs or ''}")
            print(f"  refresh_every  {spec.refresh_every}")
        return 0

    result = runner.run(only=args.run)

    print(f"\n----------Done, {len(result.runs)} run(s), "
          f"wrote {len(result.written_files)} file(s) to {config.outputs.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
