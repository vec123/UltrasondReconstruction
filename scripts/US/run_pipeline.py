"""Runs every stage of the US pipeline in order, from a single YAML config.

Usage:
    python scripts/US/run_pipeline.py --config scripts/US/configs/pipeline.yaml
    python scripts/US/run_pipeline.py --config scripts/US/configs/pipeline.yaml --set SUBJECT_ID=4
"""

import argparse
import os
import sys

# Allow running this script directly from any cwd.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.US.pipeline.cli import pipeline_main  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a pipeline YAML config file.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        dest="set_vars",
        help="Define a ${NAME} interpolation variable used by the config (repeatable), "
        "e.g. --set SUBJECT_ID=4",
    )
    args = parser.parse_args()

    extra_vars = dict(item.split("=", 1) for item in args.set_vars)
    pipeline_main(args.config, extra_vars=extra_vars)


if __name__ == "__main__":
    main()
