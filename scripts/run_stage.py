"""Runs a single pipeline stage from a YAML config.

Usage:
    python scripts/run_stage.py --config configs/preprocess/01_nrrd_to_vtp.yaml
"""

import argparse

from scripts.pipeline.cli import stage_main  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a stage YAML config file.")
    args = parser.parse_args()

    stage_main(args.config)


if __name__ == "__main__":
    main()
