"""Runs one YAML config: a stage (``stage: <name>``) or a pipeline (``stages: [...]``).

The YAML says everything: which registered code runs, its settings, and the
${NAME} variables it uses (a top-level ``variables:`` block). So the only
argument is its path.

Usage:
    python scripts/run.py configs/preprocess/01_nrrd_to_vtp.yaml
    python scripts/run.py configs/master/end_to_end_casos_mar.yaml
"""

import argparse

from scripts.pipeline.cli import run_main


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to a stage or pipeline YAML config file.")
    args = parser.parse_args()
    run_main(args.config)


if __name__ == "__main__":
    main()
