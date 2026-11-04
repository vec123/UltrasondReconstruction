"""Runs one YAML config: a stage (``stage: <name>``) or a pipeline (``stages: [...]``).

The YAML says everything: which registered code runs, its settings, the
${NAME} variables it uses (a top-level ``variables:`` block) and, with a
``targets:`` block, the cases it runs on (src/shared/core/pipeline/targets.py
and batch.py). So the only argument is its path.

Usage:
    python scripts/run.py configs/ultrasound/stages/01_nrrd_to_vtp.yaml
    python scripts/run.py configs/pipelines/end_to_end_casos_mar.yaml
"""

import argparse
import json
import sys

from scripts.pipeline.cli import run_case, run_main
from src.shared.core.pipeline.batch import CasesFailed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to a stage or pipeline YAML config file.")
    parser.add_argument("--case", type=json.loads, metavar="JSON",
                        help="Run only this one case of the YAML's targets: block, given as its variables "
                             "(the batch gives it to the process of each case).")
    args = parser.parse_args()
    if args.case is not None:
        run_case(args.config, args.case)
        return
    try:
        run_main(args.config)
    except CasesFailed as exc:
        print(f"\n{exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
