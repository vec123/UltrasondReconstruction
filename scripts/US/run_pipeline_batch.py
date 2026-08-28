"""Runs the full pipeline once per subject id, filling in ${SUBJECT_ID} each time.

Usage:
    python scripts/US/run_pipeline_batch.py --config scripts/US/configs/pipeline.yaml \
        --subject-ids 1 2 3 4 5 6 11 13 14

A subject that raises is reported and skipped, so one bad case doesn't abort
the rest of the batch; the script exits non-zero if any subject failed.
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
        "--subject-ids",
        required=True,
        nargs="+",
        help="Subject ids to run the pipeline for, one at a time, e.g. --subject-ids 1 2 3 4 5 6 11 13 14",
    )
    args = parser.parse_args()

    failed = []
    for subject_id in args.subject_ids:
        print(f"\n{'#' * 60}\n# SUBJECT_ID={subject_id}\n{'#' * 60}")
        try:
            pipeline_main(args.config, extra_vars={"SUBJECT_ID": subject_id})
        except Exception as exc:
            print(f"\nSubject {subject_id} FAILED: {exc}")
            failed.append(subject_id)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"Batch finished with failures for subject(s): {failed}")
        sys.exit(1)
    print(f"Batch finished successfully for subject(s): {args.subject_ids}")


if __name__ == "__main__":
    main()
