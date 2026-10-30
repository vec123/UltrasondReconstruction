"""Runs a pipeline once per subject id, or once per session found under a cases root.

Subject ids, in this process, filling in ${SUBJECT_ID} each time:
    python scripts/run_pipeline_batch.py --config configs/preprocess/pipeline.yaml \
        --subject-ids 1 2 3 4 5 6 11 13 14

Sessions (one case at one gestational age) of a landmark-predictor dataset,
each in its own run_pipeline.py process, --workers at a time:
    python scripts/run_pipeline_batch.py --config configs/master/gen_fused_scan_targets.yaml \
        --cases-root /home/victor/Projects/UltrasoundLandmarkPredictor/results/inference/Estudio_Dexeus \
        --dataset Estudio_Dexeus --set LANDMARKS_FRAME=LPS --workers 4 \
        --skip-if-exists '${DATA_ROOT}/fused_scan_processing/${DATASET}/${SUBJECT_ID}/${WEEK}/fused_scan_ssm/fused_scan.vtp' \
        --log-dir /home/victor/Escritorio/Data/fused_scan_processing/Estudio_Dexeus/batch_logs

A session is <cases root>/<case>/<week>/*.nrrd, or <cases root>/<case>/*.nrrd
(WEEK empty) for datasets without week folders. Each run gets CASES_ROOT,
DATASET, SUBJECT_ID, WEEK and REFERENCE_SCAN (the first .nrrd in sort order)
plus every --set. Its output goes to <log dir>/<case>_<week>.log, and one row
per session to <log dir>/batch_status.tsv as it finishes.

A subject or session that fails is reported and skipped, so one bad case
doesn't abort the rest of the batch; the script exits non-zero if any failed.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.pipeline.cli import pipeline_main  # noqa: E402
from scripts.pipeline.config import interpolate, path_variables  # noqa: E402

RUN_PIPELINE = Path(__file__).resolve().parent / "run_pipeline.py"
STATUS_COLUMNS = ("case", "week", "n_scans", "exit", "minutes", "status")


def find_sessions(cases_root):
    """[(case, week, scans)] under cases_root: week "" when the case folder holds the scans itself."""
    sessions = []
    for case in sorted(p for p in Path(cases_root).iterdir() if p.is_dir()):
        scans = sorted(case.glob("*.nrrd"))
        if scans:
            sessions.append((case.name, "", scans))
            continue
        for week in sorted(p for p in case.iterdir() if p.is_dir()):
            scans = sorted(week.glob("*.nrrd"))
            if scans:
                sessions.append((case.name, week.name, scans))
    return sessions


def failed_stage(log_path):
    """The label of the last stage the log shows starting, or None."""
    stages = re.findall(r"^Running stage: (.+)$", Path(log_path).read_text(errors="replace"), re.M)
    return stages[-1] if stages else None


def run_session(config, variables, log_path):
    """(exit code, minutes, status) of one run_pipeline.py process."""
    # -u: unbuffered, so a killed run still leaves its full log.
    command = [sys.executable, "-u", str(RUN_PIPELINE), "--config", config]
    for name, value in variables.items():
        command += ["--set", f"{name}={value}"]
    start = time.time()
    with open(log_path, "w") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    minutes = (time.time() - start) / 60
    status = "ok" if code == 0 else f"failed at: {failed_stage(log_path) or 'start-up'}"
    return code, minutes, status


def run_sessions(args, extra_vars):
    sessions = find_sessions(args.cases_root)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "batch_status.tsv"

    jobs, skipped = [], 0
    for case, week, scans in sessions:
        variables = {**extra_vars, "CASES_ROOT": str(args.cases_root), "DATASET": args.dataset,
                     "SUBJECT_ID": case, "WEEK": week, "REFERENCE_SCAN": scans[0].stem}
        if args.skip_if_exists and os.path.exists(
                interpolate(args.skip_if_exists, {**path_variables(), **variables})):
            skipped += 1
            continue
        jobs.append((case, week, len(scans), variables))
    print(f"{len(sessions)} sessions under {args.cases_root}: {skipped} already done, "
          f"{len(jobs)} to run with {args.workers} worker(s); status in {status_path}")

    failed = []
    with open(status_path, "a") as status_file, ThreadPoolExecutor(args.workers) as pool:
        if status_file.tell() == 0:
            status_file.write("\t".join(STATUS_COLUMNS) + "\n")
        futures = {
            pool.submit(run_session, args.config, variables,
                        log_dir / ("_".join(filter(None, (case, week.replace(" ", "_")))) + ".log")):
                (case, week, n_scans)
            for case, week, n_scans, variables in jobs
        }
        for done, future in enumerate(as_completed(futures), 1):
            case, week, n_scans = futures[future]
            code, minutes, status = future.result()
            status_file.write(f"{case}\t{week}\t{n_scans}\t{code}\t{minutes:.1f}\t{status}\n")
            status_file.flush()
            print(f"[{done}/{len(jobs)}] {case} {week}: {status} ({minutes:.1f} min)", flush=True)
            if code != 0:
                failed.append(f"{case} {week}".strip())
    return failed


def run_subjects(args, extra_vars):
    failed = []
    for subject_id in args.subject_ids:
        print(f"\n{'#' * 60}\n# SUBJECT_ID={subject_id}\n{'#' * 60}")
        try:
            pipeline_main(args.config, extra_vars={**extra_vars, "SUBJECT_ID": subject_id})
        except Exception as exc:
            print(f"\nSubject {subject_id} FAILED: {exc}")
            failed.append(subject_id)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a pipeline YAML config file.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--subject-ids", nargs="+",
                      help="Subject ids to run the pipeline for, one at a time, e.g. --subject-ids 1 2 3")
    mode.add_argument("--cases-root", help="Dataset folder holding <case>/<week>/*.nrrd or <case>/*.nrrd.")
    parser.add_argument("--dataset", help="${DATASET} for every session (with --cases-root).")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", dest="set_vars",
                        help="Define a ${NAME} variable for every run (repeatable).")
    parser.add_argument("--workers", type=int, default=1, help="Sessions run at the same time (with --cases-root).")
    parser.add_argument("--skip-if-exists", default=None,
                        help="Path template (${NAME} variables); a session whose path exists is not run.")
    parser.add_argument("--log-dir", help="Per-session logs and batch_status.tsv (with --cases-root).")
    args = parser.parse_args()

    extra_vars = dict(item.split("=", 1) for item in args.set_vars)
    if args.cases_root:
        if not args.dataset or not args.log_dir:
            parser.error("--cases-root needs --dataset and --log-dir")
        failed = run_sessions(args, extra_vars)
    else:
        failed = run_subjects(args, extra_vars)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"Batch finished with failures for: {failed}")
        sys.exit(1)
    print("Batch finished successfully.")


if __name__ == "__main__":
    main()
