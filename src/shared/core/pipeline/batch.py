"""Runs the cases of a ``targets:`` block (targets.py): in this process, or each in its own process.

    batch:
      workers: 4
      log_dir: ${DATA_ROOT}/fused_scan_processing/${DATASET}/batch_logs
      skip_if_exists: ${DATA_ROOT}/fused_scan_processing/${DATASET}/${SUBJECT_ID}/${WEEK}/fused_scan_ssm/fused_scan.vtp

Without ``batch:``, the cases run one after the other in this process. With
it, ``workers`` cases run at the same time, each in its own process. The output
of a case goes to ``<log_dir>/<case name>.log``, and one row per case goes to
``<log_dir>/batch_status.tsv`` when the case is finished. A case whose
``skip_if_exists`` path exists does not run. A case that fails does not stop
the others; CasesFailed names them at the end.
"""

import os
import re
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.shared.core.config import interpolate_paths
from src.shared.core.pipeline.cases import SUBJECT_ID, WEEK

KEYS = ("workers", "log_dir", "skip_if_exists")
STATUS_FILE = "batch_status.tsv"
STATUS_COLUMNS = ("case", "week", "n_scans", "exit", "minutes", "status")


class CasesFailed(RuntimeError):
    """Some cases of a batch failed; the others ran."""


def failed_stage(log_path):
    """The label of the last stage the log shows starting, or None.

    A pipeline writes ``Running stage: <label>``, a stage file alone ``Running stage '<name>' from ...``.
    """
    stages = re.findall(r"^Running stage(?:: (.+)| '([^']+)' from )", Path(log_path).read_text(errors="replace"), re.M)
    return "".join(stages[-1]) if stages else None


def run_logged(command, log_path):
    """(exit code, minutes, status) of one process, its output in ``log_path``."""
    start = time.time()
    with open(log_path, "w") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    minutes = (time.time() - start) / 60
    status = "ok" if code == 0 else f"failed at: {failed_stage(log_path) or 'start-up'}"
    return code, minutes, status


def run_in_process(cases, run_case):
    """Runs ``run_case(case)`` for each case, one after the other; the names of the failed ones."""
    failed = []
    for number, case in enumerate(cases, 1):
        print(f"\n{'#' * 60}\n# [{number}/{len(cases)}] {case.name}\n{'#' * 60}")
        try:
            run_case(case)
        except Exception as exc:
            traceback.print_exc()
            print(f"\nCase {case.name} FAILED: {exc}")
            failed.append(case.name)
    return failed


def run_in_processes(cases, batch, run_variables, case_variables, command, source):
    """Runs ``command(case)`` for each case, ``batch['workers']`` at a time; the names of the failed ones.

    Args:
        run_variables: the variables of the run, for ``log_dir``
        case_variables: case -> the variables of that case, for ``skip_if_exists``
        command: case -> the command line of its process
    """
    if not isinstance(batch, dict):
        raise ValueError(f"'batch:' in '{source}' must be a mapping, got {type(batch).__name__}.")
    unknown = sorted(set(batch) - set(KEYS))
    if unknown:
        raise ValueError(f"'batch:' in '{source}' has unknown key(s) {unknown}; the keys are {list(KEYS)}.")
    if "log_dir" not in batch:
        raise ValueError(f"'batch:' in '{source}' needs 'log_dir:', for the log of each case.")
    workers = int(batch.get("workers", 1))
    log_dir = Path(interpolate_paths(batch["log_dir"], run_variables))
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / STATUS_FILE

    skip = batch.get("skip_if_exists")
    jobs = [case for case in cases
            if not (skip and os.path.exists(interpolate_paths(skip, case_variables(case))))]
    print(f"{len(cases)} cases: {len(cases) - len(jobs)} already done, "
          f"{len(jobs)} to run with {workers} worker(s); status in {status_path}")

    failed = []
    with open(status_path, "a") as status_file, ThreadPoolExecutor(workers) as pool:
        if status_file.tell() == 0:
            status_file.write("\t".join(STATUS_COLUMNS) + "\n")
        futures = {pool.submit(run_logged, command(case), log_dir / f"{case.name}.log"): case for case in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            case = futures[future]
            code, minutes, status = future.result()
            subject, week = case.values.get(SUBJECT_ID, case.name), case.values.get(WEEK, "")
            status_file.write(f"{subject}\t{week}\t{case.n_files}\t{code}\t{minutes:.1f}\t{status}\n")
            status_file.flush()
            print(f"[{done}/{len(jobs)}] {case.name}: {status} ({minutes:.1f} min)", flush=True)
            if code != 0:
                failed.append(case.name)
    return failed
