"""The smoke suite: every configs/smoke/*.yaml runs once, in name order, each in its own process.

Each file runs one function (a stage, or in 01 the ultrasound chain) on one
small real input, on the device its YAML sets (the CPU, the GPU for the LDDMM
training and decoding), and writes only into outputs/smoke/ (${SMOKE}
of the smoke files). Later files read the outputs of earlier ones, so the
suite also checks that the stages fit together. Each file also runs alone:
python scripts/run.py configs/smoke/<file>.yaml

    python tests/smoke.py                  every file; deletes outputs/smoke/ first
    python tests/smoke.py --only 01 gpmm   the files whose name holds one of these words;
                                           keeps outputs/smoke/ (earlier files made their inputs)

Prints one row per file (seconds, ok or FAIL, and for a failure the last line of
its log) and exits with 1 when a file fails. The logs are in outputs/smoke/logs/.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "smoke"
OUTPUT = ROOT / "outputs" / "smoke"
TIMEOUT_S = 180  # per file; CLAUDE.md allows no longer run without agreement


def last_line(log_path):
    lines = [line for line in log_path.read_text(errors="replace").splitlines() if line.strip()]
    return lines[-1][:160] if lines else ""


def run_file(config, log_dir, timeout):
    """(seconds, ok, note) of one smoke file, run by scripts/run.py."""
    log_path = log_dir / f"{config.stem}.log"
    started = time.time()
    with open(log_path, "w") as log:
        try:
            code = subprocess.run([sys.executable, str(ROOT / "scripts" / "run.py"), str(config)], cwd=ROOT,
                                  stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = None
    note = f"timeout after {timeout:g} s" if code is None else last_line(log_path)
    return time.time() - started, code == 0, note


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", default=None, help="only the files whose name holds one of these words")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_S, help="seconds per file")
    args = parser.parse_args()

    configs = sorted(CONFIGS.glob("*.yaml"))
    if args.only:
        configs = [config for config in configs if any(word in config.stem for word in args.only)]
    elif OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    if not configs:
        sys.exit(f"No smoke file in {CONFIGS} fits {args.only}.")
    log_dir = OUTPUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    failed, total = [], 0.0
    for config in configs:
        seconds, ok, note = run_file(config, log_dir, args.timeout)
        total += seconds
        print(f"{config.stem:<34} {seconds:6.1f} s  {'ok' if ok else 'FAIL  ' + note}", flush=True)
        if not ok:
            failed.append(config.stem)
    print(f"\n{len(configs) - len(failed)} of {len(configs)} files ok in {total:.0f} s; logs in {log_dir}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
