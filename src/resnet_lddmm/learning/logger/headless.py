"""Headless / remote-HPC logging.

Mirror ``stdout``+``stderr`` to a timestamped, flushed log file so a run on a remote node
without a GUI stays easy to inspect (``tail -f`` the log). Headless mode is auto-detected
when ``stdout`` is not a TTY (SLURM, ``nohup``, pipes); override explicitly via the
``remote`` argument or the ``HEADLESS`` environment variable.

Usage (near the start of a script, once ``log_dir`` is known)::

    from src.learning.logger.headless import enable_headless
    enable_headless(OUTPUT_DIR, remote=REMOTE, name="my_script")

``matplotlib`` is already pinned to the ``Agg`` backend where these scripts plot, so no
display is needed; this module only handles the console/log stream.
"""

import os
import sys
import datetime


class _TimestampedTee:
    """Write to a console stream and, line-by-line + timestamped, to a log file."""

    def __init__(self, console, logfile):
        self._console = console
        self._logfile = logfile
        self._buf = ""

    def write(self, s):
        self._console.write(s)
        self._console.flush()
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._logfile.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | {line}\n")
        self._logfile.flush()
        return len(s)

    def flush(self):
        self._console.flush()
        self._logfile.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):        # delegate anything else to the console stream
        return getattr(self._console, name)


def resolve_remote(remote=None):
    """Resolve the headless toggle. ``None`` -> auto (``HEADLESS`` env var, else "stdout is
    not a TTY"). A bool forces the choice."""
    if remote is not None:
        return bool(remote)
    env = os.environ.get("HEADLESS")
    if env is not None:
        return env.strip().lower() not in ("", "0", "false", "no")
    try:
        return not sys.stdout.isatty()
    except Exception:
        return True


def enable_headless(log_dir, remote=None, name="run"):
    """If headless, tee ``stdout``/``stderr`` to ``<log_dir>/<name>_<timestamp>.log``
    (timestamped, flushed). Returns ``(is_headless, log_path_or_None)``. Idempotent-ish:
    only wraps streams that are not already a ``_TimestampedTee``."""
    is_remote = resolve_remote(remote)
    if not is_remote:
        return False, None
    if isinstance(sys.stdout, _TimestampedTee):
        return True, getattr(sys.stdout, "_log_path", None)
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{ts}.log")
    f = open(log_path, "a", buffering=1)            # line-buffered text file
    tee_out = _TimestampedTee(sys.__stdout__ or sys.stdout, f)
    tee_out._log_path = log_path
    sys.stdout = tee_out
    sys.stderr = _TimestampedTee(sys.__stderr__ or sys.stderr, f)
    print(f"[headless] mode=remote  mirroring stdout/stderr -> {log_path}")
    return True, log_path
