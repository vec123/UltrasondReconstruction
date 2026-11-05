"""The provenance manifest of an output folder: which stage, config, case and code made its files.

After a stage runs without error, the engine (engine.py) writes one entry into
``manifest.json`` in the stage's output folder, keyed by the stage id, so that
the stages of a pipeline that share a folder each keep their own entry:

    {"<stage id>": {"stage": ..., "config_file": ..., "variables": {...}, "config": {...},
                    "git": {"commit": ..., "dirty": ...}, "started": ..., "minutes": ...}}

A rerun of a stage replaces its entry. The output folder is the Config field
that the StageSpec names (``output``: a field name, a dotted path to a nested
field, or a tuple of them); a field that holds an existing file gives the
file's folder. A field without a value, or a folder or file that the run did
not make, gets no entry.
A manifest that cannot be written gives a warning, never a failed stage.
"""

import dataclasses
import functools
import json
import os
import subprocess
import time
from datetime import datetime

from src.shared.core.config import expand_known
from src.shared.core.paths import get_project_root

MANIFEST_FILE = "manifest.json"


@functools.lru_cache(maxsize=1)
def git_state():
    """{commit, dirty} of the repo, once per process; dirty counts tracked files only."""
    def git(*args):
        return subprocess.run(["git", "-C", str(get_project_root()), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "--short", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _field(cfg, path):
    value = cfg
    for name in path.split("."):
        value = value.get(name) if isinstance(value, dict) else getattr(value, name, None)
        if value is None:
            return None
    return value


def output_folders(cfg, output):
    """The existing folders that ``output`` (see the module docstring) names in ``cfg``."""
    folders = []
    for path in (output,) if isinstance(output, str) else output:
        value = _field(cfg, path)
        if value is None:
            continue
        value = os.fspath(value)
        folder = value if os.path.isdir(value) else os.path.dirname(value) if os.path.isfile(value) else None
        if folder is not None and folder not in folders:
            folders.append(folder)
    return folders


def _plain(cfg):
    return dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else cfg


def entry(stage_name, cfg, config_file, variables, started):
    """The manifest entry of one finished stage run."""
    return {
        "stage": stage_name,
        "config_file": os.path.abspath(config_file),
        "variables": {name: expand_known(str(value), variables) for name, value in sorted(variables.items())},
        "config": _plain(cfg),
        "git": git_state(),
        "started": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "minutes": round((time.time() - started) / 60, 3),
    }


def write_entry(folder, stage_id, record):
    """Sets ``record`` as the entry ``stage_id`` of ``folder``'s manifest, keeping the other entries."""
    path = os.path.join(folder, MANIFEST_FILE)
    manifest = {}
    if os.path.isfile(path):
        with open(path) as f:
            manifest = json.load(f)
    manifest[stage_id] = record
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(temporary, path)


def record(spec, cfg, stage_id, stage_name, config_file, variables, started):
    """Writes the manifest entry of a finished stage into each of its output folders."""
    try:
        folders = output_folders(cfg, getattr(spec, "output", "output_dir"))
        if folders:
            run = entry(stage_name, cfg, config_file, variables, started)
            for folder in folders:
                write_entry(folder, stage_id, run)
    except Exception as exc:  # the stage itself has finished; its files stay valid
        print(f"WARNING: no {MANIFEST_FILE} for stage '{stage_id}': {type(exc).__name__}: {exc}")
