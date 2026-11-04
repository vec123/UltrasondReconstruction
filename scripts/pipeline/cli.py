"""Entry points of run.py, run_stage.py and run_pipeline.py: the engine of
src.shared.core.pipeline, with every registered stage."""

import json
import sys
from pathlib import Path

from src.shared.core.pipeline import engine

from .registry import STAGE_REGISTRY

RUN_PY = Path(__file__).resolve().parents[1] / "run.py"


def case_command(config_path, case):
    """The command line that runs one case of a ``targets:`` YAML in its own process."""
    # -u: unbuffered, so a killed run still leaves its full log.
    return [sys.executable, "-u", str(RUN_PY), str(config_path), "--case", json.dumps(case)]


def run_main(config_path, extra_vars=None):
    return engine.run_main(config_path, STAGE_REGISTRY, extra_vars, case_command)


def run_case(config_path, case):
    return engine.run_case(config_path, STAGE_REGISTRY, case)


def stage_main(config_path, extra_vars=None):
    return engine.stage_main(config_path, STAGE_REGISTRY, extra_vars)


def pipeline_main(pipeline_config_path, extra_vars=None):
    return engine.pipeline_main(pipeline_config_path, STAGE_REGISTRY, extra_vars)
