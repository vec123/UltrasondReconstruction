"""Entry points of run.py, run_stage.py and run_pipeline.py: the engine of
src.shared.core.pipeline, with every registered stage."""

from src.shared.core.pipeline import engine

from .registry import STAGE_REGISTRY


def run_main(config_path, extra_vars=None):
    return engine.run_main(config_path, STAGE_REGISTRY, extra_vars)


def stage_main(config_path, extra_vars=None):
    return engine.stage_main(config_path, STAGE_REGISTRY, extra_vars)


def pipeline_main(pipeline_config_path, extra_vars=None):
    return engine.pipeline_main(pipeline_config_path, STAGE_REGISTRY, extra_vars)
