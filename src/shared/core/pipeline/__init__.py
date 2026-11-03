"""The stage pipeline: StageSpec (one stage's Config and run()) and the engine
that binds stage YAMLs and runs them (engine.stage_main, engine.pipeline_main)."""

from src.shared.core.pipeline.engine import no_pipeline_context, pipeline_main, stage_main
from src.shared.core.pipeline.stage_spec import StageSpec

__all__ = ["StageSpec", "no_pipeline_context", "pipeline_main", "stage_main"]
