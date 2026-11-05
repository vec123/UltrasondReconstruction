"""The GPMM stages that run from a stage YAML, by name (joined into one registry
by scripts/pipeline/registry.py).

ssm_fit's Config is a plain dataclass and imported here; its run() pulls in the
whole fitting stack, so that import waits in the function body until a run
reaches the stage. The ssm_posterior module is imported when a run first uses
it (StageSpec.of_module).
"""

from src.models.gpmm.config import PipelineConfig
from src.shared.core.config import ConfigClient
from src.shared.core.pipeline import StageSpec


def _run_ssm_fit(cfg):
    from src.models.gpmm.runners.pipeline import PipelineRunner
    return PipelineRunner(cfg).run()


STAGE_SPECS = {
    "ssm_posterior": StageSpec.of_module("src.models.gpmm.stages.ssm_posterior"),
    "ssm_fit": StageSpec(
        config_cls=PipelineConfig,
        run=_run_ssm_fit,
        from_dict=lambda d: ConfigClient().from_dict(d, PipelineConfig),
    ),
}
