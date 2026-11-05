"""The GPMM stages that run from a stage YAML, by name (joined into one registry
by scripts/pipeline/registry.py).

The runner stages bind their Config with ConfigClient, as the scripts in
scripts/SSM do. Their Configs are plain dataclasses and imported here; run()
imports the runner, and with it the fitting stack, only when a run reaches the
stage. The ssm_posterior module is imported when a run first uses it
(StageSpec.of_module).
"""

from src.models.gpmm.config import (
    AlignLandmarksConfig,
    AlignMeshesConfig,
    GpmmModelConfig,
    MeanShapeConfig,
    NicpFitConfig,
    PipelineConfig,
)
from src.shared.core.config import ConfigClient
from src.shared.core.pipeline import StageSpec


def _runner_stage(config_cls, runner, output):
    """The stage that runs ``src.models.gpmm.runners.<runner>(cfg).run()``; ``output`` is its output field."""
    def run(cfg):
        from src.models.gpmm import runners
        return getattr(runners, runner)(cfg).run()
    return StageSpec(config_cls, run, from_dict=lambda d: ConfigClient().from_dict(d, config_cls), output=output)


def _run_build(cfg):
    from src.models.gpmm.stages.build import run
    return run(cfg)


# The GPMM as a ShapeModel kind (src/shared/shape_model.py), imported when a model is loaded.
MODEL_KINDS = {"gpmm": "src.models.gpmm.shape_model:GpmmShapeModel"}

STAGE_SPECS = {
    "ssm_posterior": StageSpec.of_module("src.models.gpmm.stages.ssm_posterior"),
    "ssm_fit": _runner_stage(PipelineConfig, "PipelineRunner", "outputs.dir"),
    "ssm_nicp": _runner_stage(NicpFitConfig, "NicpRunner", "outputs.dir"),
    "ssm_mean_shape": _runner_stage(MeanShapeConfig, "MeanShapeRunner", "outputs.dir"),
    "align_landmarks": _runner_stage(AlignLandmarksConfig, "AlignLandmarksRunner", "output.dir"),
    "align_meshes": _runner_stage(AlignMeshesConfig, "AlignMeshesRunner", "outputs.dir"),
    "ssm_build": StageSpec(GpmmModelConfig, _run_build,
                           from_dict=lambda d: ConfigClient().from_dict(d, GpmmModelConfig), output="output.dir"),
}
