"""The LDDMM stages that run from a stage YAML, by name (joined into one registry
by scripts/pipeline/registry.py). Not resnet_flow/registrations.py: that one
registers the model's components (fields, losses, ...), not stages.

The Configs are torch-free and imported here; run() pulls in torch and the
training/inference stack, so that import waits in the function body until a
run reaches the stage. Both Configs bind with their own from_dict, which checks
across fields and accepts aliases.
"""

from src.models.lddmm.resnet_flow.apply.config import ApplyCfg
from src.models.lddmm.resnet_flow.config import ExperimentCfg
from src.shared.core.pipeline import StageSpec


def _run_resnet_lddmm_train(cfg):
    from src.models.lddmm.resnet_flow.runner import run
    return run(cfg)


def _run_resnet_lddmm_apply(cfg):
    from src.models.lddmm.resnet_flow.apply.runner import run
    return run(cfg)


# A trained cohort run as a ShapeModel kind (src/shared/shape_model.py), imported when a model is loaded.
MODEL_KINDS = {"lddmm": "src.models.lddmm.shape_model:FlowShapeModel"}

STAGE_SPECS = {
    "resnet_lddmm_train": StageSpec(
        config_cls=ExperimentCfg,
        run=_run_resnet_lddmm_train,
        from_dict=ExperimentCfg.from_dict,
    ),
    # Full-resolution inference off a resnet_lddmm_train checkpoint - see
    # src/models/lddmm/resnet_flow/apply/config.py. train_config normally
    # points at that run's own output_dir/config.json (written by
    # runner._dump_config), so nothing about the model is restated here.
    "resnet_lddmm_apply": StageSpec(
        config_cls=ApplyCfg,
        run=_run_resnet_lddmm_apply,
        from_dict=ApplyCfg.from_dict,
    ),
}
