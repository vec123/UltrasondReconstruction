"""Name -> implementation lookup shared by every pipeline in this repo
(preprocess, SSM fitting, resnet-lddmm training).

Lookup itself (name -> registered item, with a friendly "unknown X -
available: [...]" error on a typo) is `src.core.registry.TypedRegistry`. A
registered item is a `StageSpec` (see `stage_spec.py`).

Each pipeline owns the list of what it registers (e.g.
`scripts/US/pipeline/registrations.py` for preprocess); this module only
aggregates them into one shared registry so a master pipeline.yaml can name
a stage from any of them.

`ssm_fit`'s and `resnet_lddmm_train`'s `config_cls`/`from_dict` are cheap
(plain dataclasses, no optional deps) and imported eagerly like every
preprocess stage's; their `run` underneath pulls in a full training/fitting
stack (torch, for resnet-lddmm), so - same reasoning as
`src.core.registry.LazyRegistry` elsewhere in this repo - that import is
deferred to the function body, only paid by a pipeline run that actually
reaches that stage.
"""

from src.core.registry import TypedRegistry
from src.config.SSM.client import ConfigClient
from src.config.SSM.schema import PipelineConfig
from src.resnet_lddmm.resnet_flow.apply.config import ApplyCfg
from src.resnet_lddmm.resnet_flow.config import ExperimentCfg

from scripts.US.pipeline.registrations import STAGE_SPECS as _US_STAGE_SPECS
from .stage_spec import StageSpec


def _run_ssm_fit(cfg):
    from src.SSM.runners.pipeline import PipelineRunner
    return PipelineRunner(cfg).run()


def _run_resnet_lddmm_train(cfg):
    from src.resnet_lddmm.resnet_flow.runner import run
    return run(cfg)


def _run_resnet_lddmm_apply(cfg):
    from src.resnet_lddmm.resnet_flow.apply.runner import run
    return run(cfg)


STAGE_REGISTRY = TypedRegistry("stage")
for _name, _spec in {
    **_US_STAGE_SPECS,
    "ssm_fit": StageSpec(
        config_cls=PipelineConfig,
        run=_run_ssm_fit,
        from_dict=lambda d: ConfigClient().from_dict(d, PipelineConfig),
    ),
    "resnet_lddmm_train": StageSpec(
        config_cls=ExperimentCfg,
        run=_run_resnet_lddmm_train,
        from_dict=ExperimentCfg.from_dict,
    ),
    # Full-resolution inference off a resnet_lddmm_train checkpoint - see
    # src/resnet_lddmm/resnet_flow/apply/config.py. train_config normally
    # points at that run's own output_dir/config.json (written by
    # runner._dump_config), so nothing about the model is restated here.
    "resnet_lddmm_apply": StageSpec(
        config_cls=ApplyCfg,
        run=_run_resnet_lddmm_apply,
        from_dict=ApplyCfg.from_dict,
    ),
}.items():
    STAGE_REGISTRY.register(_name)(_spec)
