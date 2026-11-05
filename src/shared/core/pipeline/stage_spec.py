"""The unit every pipeline (preprocess, SSM fitting, resnet-lddmm training)
registers under a stage name - see `scripts/pipeline/registry.py`, which joins
the `registrations.py` of each layer.

Split out from `registry.py` so a pipeline's own registrations module (e.g.
`src/ultrasound/registrations.py`) can build `StageSpec`s without
importing the registry itself, which in turn imports every pipeline's
registrations to assemble the shared `STAGE_REGISTRY`.

``StageSpec.of_module`` names a stage module instead of importing it: its
Config and run() are imported when a run first uses the stage, so loading the
registry imports no stage and none of their dependencies (torch, vtk, ...).
"""

import dataclasses
from typing import Callable, Optional, Type

from src.shared.core.registry import LazyRegistry


@dataclasses.dataclass
class StageSpec:
    config_cls: Type
    run: Callable[[object], object]
    # None -> bind via the generic nested-dataclass binder
    # (src.shared.core.config.from_dict). Set this when the stage's own
    # system already has a stricter/richer binder that needs to run instead
    # (e.g. one with cross-field validation or nested unions).
    from_dict: Optional[Callable[[dict], object]] = None

    @classmethod
    def of_module(cls, module, config="Config", run="run", from_dict=None):
        """The stage whose Config and run() are ``module``'s ``config`` and ``run``, imported on first use."""
        return ModuleStageSpec(module, config, run, from_dict)


class ModuleStageSpec:
    """A StageSpec that holds only the names of its Config and run(); see StageSpec.of_module."""

    def __init__(self, module, config, run, from_dict=None):
        self.module = module
        self.from_dict = from_dict
        # LazyRegistry imports on resolve and names a moved or renamed target in its error.
        self._targets = LazyRegistry()
        self._targets.register("stage", "config", f"{module}:{config}")
        self._targets.register("stage", "run", f"{module}:{run}")

    @property
    def config_cls(self):
        return self._targets.resolve("stage", "config")

    def run(self, cfg):
        return self._targets.resolve("stage", "run")(cfg)
