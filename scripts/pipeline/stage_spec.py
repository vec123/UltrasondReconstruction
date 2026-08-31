"""The unit every pipeline (preprocess, SSM fitting, resnet-lddmm training)
registers under a stage name - see `scripts/pipeline/registry.py`.

Split out from `registry.py` so a pipeline's own registrations module (e.g.
`scripts/US/pipeline/registrations.py`) can build `StageSpec`s without
importing the registry itself, which in turn imports every pipeline's
registrations to assemble the shared `STAGE_REGISTRY`.
"""

import dataclasses
from typing import Callable, Optional, Type


@dataclasses.dataclass
class StageSpec:
    config_cls: Type
    run: Callable[[object], object]
    # None -> bind via the generic nested-dataclass binder
    # (scripts.pipeline.config.from_dict). Set this when the stage's own
    # system already has a stricter/richer binder that needs to run instead
    # (e.g. one with cross-field validation or nested unions).
    from_dict: Optional[Callable[[dict], object]] = None
