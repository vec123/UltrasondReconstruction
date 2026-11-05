"""Name -> implementation lookup shared by every pipeline in this repo: the
composition root, the one place that knows the stages of every layer.

Lookup itself (name -> registered item, with a friendly "unknown X -
available: [...]" error on a typo) is `src.shared.core.registry.TypedRegistry`. A
registered item is a `StageSpec` (see `src/shared/core/pipeline/stage_spec.py`).

Each layer owns the list of what it registers, in its `registrations.py`
(src/ultrasound, src/evaluation, src/models/gpmm, src/models/lddmm); this
module only joins them into one registry, so a pipeline YAML can name a
stage from any of them. Loading this registry imports almost no stage code:
most stages name their module (StageSpec.of_module), which is imported when a
run first uses the stage; the others import only their Config here, and their
run() imports the rest when a run reaches it.
"""

from src.evaluation.registrations import STAGE_SPECS as _EVALUATION_STAGE_SPECS
from src.models.gpmm.registrations import MODEL_KINDS as _GPMM_MODEL_KINDS
from src.models.gpmm.registrations import STAGE_SPECS as _GPMM_STAGE_SPECS
from src.models.lddmm.registrations import MODEL_KINDS as _LDDMM_MODEL_KINDS
from src.models.lddmm.registrations import STAGE_SPECS as _LDDMM_STAGE_SPECS
from src.shared.core.registry import TypedRegistry
from src.shared.shape_model import CATEGORY as _MODEL, MODEL_KINDS
from src.ultrasound.registrations import STAGE_SPECS as _US_STAGE_SPECS

STAGE_REGISTRY = TypedRegistry("stage")
for _name, _spec in {
    **_US_STAGE_SPECS,
    **_EVALUATION_STAGE_SPECS,
    **_GPMM_STAGE_SPECS,
    **_LDDMM_STAGE_SPECS,
}.items():
    STAGE_REGISTRY.register(_name)(_spec)

# The shape-model kinds (src/shared/shape_model.py), which load_model finds by a YAML's `kind:`.
for _kind, _target in {**_GPMM_MODEL_KINDS, **_LDDMM_MODEL_KINDS}.items():
    MODEL_KINDS.register(_MODEL, _kind, _target)
