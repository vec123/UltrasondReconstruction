"""Runner: composition root for ResNetLDDMM (STEPS T16).

The ONE place where Registry.create is called and all components are assembled:
config -> shapes -> normalization -> stepper -> orchestrator -> run.

Does NOT import config directly; reads attributes dynamically (layering rule).
"""

from src.models.lddmm.resnet_flow.io import export_trajectory  # noqa: F401  (unused in the old module; kept so the import still happens)
from src.models.lddmm.resnet_flow import registrations  #  Side effect: registers all component Load component registrations

from .build import (  # noqa: F401
    _build_cohort,
    _build_pair,
    _flow_param_groups,
    _set_grid_penalty_weights,
    build,
    trainable_parameters,
)
from .callbacks import _code_prior_em, complete_callbacks, default_callbacks  # noqa: F401
from .common import (  # noqa: F401
    _dump_config,
    _normalization_domain,
    code_size,
    seed_everything,
    set_default_dtype,
)
from .data import _shape_files, cohort_paths, load_cohort, load_pair  # noqa: F401
from .field_weighted_terms import (  # noqa: F401
    FIELD_WEIGHTED_DATA_NAMES,
    _add_cohort_field_weighted_chamfer,
    _add_field_weighted_chamfer,
    _ensure_source_normals,
)
from .loss_terms import (  # noqa: F401
    _add_fshape_term,
    _add_kinetic_field_term,
    _add_landmark_term,
    _add_part_kinetic_terms,
    _build_loss_terms,
    priced_parts,
)
from .model import (  # noqa: F401
    _residual_conditioning,
    build_conditioning,
    build_field,
    build_flow,
    build_network,
)
from .placement import _place_local_mask, _place_modes, _place_symmetry  # noqa: F401
from .prior import (  # noqa: F401
    _check_prior_frame,
    _init_prior_codes,
    _load_prior_field,
    _prior_checkpoint,
    load_prior_config,
)
from .restore import _set_code_prior_from_file, restore_from  # noqa: F401
from .run import _run, run  # noqa: F401
from .sources import (  # noqa: F401
    _build_encoder_code_source,
    _encoder_layers_to_list_of_dicts,
    _validate_pose_config,
    build_code_source,
    build_code_source_cohort,
    build_pose_source,
    init_projections,
)
