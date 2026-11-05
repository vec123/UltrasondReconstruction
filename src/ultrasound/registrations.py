"""Which stage name (as used in YAML `stage:` fields and pipeline.yaml
`stages[].name` entries) maps to which preprocess stage's Config dataclass
and run() function.

The registry engine itself (StageSpec, TypedRegistry-based lookup, YAML <->
Config binding) is generic across every pipeline in this repo and lives in
`src/shared/core/pipeline/`; this module only holds the preprocess-specific
list of what exists. scripts/pipeline/registry.py joins it with the lists of
the other layers.

Each stage is a module of src/ultrasound/stages with the stage's name, its
``Config`` and ``run`` imported when a run first uses it (StageSpec.of_module).
"""

from src.shared.core.pipeline import StageSpec

STAGE_SPECS = {
    name: StageSpec.of_module(f"src.ultrasound.stages.{name}")
    for name in (
        "nrrd_to_vtp",
        "echotexture",
        "point_classify",
        "nrrd_to_world_vtp",
        "roughness_filter",
        "smoothing_filter",
        "align_to_reference",
        "align_to_template",
        "landmark_bbox_cut",
        "landmark_distance_cut",
        "scan_consistency",
        "scan_fusion",
        "landmark_geodesics",
        "similarity_transform",
        "vtp_to_nrrd_overlay",
    )
}
