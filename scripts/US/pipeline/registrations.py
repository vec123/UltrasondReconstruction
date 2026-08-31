"""Which stage name (as used in YAML `stage:` fields and pipeline.yaml
`stages[].name` entries) maps to which preprocess stage's Config dataclass
and run() function.

The registry engine itself (StageSpec, TypedRegistry-based lookup, YAML <->
Config binding) is generic across every pipeline in this repo and lives in
`scripts/pipeline/registry.py`; this module only holds the preprocess-specific
list of what exists.
"""

from scripts.pipeline.stage_spec import StageSpec

from .stages import (
    align_to_reference,
    align_to_template,
    echotexture,
    landmark_bbox_cut,
    landmark_distance_cut,
    landmark_geodesics,
    nrrd_to_vtp,
    nrrd_to_world_vtp,
    point_classify,
    recon_target_distance,
    roughness_filter,
    scan_consistency,
    scan_fusion,
    similarity_transform,
    smoothing_filter,
    vtp_to_nrrd_overlay,
)

STAGE_SPECS = {
    "nrrd_to_vtp": StageSpec(nrrd_to_vtp.Config, nrrd_to_vtp.run),
    "echotexture": StageSpec(echotexture.Config, echotexture.run),
    "point_classify": StageSpec(point_classify.Config, point_classify.run),
    "nrrd_to_world_vtp": StageSpec(nrrd_to_world_vtp.Config, nrrd_to_world_vtp.run),
    "roughness_filter": StageSpec(roughness_filter.Config, roughness_filter.run),
    "smoothing_filter": StageSpec(smoothing_filter.Config, smoothing_filter.run),
    "align_to_reference": StageSpec(align_to_reference.Config, align_to_reference.run),
    "align_to_template": StageSpec(align_to_template.Config, align_to_template.run),
    "landmark_bbox_cut": StageSpec(landmark_bbox_cut.Config, landmark_bbox_cut.run),
    "landmark_distance_cut": StageSpec(landmark_distance_cut.Config, landmark_distance_cut.run),
    "scan_consistency": StageSpec(scan_consistency.Config, scan_consistency.run),
    "scan_fusion": StageSpec(scan_fusion.Config, scan_fusion.run),
    "landmark_geodesics": StageSpec(landmark_geodesics.Config, landmark_geodesics.run),
    "similarity_transform": StageSpec(similarity_transform.Config, similarity_transform.run),
    "vtp_to_nrrd_overlay": StageSpec(vtp_to_nrrd_overlay.Config, vtp_to_nrrd_overlay.run),
    "recon_target_distance": StageSpec(recon_target_distance.Config, recon_target_distance.run),
}
