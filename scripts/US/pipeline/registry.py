"""Maps a stage name (as used in YAML `stage:` fields and pipeline.yaml
`stages[].name` entries) to its Config dataclass and run() function.

Lookup itself (name -> registered item, with a friendly "unknown X -
available: [...]" error on a typo) is shared with every other pipeline's
component registries via `src.core.registry.TypedRegistry` - only the
per-stage YAML -> Config binding stays local to this pipeline (see
`_bind_stage_config` in `cli.py`), since it needs `from_dict`'s nested-
dataclass and cross-stage-reference support, not `TypedRegistry.build()`'s
flat kwargs-to-constructor call.
"""

import dataclasses
from typing import Callable, Type

from src.core.registry import TypedRegistry

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
    smoothing_filter,
    vtp_to_nrrd_overlay,
)


@dataclasses.dataclass
class StageSpec:
    config_cls: Type
    run: Callable[[object], object]


STAGE_REGISTRY = TypedRegistry("stage")
for _name, _spec in {
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
    "vtp_to_nrrd_overlay": StageSpec(vtp_to_nrrd_overlay.Config, vtp_to_nrrd_overlay.run),
    "recon_target_distance": StageSpec(recon_target_distance.Config, recon_target_distance.run),
}.items():
    STAGE_REGISTRY.register(_name)(_spec)


def stage_fields(config_cls):
    return {f.name for f in dataclasses.fields(config_cls)}
