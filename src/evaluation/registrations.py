"""The evaluation stages, by the name YAML `stage:` fields use (joined into one
registry by scripts/pipeline/registry.py)."""

from src.evaluation.stages import distance_distribution_overlay, recon_target_distance
from src.shared.core.pipeline import StageSpec

STAGE_SPECS = {
    "recon_target_distance": StageSpec(recon_target_distance.Config, recon_target_distance.run),
    "distance_distribution_overlay": StageSpec(
        distance_distribution_overlay.Config, distance_distribution_overlay.run
    ),
}
