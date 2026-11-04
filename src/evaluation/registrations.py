"""The evaluation stages, by the name YAML `stage:` fields use (joined into one
registry by scripts/pipeline/registry.py)."""

from src.evaluation.stages import correspondence_views, distance_distribution_overlay, recon_target_distance
from src.shared.core.pipeline import StageSpec

STAGE_SPECS = {
    "correspondence_views": StageSpec(correspondence_views.Config, correspondence_views.run),
    "correspondence_overview": StageSpec(correspondence_views.OverviewConfig, correspondence_views.run_overview),
    "recon_target_distance": StageSpec(recon_target_distance.Config, recon_target_distance.run),
    "distance_distribution_overlay": StageSpec(
        distance_distribution_overlay.Config, distance_distribution_overlay.run
    ),
}
