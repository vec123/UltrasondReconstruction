"""The evaluation stages, by the name YAML `stage:` fields use (joined into one
registry by scripts/pipeline/registry.py). Each stage module is imported when a
run first uses it (StageSpec.of_module)."""

from src.shared.core.pipeline import StageSpec

_STAGES = "src.evaluation.stages"

STAGE_SPECS = {
    "correspondence_views": StageSpec.of_module(f"{_STAGES}.correspondence_views", output="output_path"),
    "correspondence_overview": StageSpec.of_module(f"{_STAGES}.correspondence_views", config="OverviewConfig",
                                                   run="run_overview", output="output_path"),
    "recon_target_distance": StageSpec.of_module(f"{_STAGES}.recon_target_distance"),
    "distance_distribution_overlay": StageSpec.of_module(f"{_STAGES}.distance_distribution_overlay"),
    "compare": StageSpec.of_module(f"{_STAGES}.compare", patterns=("models",)),
}
