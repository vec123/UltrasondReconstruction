"""The point-field stages, by name (joined into one registry by scripts/pipeline/registry.py).
The stage module is imported when a run first uses it (StageSpec.of_module)."""

from src.shared.core.pipeline import StageSpec

STAGE_SPECS = {
    "point_fields": StageSpec.of_module("src.shared.point_fields.stage", output="output_path"),
}
