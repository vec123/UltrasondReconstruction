"""The fitting stages, by name (joined into one registry by scripts/pipeline/registry.py).
The stage module is imported when a run first uses it (StageSpec.of_module)."""

from src.shared.core.pipeline import StageSpec

STAGE_SPECS = {
    "code_fit": StageSpec.of_module("src.shared.fitting.stage"),
}
