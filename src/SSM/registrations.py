"""The GPMM stages that run from a stage YAML, by name (joined into one registry
by scripts/pipeline/registry.py)."""

from src.SSM.stages import ssm_posterior
from src.shared.core.pipeline import StageSpec

STAGE_SPECS = {
    "ssm_posterior": StageSpec(ssm_posterior.Config, ssm_posterior.run),
}
