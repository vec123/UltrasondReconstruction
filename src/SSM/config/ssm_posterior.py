"""Config of the ssm_posterior stage (scripts/US/pipeline/stages/ssm_posterior.py)."""

import dataclasses
from typing import Any, Dict


@dataclasses.dataclass
class NoiseModelConfig:
    """A src.SSM.model.inference.observations noise model, by name, with its kwargs.

    Every kwarg is in the error's units (mm for recon_dist_2_target):
      cutoff       sigma, threshold
      logistic     sigma, sigma_max, threshold, width
      linear       sigma, slope
      exponential  sigma, scale
    """

    name: str = "logistic"
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class SsmPosteriorConfig:
    """Conditions an SSM on the trusted vertices of a registration of its own
    mean shape - see scripts/US/pipeline/stages/ssm_posterior.py."""

    shape_model_mat: str = ""  # prior model (.npz / .mat) whose mean template_vtp is
    template_vtp: str = ""     # the model mean in observed_vtp's frame (resnet_lddmm_apply's world/template_points.vtp)
    observed_vtp: str = ""     # template_vtp after the registration, same vertices (world/deformed.vtp)
    error_vtp: str = ""        # same vertices again, carrying error_array (recon_target_distance's recon_w_distance.vtp)
    error_array: str = "recon_dist_2_target"
    # Where each vertex's error comes from:
    #   "recon_to_target"  error_array of error_vtp, per vertex
    #   "target_to_recon"  every target_vtp point picks its nearest error_vtp vertex; a
    #                      vertex's error is the smallest distance of the points that
    #                      picked it, and a vertex no point picked is unobserved
    error_direction: str = "recon_to_target"
    target_vtp: str = ""       # target_to_recon only: the target in error_vtp's frame (recon_target_distance's target_w_distance.vtp)
    output_dir: str = ""

    noise: NoiseModelConfig = dataclasses.field(default_factory=NoiseModelConfig)
    # Keep about one observation per this many mm (no two kept closer), best-trusted
    # first; neighbouring vertices share their error. 0 keeps every observation.
    observation_spacing: float = 0.0
    # Add the model's own unrepresented residual variance to every observation's.
    include_model_noise: bool = True
    # Pose removed before conditioning: "similarity", "rigid" or "none"; refit
    # against the posterior mean pose_iterations times (1 = pre-alignment only).
    pose: str = "similarity"
    pose_iterations: int = 3
    # Largest Procrustes residual of the model mean onto template_vtp, as a share
    # of template_vtp's bounding-box diagonal, before the two count as different shapes.
    frame_tolerance: float = 1e-3

    n_samples: int = 0  # posterior draws written to output_dir/samples
    seed: int = 0
