"""Point-weighted kinetic energy: a flow cost that depends on where a point STARTED.

``Trajectory.kinetic_energy`` charges every point the same for moving. This module
charges point i by ``w_i``, a per-vertex weight on the TEMPLATE -- "moving the
template here is expensive, there it is cheap".

That is well defined only because the integrator is elementwise: row i of every
trajectory snapshot is, by construction, the image of template vertex i (the
same structural correspondence ``FlowRotationPenalty`` relies on). A weight baked
onto template vertex i therefore stays attached to that point for the whole
path, however far it moves.

Added ON TOP of the plain kinetic term (``loss.kinetic_weight``), never in place
of it: the uniform term keeps every region regularised, so a low-weight region
cannot float free, and this term raises the cost where the field is high. Set
``kinetic_weight: 0`` to use the field-weighted cost alone.

The weight field is resolved once, at build time, by
``runner._add_kinetic_field_term`` (see ``field_weights.py``); this module only
ever sees the resulting tensor.
"""

from typing import Optional

import torch.nn as nn
from torch import Tensor

from src.resnet_lddmm.resnet_flow.losses.terms import LossContext
from src.resnet_lddmm.resnet_flow.trajectory import Trajectory


def weighted_kinetic_energy(trajectory: Trajectory, weights: Tensor) -> Tensor:
    """½ ∫ mean_i w_i ||v_i(t)||² dt: ``Trajectory.kinetic_energy`` with a per-point weight.

    Same reduction as ``Trajectory.kinetic_energy`` -- mean over batch and points,
    sum over steps, times dt -- so ``weights = 1`` reproduces it exactly and a
    mean-1 field keeps the two terms on one scale.

    Args:
        trajectory: velocities [K, B, N, 3] and dt
        weights: [N] or [B, N] non-negative weight per trajectory point, in the
            trajectory's own point order. Moved to the velocities' device/dtype.

    Returns:
        Scalar tensor.
    """
    velocities = trajectory.velocities
    weights = weights.to(device=velocities.device, dtype=velocities.dtype)
    speed_sq = velocities.pow(2).sum(-1)                        # [K, B, N]
    return 0.5 * (speed_sq * weights).mean(dim=(1, 2)).sum() * trajectory.dt


class PointWeightedKineticLoss(nn.Module):
    """Kinetic energy of the forward trajectory, weighted per TEMPLATE vertex.

    Holds the weight for EVERY vertex of the full template and gathers the rows
    this step actually flowed via ``ctx.template_idx`` -- the same subsample draw
    the mapping error used for ``ctx.template_points``. A random subset of a
    mean-1 field therefore gives an unbiased estimate of the full-mesh cost.

    Reads ``ctx.fwd_traj`` only: that is the trajectory that starts on the
    template. Under ``direction: bidirectional`` the backward trajectory starts on
    the TARGET, whose vertices this field does not describe, so it is left to the
    plain kinetic term.

    No learnable parameters: the weights are a buffer, so the term only ever
    pulls on the flow.
    """

    def __init__(self, vertex_weights: Tensor):
        """Initialize with the resolved full-template weight field.

        Args:
            vertex_weights: [N_full] non-negative, finite weight per template
                vertex, in template vertex order.

        Raises:
            ValueError: vertex_weights is not 1-D, or has negative or non-finite
                entries.
        """
        super().__init__()
        if vertex_weights.dim() != 1:
            raise ValueError(
                f"vertex_weights must be one weight per template vertex, [N]; got "
                f"shape {tuple(vertex_weights.shape)}"
            )
        if not vertex_weights.isfinite().all():
            raise ValueError("vertex_weights has non-finite entries")
        if (vertex_weights < 0).any():
            raise ValueError("vertex_weights has negative entries; a cost weight must be >= 0")
        self.register_buffer("vertex_weights", vertex_weights)

    def gather(self, template_idx: Optional[Tensor]) -> Tensor:
        """The weights of the vertices one step flowed, in flowed order.

        Args:
            template_idx: [n] subsample draw into the full template, or None when
                the whole template was flowed.

        Returns:
            [n] (or [N_full]) weights.
        """
        if template_idx is None:
            return self.vertex_weights
        return self.vertex_weights[template_idx.to(self.vertex_weights.device)]

    def forward(self, ctx: LossContext) -> Optional[Tensor]:
        """Compute the point-weighted kinetic energy of ``ctx.fwd_traj``.

        Args:
            ctx: the step's LossContext

        Returns:
            Scalar tensor, or None when there is no forward trajectory.

        Raises:
            ValueError: the gathered weights do not address the trajectory's
                points -- e.g. the template was subsampled but no
                ``template_idx`` reached the context, which would otherwise pair
                every point with some other vertex's weight.
        """
        if ctx.fwd_traj is None:
            return None

        weights = self.gather(ctx.template_idx)
        n_points = ctx.fwd_traj.velocities.shape[2]
        if weights.shape[0] != n_points:
            raise ValueError(
                f"kinetic weight field gives {weights.shape[0]} weights for a "
                f"trajectory of {n_points} points. The template was subsampled "
                f"without its draw reaching LossContext.template_idx, or the field "
                f"describes a different mesh than the one flowed."
            )
        return weighted_kinetic_energy(ctx.fwd_traj, weights)
