"""VelocityField ABC: interface for flow network components."""

import abc
from typing import Dict, Optional, Tuple
import torch.nn as nn
from torch import Tensor


class VelocityField(nn.Module, abc.ABC):
    """Maps positions to VELOCITIES.

    Never returns positions, never applies an activation to a position sum —
    """

    @abc.abstractmethod
    def forward(
        self, x: Tensor, step: Optional[int] = None, code: Optional[Tensor] = None
    ) -> Tensor:
        """Map positions and optional code to velocity.

        Args:
            x: [B, N, 3] point positions
            step: which Euler step is asking (time-varying fields select block l;
                  stationary fields ignore it)
            code: [B, N_z] per-shape code (fields with NoConditioning ignore it)

        Returns:
            [B, N, 3] velocity field
        """


def split_velocity(field: VelocityField, x: Tensor, step: Optional[int] = None,
                   code: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
    """(velocity, its priced parts by name) of any field, each [B, N, 3].

    A priced part is a share of the velocity that the kinetic cost charges at its
    own price (losses.kinetic.PartKineticLoss): designed modes ("modes",
    fields/modes.py), a frozen prior field ("prior", fields/prior.py), a symmetric
    part ("symmetric", fields/symmetry.py). The rest, ``Trajectory.free_velocities``,
    is charged at kinetic_weight. Fields with parts have a ``split`` method that
    returns them; a plain field has none.
    """
    split = getattr(field, "split", None)
    if split is None:
        return field(x, step=step, code=code), {}
    return split(x, step=step, code=code)
