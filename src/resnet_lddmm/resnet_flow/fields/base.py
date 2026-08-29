"""VelocityField ABC: interface for flow network components."""

import abc
from typing import Optional
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
