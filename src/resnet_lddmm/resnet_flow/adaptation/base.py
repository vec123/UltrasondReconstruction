"""Adaptation ABC: random group transformations on source shapes (SO(3) or SE(3))."""

import abc
from torch import Tensor
import torch.nn as nn


class Adaptation(nn.Module, abc.ABC):
    """Strategy for applying random group transformations to source point clouds.

    Before the source points enter the neural ODE flow, an optional adaptation
    step can apply a random group element (rotation, rotation+translation) to
    each shape in the batch. The template remains fixed in canonical space.

    This is used to augment training: the flow learns to handle shapes at
    various random poses, encouraging pose-invariant learned deformations.

    Semantics:
    - Source points [B, N, 3] → (optionally transformed) → flow
    - Template points [M, 3] → (unchanged) canonical space
    - Gradients flow: loss → adapted_points → transformation → flow params
    """

    @abc.abstractmethod
    def forward(self, points: Tensor) -> Tensor:
        """Apply adaptation to source point cloud batch.

        Args:
            points: [B, N, 3] source point cloud coordinates

        Returns:
            [B, N, 3] adapted point cloud (same batch size and point count)

        Invariant:
            Output shape matches input shape exactly.
            Gradients flow from output back through transformation to input.
        """
