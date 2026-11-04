"""Augmentation ABC: random group transformations on source shapes (SO(3) or SE(3))."""

import abc
from torch import Tensor
import torch.nn as nn


class Augmentation(nn.Module, abc.ABC):
    """Strategy for applying random group transformations to source point clouds.

    Before the source points enter the neural ODE flow, an optional augmentation
    step can apply a random group element (rotation, rotation+translation) to
    each shape in the batch. The template remains fixed in canonical space.

    This is used as data augmentation: the flow learns to handle shapes at
    various random poses, encouraging pose-invariant learned deformations.

    Semantics:
    - Source points [B, N, 3] → (optionally transformed) → flow
    - Template points [M, 3] → (unchanged) canonical space
    - Gradients flow: loss → augmented_points → transformation → flow params
    """

    def __init__(self):
        super().__init__()
        # The group element drawn by the most recent forward, recorded so a loss
        # term can supervise a predicted pose against the transform that actually
        # produced the input. (None, None) for augmentations that draw nothing.
        self._last_element = (None, None)

    @abc.abstractmethod
    def forward(self, points: Tensor) -> Tensor:
        """Apply augmentation to source point cloud batch.

        Args:
            points: [B, N, 3] source point cloud coordinates

        Returns:
            [B, N, 3] augmented point cloud (same batch size and point count)

        Invariant:
            Output shape matches input shape exactly.
            Gradients flow from output back through transformation to input.
            Subclasses that sample a group element record it via _record_element.
        """

    def _record_element(self, rotation, translation) -> None:
        """Store the group element this forward drew, for supervision terms.

        Args:
            rotation: [B, 3, 3] or None
            translation: [B, 3] or None
        """
        self._last_element = (rotation, translation)

    def last_element(self):
        """The group element applied by the most recent forward.

        This is ground truth in the strict sense — the transform that generated
        the network's input — so a term comparing a predicted pose against it is
        supervised, not self-referential.

        Returns:
            (rotation [B,3,3] or None, translation [B,3] or None)
        """
        return self._last_element
