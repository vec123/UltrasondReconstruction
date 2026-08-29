"""NoAugmentation: identity transformation (Null Object pattern)."""

from torch import Tensor
from src.resnet_lddmm.augmentation.base import Augmentation


class NoAugmentation(Augmentation):
    """Null Object: returns input unchanged (default, backward-compatible).

    Used when augmentation is disabled (kind: none) or as baseline for comparison.
    Zero computational overhead.
    """

    def __init__(self):
        super().__init__()

    def forward(self, points: Tensor) -> Tensor:
        """Return points unchanged.

        Args:
            points: [B, N, 3]

        Returns:
            [B, N, 3] unchanged
        """
        return points
