"""Auto-decoder code implementation: learned per-shape embeddings via lookup table."""

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from src.resnet_lddmm.resnet_flow.codes.base import ShapeCode


class AutoDecoderCodes(ShapeCode):
    """Per-shape learned codes via embedding lookup table.

    Maintains a table of learned vectors, one per shape. Given a batch
    with shape IDs, returns the corresponding codes. Regularizes via L2
    on the embedding table (Manifold Regularization).
    """

    def __init__(self, num_shapes: int, n_z: int = 256, regularization_weight: float = 1.0,
                 pose=None):
        """Initialize auto-decoder code table.

        Args:
            num_shapes: number of distinct shapes (table size)
            n_z: code dimension
            regularization_weight: weight for L2 regularization (typically 1.0)
            pose: optional PoseSource. Auto-decoding a POSE is the same pattern as
                auto-decoding a code -- free per-shape parameters, gradient
                straight from the data term -- so the two compose here rather than
                needing a combined class. penalty() below deliberately does not
                regularise it: a pose is constrained by its own bounds, and an L2
                pull toward zero would fight the fit rather than regularise it.
        """
        super().__init__(pose=pose)
        self.num_shapes = num_shapes
        self.n_z = n_z
        self.regularization_weight = regularization_weight

        self.codes = nn.Embedding(num_shapes, n_z)
        nn.init.normal_(self.codes.weight, mean=0.0, std=0.01)

    def forward(self, batch) -> Optional[Tensor]:
        """Extract codes for shapes in batch.

        Args:
            batch: batch-like object with shape_ids attribute [B] or similar
                   If batch has no shape_ids, returns None (unconditioned)

        Returns:
            [B, n_z] tensor of per-shape codes, or None if batch has no shape_ids
        """
        if not hasattr(batch, "shape_ids"):
            return None

        shape_ids = batch.shape_ids
        if shape_ids is None:
            return None

        # shape_ids can be [B], [B,1], or similar — flatten to 1D
        if isinstance(shape_ids, Tensor):
            shape_ids = shape_ids.view(-1)
            return self.codes(shape_ids)

        return None

    def penalty(self) -> Optional[Tensor]:
        """Compute L2 regularization on embedding table.

        Returns:
            Scalar loss tensor: regularization_weight * ||codes.weight||_2^2
        """
        l2_loss = torch.sum(self.codes.weight ** 2)
        return self.regularization_weight * l2_loss
