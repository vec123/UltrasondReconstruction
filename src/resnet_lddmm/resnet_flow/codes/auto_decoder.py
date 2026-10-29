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
                 pose=None, init_std: float = 0.01):
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
        nn.init.normal_(self.codes.weight, mean=0.0, std=init_std)
        # Per-entry penalty weights [num_shapes, n_z], or None for a plain L2. Not
        # persistent: they are recomputed from the config and data, not trained, so
        # checkpoints and apply stay unchanged.
        self.register_buffer("penalty_weights", None, persistent=False)
        # The code the penalty pulls towards [n_z], or None for zero. Not persistent either.
        self.register_buffer("penalty_center", None, persistent=False)
        # A full precision [n_z, n_z] in place of the per-entry L2, or None. Not persistent
        # either: a learned prior (callbacks.schedules.CodePriorEM) writes its own files.
        self.register_buffer("penalty_precision", None, persistent=False)

    def set_penalty_weights(self, weights: Tensor):
        """Weight the L2 penalty per shape and code entry (e.g. per grid node).

        Args:
            weights: [num_shapes, n_z], non-negative
        """
        if tuple(weights.shape) != tuple(self.codes.weight.shape):
            raise ValueError(f"penalty weights {tuple(weights.shape)} do not match the "
                             f"code table {tuple(self.codes.weight.shape)}")
        self.penalty_weights = weights.to(self.codes.weight)

    def set_penalty_center(self, center: Tensor):
        """Pull the codes towards `center` [n_z] instead of zero (e.g. a trained run's code mean).

        Args:
            center: [n_z]
        """
        if tuple(center.shape) != (self.n_z,):
            raise ValueError(f"penalty center {tuple(center.shape)} does not match n_z={self.n_z}")
        self.penalty_center = center.to(self.codes.weight)

    def set_penalty_precision(self, center: Tensor, precision: Tensor):
        """Penalise each code by ``(c - center)^T P (c - center)``: a Gaussian prior with a full covariance.

        Args:
            center: [n_z]
            precision: [n_z, n_z], symmetric positive semi-definite

        Raises:
            ValueError: per-entry penalty weights are set; the two do not combine.
        """
        if tuple(precision.shape) != (self.n_z, self.n_z):
            raise ValueError(f"penalty precision {tuple(precision.shape)} does not match n_z={self.n_z}")
        if self.penalty_weights is not None:
            raise ValueError("a full penalty precision replaces per-entry penalty weights; this table has both")
        self.set_penalty_center(center)
        self.penalty_precision = precision.to(self.codes.weight)

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
            Scalar loss tensor: regularization_weight * ||codes.weight - center||_2^2,
            each entry weighted by penalty_weights when those are set; the center
            is penalty_center, or zero. With penalty_precision set, the squared norm
            is (codes - center)^T P (codes - center) per shape instead.
        """
        offset = self.codes.weight if self.penalty_center is None else self.codes.weight - self.penalty_center
        if self.penalty_precision is not None:
            return self.regularization_weight * torch.einsum("si,ij,sj->", offset, self.penalty_precision, offset)
        squared = offset ** 2
        if self.penalty_weights is not None:
            squared = squared * self.penalty_weights
        l2_loss = torch.sum(squared)
        return self.regularization_weight * l2_loss
