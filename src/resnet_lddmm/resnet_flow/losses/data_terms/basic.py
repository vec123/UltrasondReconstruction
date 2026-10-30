"""Data terms without a nearest-neighbour search: pointwise L2 on corresponded clouds, and the no-op term."""

from typing import Optional
from torch import Tensor

from .base import DataTerm


class L2Data(DataTerm):
    """Mean squared pointwise error for corresponded point clouds.

    Assumes points are in correspondence by index (e.g., template points
    matched to target points at the same position). Simpler than Chamfer
    but requires pre-established correspondence.

    Supports both scalar loss and per-point losses (for adaptive subsampling in T28).
    """

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
        per_point: bool = False,
    ) -> Tensor:
        """Compute mean squared error between matched clouds.

        Args:
            pred: [B, N, 3] predicted positions
            target: [B, N, 3] target positions (same N as pred)
            pred_w, tgt_w, normals: ignored (kept for protocol compatibility)
            per_point: if True, return per-point losses [B, N]; if False, return scalar

        Returns:
            Scalar MSE loss if per_point=False, else per-point MSE [B, N]
        """
        per_point_losses = (pred - target).pow(2).sum(-1)  # [B, N]

        if per_point:
            return per_point_losses

        return per_point_losses.mean()


class NullData(DataTerm):
    """No-op data term: always zero, contributes no gradient.

    TwoSidedDataTerm.__init__ refuses source_to_target_weight=target_to_source_
    weight=0 specifically to catch that combination happening BY ACCIDENT (see
    its docstring: "training would silently optimize the regularizers alone").
    ``data_name: none`` is the explicit, intentional version of exactly that --
    for a run where a non-chamfer term (e.g. loss.landmark_weight) is meant to be
    the ONLY source of positional supervision, with kinetic/isometry doing the
    rest as regularizers.
    """

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
    ) -> Tensor:
        """Zero, connected to ``pred`` so autograd sees a valid (zero) gradient."""
        return pred.sum() * 0.0
