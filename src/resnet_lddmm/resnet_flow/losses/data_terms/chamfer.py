"""Chamfer-family data terms: plain, area-weighted, unmasked and normal-aware."""

from typing import Optional
import torch
from torch import Tensor

from .two_sided import TwoSidedDataTerm


class CDData(TwoSidedDataTerm):
    """Chamfer distance on unpadded point clouds.

    Wraps the reused chamfer_loss with an all-ones mask, since our inputs
    have no padding. Differentiable, and symmetric at the default weights --
    see TwoSidedDataTerm for the per-direction weights that break that symmetry.

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
        """Compute Chamfer distance with all-ones mask.

        Args:
            pred: [B, N, 3] predicted positions
            target: [B, M, 3] target positions
            pred_w, tgt_w, normals: ignored (kept for protocol compatibility)
            per_point: if True, return per-point losses [B, M]; if False, return scalar

        Returns:
            Scalar Chamfer loss if per_point=False, else per-point distances [B, M]
        """
        # Create all-ones mask for unpadded target cloud
        mask = torch.ones(
            target.shape[:2], dtype=torch.bool, device=target.device
        )

        # Compute pairwise distances [B, N, M]
        dist_sq = torch.cdist(pred, target, p=2).pow(2)
        dist_sq = dist_sq.masked_fill(~mask.unsqueeze(1), 1e6)

        if per_point:
            # Return per-point losses from target to pred (term2 of Chamfer).
            # Deliberately UNWEIGHTED by the direction weights: this feeds adaptive
            # subsampling, which needs raw distances to RANK target points by. Under
            # target_to_source_weight=0 the weighted version would be all zeros and
            # rank nothing, which is not what a sampler is asking for.
            per_point_losses = dist_sq.min(dim=1)[0]  # [B, M]
            return per_point_losses

        # Standard scalar loss
        term1 = self.loss_distance_handle(dist_sq.min(dim=2)[0]).mean() if self.scores_source_to_target else None

        term2 = None
        if self.scores_target_to_source:
            dist_t_to_p = self.loss_distance_handle(dist_sq.min(dim=1)[0])
            term2 = (dist_t_to_p * mask).sum() / (mask.sum() + 1e-8)

        return self._combine(term1, term2)


class WeightedCDData(TwoSidedDataTerm):
    """Weighted Chamfer distance using per-point target areas (T33).

    Similar to CDData but weights the target→pred term by target point weights
    (typically surface areas). Reduces to standard Chamfer with uniform weights.

    Two independent weightings meet here, and they are not the same thing:
    ``tgt_w`` is PER TARGET POINT (which observations count for more), while the
    inherited direction weights are PER DIRECTION (which half of Chamfer is
    scored at all). See TwoSidedDataTerm.
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
        """Compute weighted Chamfer distance.

        Args:
            pred: [B, N, 3] predicted positions
            target: [B, M, 3] target positions
            pred_w: [B, N] weights for pred (ignored)
            tgt_w: [B, M] per-point target weights (surface areas); uniform if None
            normals: ignored (kept for protocol compatibility)
            per_point: if True, return per-point losses [B, M]; if False, return scalar

        Returns:
            Scalar weighted Chamfer loss if per_point=False, else per-point losses [B, M]
        """
        B, N, D = pred.shape
        B_tgt, M, D = target.shape

        # Compute pairwise distances [B, N, M]
        dist_sq = torch.cdist(pred, target, p=2).pow(2)

        if per_point:
            # Return per-point losses from target to pred, weighted by tgt_w
            per_point_losses = dist_sq.min(dim=1)[0]  # [B, M]
            if tgt_w is not None:
                per_point_losses = per_point_losses * tgt_w
            return per_point_losses

        # Term 1: pred → target (unweighted, average nearest target distance)
        term1 = self.loss_distance_handle(dist_sq.min(dim=2)[0]).mean() if self.scores_source_to_target else None

        # Term 2: target → pred (weighted by tgt_w if provided)
        term2 = None
        if self.scores_target_to_source:
            tgt_to_pred = self.loss_distance_handle(dist_sq.min(dim=1)[0])  # [B, M]
            if tgt_w is not None:
                # Normalize weights to avoid changing overall scale
                w_normalized = tgt_w / (tgt_w.mean(dim=1, keepdim=True) + 1e-8)
                term2 = (tgt_to_pred * w_normalized).mean()
            else:
                term2 = tgt_to_pred.mean()

        return self._combine(term1, term2)


class PCDData(TwoSidedDataTerm):
    """Point Cloud Distance: symmetric Chamfer-like metric without explicit masking (T33).

    Similar to CDData but computed more directly without mask operations.
    Equivalent to CDData on unpadded point clouds. Per-direction weights are
    inherited from TwoSidedDataTerm; "symmetric" describes the defaults.
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
        """Compute point cloud distance (symmetric).

        Args:
            pred: [B, N, 3] predicted point positions
            target: [B, M, 3] target point positions
            pred_w, tgt_w, normals: ignored (kept for protocol compatibility)
            per_point: if True, return per-point losses [B, M]; if False, return scalar

        Returns:
            Scalar PCD loss if per_point=False, else per-point distances [B, M]
        """
        # Compute pairwise distances [B, N, M]
        dist_sq = torch.cdist(pred, target, p=2).pow(2)

        if per_point:
            # Return per-point losses from target to pred
            per_point_losses = dist_sq.min(dim=1)[0]  # [B, M]
            return per_point_losses

        # Symmetric at the default weights: pred→target + target→pred
        term1 = self.loss_distance_handle(dist_sq.min(dim=2)[0]).mean() if self.scores_source_to_target else None
        term2 = self.loss_distance_handle(dist_sq.min(dim=1)[0]).mean() if self.scores_target_to_source else None
        return self._combine(term1, term2)


class NCDData(TwoSidedDataTerm):
    """Normal-aware Chamfer distance: penalizes surfaces with wrong orientation (T33).

    Combines point-to-point distance with normal alignment penalty.
    Encourages normals to be oriented away from the matching pred points.

    The normal penalty lives INSIDE the source->target direction -- it scores
    where each SOURCE point sits relative to its nearest target's surface -- so
    ``source_to_target_weight=0`` drops the orientation penalty along with the
    distance. A one-sided term that still reads normals is a point-to-plane
    target->source term, which is a different loss, not a setting of this one.
    """

    def __init__(self, normal_weight: float = 1.0,
                 source_to_target_weight: float = 1.0,
                 target_to_source_weight: float = 1.0,
                 loss_distance_handle: Optional[dict] = None):
        """Initialize NCD loss.

        Args:
            normal_weight: balance between geometric and normal penalties (default 1.0)
            source_to_target_weight: weight on D(phi(source) -> target); 0.0 disables it
            target_to_source_weight: weight on D(target -> phi(source)); 0.0 disables it
            loss_distance_handle: see TwoSidedDataTerm. Applied to the
                GEOMETRIC distance only -- the normal-alignment penalty is a
                separate, unrelated quantity and never goes through it.
        """
        super().__init__(source_to_target_weight=source_to_target_weight,
                         target_to_source_weight=target_to_source_weight,
                         loss_distance_handle=loss_distance_handle)
        self.normal_weight = normal_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
        per_point: bool = False,
    ) -> Tensor:
        """Compute normal-aware Chamfer distance.

        Args:
            pred: [B, N, 3] predicted point positions
            target: [B, M, 3] target point positions
            pred_w, tgt_w: ignored
            normals: [B, M, 3] target surface normals; if None, reduces to CDData
            per_point: if True, return per-point losses [B, M]; if False, return scalar

        Returns:
            Scalar NCD loss if per_point=False, else per-point losses [B, M]
        """
        B, N, D = pred.shape
        B_tgt, M, D = target.shape

        # Compute pairwise distances [B, N, M]
        dist_vec = pred.unsqueeze(2) - target.unsqueeze(1)  # [B, N, M, 3]
        dist_sq = (dist_vec ** 2).sum(dim=3)  # [B, N, M]

        # If no normals, fall back to standard Chamfer
        if normals is None:
            if per_point:
                return dist_sq.min(dim=1)[0]
            term1 = self.loss_distance_handle(dist_sq.min(dim=2)[0]).mean() if self.scores_source_to_target else None
            term2 = self.loss_distance_handle(dist_sq.min(dim=1)[0]).mean() if self.scores_target_to_source else None
            return self._combine(term1, term2)

        if per_point:
            # Per-point losses from target to pred (geometric only). Returned BEFORE
            # the normal penalty, which this path never used and which costs two
            # [B, N, 3] gathers to build.
            return dist_sq.min(dim=1)[0]  # [B, M]

        # Term 1: pred → target (point distance + normal penalty)
        term1 = None
        if self.scores_source_to_target:
            # Compute normal alignment penalty: penalizes normals pointing towards pred
            # For each pred point, find nearest target and its normal
            pred_to_tgt_indices = dist_sq.min(dim=2)[1]  # [B, N]
            pred_to_tgt_dists = dist_sq.min(dim=2)[0]  # [B, N]

            # Vectorized normal penalty: gather directions and normals, compute dot products in parallel
            # direction: [B, N, 3] - vector from target to pred
            b_idx = torch.arange(B, device=pred.device).unsqueeze(1)  # [B, 1]
            n_idx = torch.arange(N, device=pred.device).unsqueeze(0)  # [1, N]
            direction = dist_vec[b_idx, n_idx, pred_to_tgt_indices]  # [B, N, 3]

            # normal: [B, N, 3] - normals at nearest target points
            normal = normals[b_idx, pred_to_tgt_indices]  # [B, N, 3]

            # Normalize both
            dir_norm = torch.norm(direction, dim=-1, keepdim=True) + 1e-8  # [B, N, 1]
            normal_norm = torch.norm(normal, dim=-1, keepdim=True) + 1e-8  # [B, N, 1]
            direction_normalized = direction / dir_norm  # [B, N, 3]
            normal_normalized = normal / normal_norm  # [B, N, 3]

            # Compute dot products and penalty
            dot_products = (direction_normalized * normal_normalized).sum(dim=-1)  # [B, N]
            normal_penalty = torch.clamp(dot_products, min=0.0).mean()  # scalar

            # Geometric distance goes through the loss_distance_handle; the
            # normal-alignment penalty is a different quantity (a bounded [0,1]
            # dot-product term already) and never does.
            term1 = self.loss_distance_handle(pred_to_tgt_dists).mean() + self.normal_weight * normal_penalty

        # Term 2: target → pred (point distance only)
        term2 = self.loss_distance_handle(dist_sq.min(dim=1)[0]).mean() if self.scores_target_to_source else None

        return self._combine(term1, term2)
