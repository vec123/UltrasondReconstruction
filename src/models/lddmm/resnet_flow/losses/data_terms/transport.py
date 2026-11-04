"""Optimal transport data terms: Sinkhorn, and its legacy EMD alias."""

from typing import Optional
import torch
from torch import Tensor

from .base import DataTerm


class SinkhornData(DataTerm):
    """Sinkhorn optimal transport distance via geomloss (T34).

    Solves the optimal transport problem using Sinkhorn's algorithm.
    Recommended for articulated shapes (e.g., hands) where Chamfer can
    produce incorrect finger-to-finger matchings.

    Point weights are normalised to probability measures before being handed to
    geomloss, which solves BALANCED transport and therefore needs both sides to
    carry equal mass; see _as_measure.

    Lazy-loads geomloss: auto-decoder configs work even without it installed.
    """

    def __init__(self, p: float = 2, blur: float = 0.01, backend: str = "auto", **kwargs):
        """Initialize Sinkhorn loss.

        Args:
            p: Distance norm (default 2 for L2)
            blur: Sinkhorn blur radius (higher = smoother gradient, default 0.01)
            backend: geomloss backend ("auto", "keops", "torch", default "auto")
            **kwargs: ignored (compatibility with other data terms)
        """
        super().__init__()
        self.p = p
        self.blur = blur
        self.backend = backend
        self.loss_fn = None  # Lazy initialized

    def _get_loss_fn(self):
        """Lazy-load geomloss on first call."""
        if self.loss_fn is None:
            try:
                import geomloss
            except ImportError:
                raise ImportError(
                    "geomloss required for Sinkhorn distance. "
                    "Install with: pip install geomloss"
                )
            self.loss_fn = geomloss.SamplesLoss(
                "sinkhorn", p=self.p, blur=self.blur, backend=self.backend
            )
        return self.loss_fn

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute Sinkhorn optimal transport distance.

        Args:
            pred: [B, N, 3] predicted point positions
            target: [B, M, 3] target point positions
            pred_w: [B, N] weights for pred (optional, passed to geomloss)
            tgt_w: [B, M] weights for target (optional, passed to geomloss)
            normals: ignored (kept for protocol compatibility)

        Returns:
            Scalar Sinkhorn loss
        """
        loss_fn = self._get_loss_fn()

        B, N, D = pred.shape
        B_tgt, M, D = target.shape
        assert B == B_tgt, f"Batch size mismatch: {B} vs {B_tgt}"

        loss_total = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
        for b in range(B):
            p_b = pred[b]  # [N, 3]
            t_b = target[b]  # [M, 3]

            loss_b = loss_fn(
                self._as_measure(None if pred_w is None else pred_w[b], N, p_b),
                p_b,
                self._as_measure(None if tgt_w is None else tgt_w[b], M, t_b),
                t_b,
            )
            loss_total = loss_total + loss_b

        return loss_total / B

    @staticmethod
    def _as_measure(weights, count, like):
        """Per-point weights as a probability measure: non-negative, summing to one.

        geomloss solves BALANCED optimal transport, so both arguments must carry the
        same total mass. Passing raw ``ones(N)`` gives mass N rather than 1, which
        rescales the loss by that mass -- and is outright ill-posed once N != M, as
        happens whenever the two clouds are subsampled to different sizes.
        Normalising also makes the value independent of the point count, so a run is
        comparable across subsample_M settings.

        Args:
            weights: [K] per-point weights, or None for uniform
            count: K, the number of points
            like: tensor supplying device and dtype

        Returns:
            [K] non-negative weights summing to 1
        """
        uniform = torch.full((count,), 1.0 / count, device=like.device, dtype=like.dtype)
        if weights is None:
            return uniform

        w = weights.to(dtype=like.dtype).clamp_min(0)
        total = w.sum()
        # Degenerate input (all zero, or negative before clamping) would divide by ~0
        # and hand geomloss a non-finite measure; fall back rather than propagate it.
        if not torch.isfinite(total) or total <= 0:
            return uniform
        return w / total


class EMDData(DataTerm):
    """Earth Mover's Distance via Sinkhorn's algorithm (legacy alias for SinkhornData).

    Deprecated: use SinkhornData instead. Kept for backward compatibility.
    """

    def __init__(self, p: float = 2, blur: float = 0.01, backend: str = "auto"):
        """Initialize EMD loss (delegates to SinkhornData)."""
        super().__init__()
        self._sinkhorn = SinkhornData(p=p, blur=blur, backend=backend)

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute Earth Mover's Distance via Sinkhorn."""
        return self._sinkhorn(pred, target, pred_w=pred_w, tgt_w=tgt_w, normals=normals)
