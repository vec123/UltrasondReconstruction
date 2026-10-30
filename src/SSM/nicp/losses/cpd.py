"""Coherent point drift loss."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
import torch
from src.SSM.nicp.neighbours import nearest

from .base import Loss, LossContext, LossValue, _check_ratio, _reduce, _subsample, register_loss


@register_loss("cpd")
@dataclass(frozen=True)
class CpdLoss(Loss):
    """Coherent-Point-Drift negative log-likelihood - genuinely match-free.

    The model vertices are the centroids of an isotropic Gaussian mixture and
    the target points are samples from it, so the objective is

        -sum_m log[ (1-w)/N sum_n N(y_m | x_n, sigma^2 I) + w/V ]

    No point is ever assigned to one partner: every centroid contributes to
    every target point, weighted by distance. ``w`` is the fraction of the
    target treated as uniform outliers over its bounding volume ``V``, which is
    what makes it tolerant of scan noise.

    Two practical notes. The mixture is truncated to the ``knn`` nearest
    centroids - the neglected terms are smaller by exp(-d^2/2 sigma^2) and the
    full 31k x 11k sum is not worth its cost. And ``sigma`` sets the capture
    radius: large values give a smooth, almost convex objective that ignores
    detail, small ones fit detail but only see nearby points. Annealing from
    large to small with ``sigma_end`` is the standard remedy.
    """

    sigma: Optional[float] = None        # None -> from the data on first call
    sigma_end: Optional[float] = None    # geometric anneal from sigma to this
    outlier_weight: float = 0.0
    knn: int = 16
    reduction: str = "mean"
    source_ratio: float = 0.0     # fraction of the model vertices used as centroids
    target_ratio: float = 0.0     # fraction of the target points treated as samples

    def __post_init__(self):
        if not 0.0 <= self.outlier_weight < 1.0:
            raise ValueError(f"outlier_weight must be in [0, 1), got {self.outlier_weight}")
        if self.knn < 1:
            raise ValueError(f"knn must be >= 1, got {self.knn}")
        _check_ratio(self.source_ratio, "source_ratio")
        _check_ratio(self.target_ratio, "target_ratio")
        for value, label in ((self.sigma, "sigma"), (self.sigma_end, "sigma_end")):
            if value is not None and value <= 0:
                raise ValueError(f"{label} must be positive or null, got {value}")

    def _sigma(self, pred: torch.Tensor, ctx: LossContext) -> float:
        start = self.sigma
        if start is None:
            if "cpd_sigma0" not in ctx.cache:
                # data-driven scale: RMS distance from each target point to the
                # closest centroid of the initial shape
                d, _ = nearest(ctx.target, pred.detach(), k=1)
                ctx.cache["cpd_sigma0"] = float(d.pow(2).mean().sqrt())
            start = ctx.cache["cpd_sigma0"]
        if self.sigma_end is None:
            return start
        return float(start * (self.sigma_end / start) ** ctx.progress)

    def __call__(self, pred: torch.Tensor, ctx: LossContext) -> LossValue:
        source, _ = _subsample(pred, self.source_ratio, ctx.generator)
        target, _ = _subsample(ctx.target, self.target_ratio, ctx.generator)

        sigma = self._sigma(source, ctx)
        sigma_sq = sigma ** 2
        n_centroids = source.shape[0]

        # nearest centroids per target point; distances recomputed live for grad
        _, idx = nearest(target, source.detach(), k=self.knn)
        d2 = (target.unsqueeze(1) - source[idx]).pow(2).sum(-1)            # (M, knn)

        log_kernel = (
            -d2 / (2.0 * sigma_sq)
            - 1.5 * math.log(2.0 * math.pi * sigma_sq)
            - math.log(n_centroids)
        )
        log_p = torch.logsumexp(log_kernel, dim=1)

        if self.outlier_weight > 0:
            extent = target.max(0).values - target.min(0).values
            volume = float(torch.clamp(extent, min=1e-12).prod())
            log_outlier = torch.full_like(log_p, math.log(self.outlier_weight / volume))
            log_p = torch.logaddexp(log_p + math.log1p(-self.outlier_weight), log_outlier)

        total = -_reduce(log_p, self.reduction)
        return LossValue(total, {"cpd_nll": float(total.detach()), "cpd_sigma": sigma})
