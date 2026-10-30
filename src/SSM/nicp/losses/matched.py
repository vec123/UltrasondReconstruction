"""Squared error against a fixed pairing produced by a correspondence method."""

from __future__ import annotations

from dataclasses import dataclass
import torch

from .base import Loss, LossContext, LossValue, _reduce, register_loss


# --------------------------------------------------------------------------
# correspondence-based
# --------------------------------------------------------------------------

@register_loss("matched_l2")
@dataclass(frozen=True)
class MatchedL2(Loss):
    """Squared distance over a fixed set of matched pairs - the classic NICP term.

    This is the objective the closed-form solver minimises, so it is also the
    one to use when comparing a gradient fit against the original pipeline.
    """

    reduction: str = "mean"

    requires_correspondences = True

    def __post_init__(self):
        _reduce(torch.zeros(1), self.reduction)   # validate eagerly

    def __call__(self, pred: torch.Tensor, ctx: LossContext) -> LossValue:
        corr = ctx.require_correspondences(self.name)
        device, dtype = pred.device, pred.dtype

        src = torch.as_tensor(corr.source_idx, dtype=torch.long, device=device)
        tgt = torch.as_tensor(corr.target_idx, dtype=torch.long, device=device)
        weight = torch.as_tensor(corr.weight, dtype=dtype, device=device)

        d2 = (pred[src] - ctx.target[tgt]).pow(2).sum(-1)
        weighted = weight * d2
        total = weighted.sum() / weight.sum() if self.reduction == "mean" else weighted.sum()

        return LossValue(total, {"matched_l2": float(total.detach()),
                                 "rms": float(d2.detach().mean().sqrt())})
