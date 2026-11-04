"""Weighted sum of other losses, each built by name."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
import torch

from .base import Loss, LossContext, LossValue, build_loss, register_loss


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------

@register_loss("composite")
class CompositeLoss(Loss):
    """Weighted sum of other losses.

        loss:
          name: composite
          kwargs:
            terms:
              - {name: chamfer, weight: 1.0, kwargs: {backward_weight: 5.0}}
              - {name: point_to_plane, weight: 0.5}
    """

    def __init__(self, terms: Sequence[Dict[str, Any]]):
        if not terms:
            raise ValueError("composite loss needs at least one term")
        self.weights: List[float] = []
        self.losses: List[Loss] = []
        for entry in terms:
            entry = dict(entry)
            weight = float(entry.pop("weight", 1.0))
            name = entry.pop("name", None)
            if name is None:
                raise ValueError(f"composite term is missing 'name': {entry}")
            unknown = sorted(set(entry) - {"kwargs"})
            if unknown:
                raise ValueError(f"composite term '{name}' has unexpected key(s) {unknown}")
            self.weights.append(weight)
            self.losses.append(build_loss(name, entry.get("kwargs")))

    @property
    def requires_correspondences(self) -> bool:
        return any(loss.requires_correspondences for loss in self.losses)

    def __call__(self, pred: torch.Tensor, ctx: LossContext) -> LossValue:
        total = torch.zeros((), dtype=pred.dtype, device=pred.device)
        terms: Dict[str, float] = {}
        for weight, loss in zip(self.weights, self.losses):
            value = loss(pred, ctx)
            total = total + weight * value.total
            terms.update(value.terms)
        terms["total"] = float(total.detach())
        return LossValue(total, terms)

    def __repr__(self) -> str:
        parts = ", ".join(f"{w:g}*{l.name}" for w, l in zip(self.weights, self.losses))
        return f"CompositeLoss({parts})"
