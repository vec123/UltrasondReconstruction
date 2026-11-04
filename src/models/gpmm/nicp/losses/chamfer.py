"""Correspondence-free Chamfer loss with a weight and a base sigma per direction."""

from __future__ import annotations

from typing import Dict, Optional
import torch
from src.models.gpmm.nicp.matching.neighbours import nearest
from src.models.gpmm.nicp.weight_fields import ChamferSide
# The robust penalties are pure functions of a squared-distance tensor with
# nothing LDDMM-specific in them, and they are already documented and exercised
# there. Importing beats keeping a second copy of the same four kernels in step.
from src.shared.losses.robust import ROBUST_KINDS, robust_penalty

from .base import Loss, LossContext, LossValue, _check_ratio, _subsample, register_loss


# --------------------------------------------------------------------------
# correspondence-free
# --------------------------------------------------------------------------

@register_loss("chamfer")
class ChamferLoss(Loss):
    """Bidirectional nearest-neighbour distance, each direction weighted its own way.

        source_to_target   every model vertex reaches for the scan
        target_to_source   every scan point must be explained by the model

    The two are not interchangeable on a partial scan. Model vertices outside
    the scanned region have no counterpart, so ``source_to_target`` invents
    attraction for them; ``target_to_source`` has no such problem and is usually
    the one to trust. Each therefore carries its own ``weight``, its own
    ``base_sigma`` and, optionally, its own per-point field - see
    :class:`~src.models.gpmm.nicp.weight_fields.ChamferSide`, whose keys match
    ``ChamferSideCfg`` in the resnet_lddmm subsystem so the two configs read
    alike::

        loss:
          name: chamfer
          kwargs:
            source_to_target: {weight: 1.0, base_sigma: 1.0}
            target_to_source: {weight: 1.0, base_sigma: 0.5,
                               field_name: consistency_score}
            robust_kernel: linear
            robust_breakpoint: 5.0

    A direction contributes ``weight / (2*base_sigma**2)`` times the mean of its
    per-point penalties. The field, when given, is normalised to mean 1 before
    it multiplies those penalties, so switching a field on changes *where* the
    loss concentrates rather than how big it is - a config that adds one does
    not also need every weight retuned.

    **Unfiltered.** Which pairs are admitted is a property of the pairing, not
    of the objective, so ``max_dist`` and ``trim_ratio`` live on the
    correspondence methods - see ``symmetric``, the same bidirectional matching
    with the rejection rules attached, which also works with the closed-form
    solver. ``robust_kernel`` is this loss's own answer to the same problem:
    it leaves the penalty quadratic out to ``robust_breakpoint`` and tapers
    beyond it, so a far point loses influence without being discarded.

    ``p=2`` uses squared distances (the usual definition), ``p=1`` distances,
    which is less sensitive to a few far-away points. ``source_ratio`` and
    ``target_ratio`` thin the clouds before every evaluation, where the cost is:
    the search is linear in both, so halving each quarters it. 0 uses everything.
    """

    def __init__(self, source_to_target=None, target_to_source=None,
                 p: int = 2, source_ratio: float = 0.0, target_ratio: float = 0.0,
                 robust_kernel: Optional[str] = None,
                 robust_breakpoint: Optional[float] = None,
                 robust_slope: Optional[float] = None,
                 robust_tau: Optional[float] = None):
        self.source_to_target = ChamferSide.from_dict(source_to_target, "source_to_target")
        self.target_to_source = ChamferSide.from_dict(target_to_source, "target_to_source")

        if p not in (1, 2):
            raise ValueError(f"p must be 1 or 2, got {p}")
        self.p = p
        self.source_ratio = _check_ratio(source_ratio, "source_ratio")
        self.target_ratio = _check_ratio(target_ratio, "target_ratio")

        if not (self.source_to_target.active or self.target_to_source.active):
            raise ValueError(
                "both direction weights are 0; nothing would be optimised. Set "
                "source_to_target.weight and/or target_to_source.weight above 0."
            )

        if robust_kernel is not None and robust_kernel not in ROBUST_KINDS:
            raise ValueError(
                f"unknown robust_kernel {robust_kernel!r}; expected one of {list(ROBUST_KINDS)}"
            )
        if robust_kernel not in (None, "quadratic") and robust_breakpoint is None:
            raise ValueError(
                f"robust_kernel={robust_kernel!r} needs a robust_breakpoint - the distance "
                f"at which the penalty stops being quadratic, in the units the fit runs in"
            )
        self.robust_kernel = robust_kernel
        self.robust_breakpoint = robust_breakpoint
        self.robust_slope = robust_slope
        self.robust_tau = robust_tau

    def _penalty(self, d2: torch.Tensor) -> torch.Tensor:
        """Squared distance, robustified if asked, then dropped to p=1 if asked."""
        if self.robust_kernel not in (None, "quadratic"):
            d2 = robust_penalty(d2, self.robust_kernel, self.robust_breakpoint,
                                 slope=self.robust_slope, tau=self.robust_tau)
        return d2 if self.p == 2 else (d2 + 1e-24).sqrt()

    def _field(self, side: ChamferSide, label: str, n_expected: int,
               ctx: LossContext, cloud) -> Optional[torch.Tensor]:
        """The direction's per-point weight, resolved once per fit and cached."""
        if side.field_name is None:
            return None
        key = f"chamfer_field_{label}"
        if key not in ctx.cache:
            values = side.resolve_field(n_expected, cloud=cloud, label=label)
            ctx.cache[key] = torch.as_tensor(
                values, dtype=ctx.target.dtype, device=ctx.target.device
            )
        return ctx.cache[key]

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
        """Mean of ``values``, under a per-point weight normalised to mean 1.

        Normalising is what keeps a field from doubling as a global gain: it
        redistributes emphasis across the points without moving the direction's
        overall magnitude, which stays the business of weight and base_sigma.
        """
        if values.numel() == 0:
            return values.sum()          # 0.0, and keeps the graph connected
        if weight is None:
            return values.mean()
        w = weight / (weight.mean() + 1e-12)
        return (values * w).mean()

    def __call__(self, pred: torch.Tensor, ctx: LossContext) -> LossValue:
        source, source_idx = _subsample(pred, self.source_ratio, ctx.generator)
        target, target_idx = _subsample(ctx.target, self.target_ratio, ctx.generator)

        total = torch.zeros((), dtype=pred.dtype, device=pred.device)
        terms: Dict[str, float] = {}

        st, ts = self.source_to_target, self.target_to_source

        if st.active:
            # The target never moves, so its tree is built once per fit - but
            # only the full cloud has one; a fresh subsample needs its own.
            tree = ctx.target_tree if target_idx is None else None
            _, idx = nearest(source.detach(), target, k=1, tree=tree)
            d2 = (source - target[idx[:, 0]]).pow(2).sum(-1)
            # indexed by the SOURCE, so the model's own vertices - it has no
            # point data, hence ChamferSide.field_source for this direction
            weight = self._field(st, "source_to_target", ctx.space.n_vertices, ctx, None)
            if weight is not None and source_idx is not None:
                weight = weight[source_idx]
            value = self._weighted_mean(self._penalty(d2), weight)
            total = total + st.precision * value
            terms["chamfer_source_to_target"] = float(value.detach())

        if ts.active:
            # the model moves every iteration, so this side is rebuilt each call
            _, idx = nearest(target, source.detach(), k=1)
            d2 = (target - source[idx[:, 0]]).pow(2).sum(-1)
            weight = self._field(ts, "target_to_source", ctx.target.shape[0],
                                 ctx, ctx.target_cloud)
            if weight is not None and target_idx is not None:
                weight = weight[target_idx]
            value = self._weighted_mean(self._penalty(d2), weight)
            total = total + ts.precision * value
            terms["chamfer_target_to_source"] = float(value.detach())

        terms["chamfer"] = float(total.detach())
        terms["chamfer_n_source"] = float(source.shape[0])
        terms["chamfer_n_target"] = float(target.shape[0])
        return LossValue(total, terms)

    def describe(self) -> str:
        robust = "" if self.robust_kernel in (None, "quadratic") else \
            f", robust={self.robust_kernel}(breakpoint={self.robust_breakpoint:g})"
        return (f"ChamferLoss(p={self.p}"
                f", source_to_target[{self.source_to_target.describe()}]"
                f", target_to_source[{self.target_to_source.describe()}]{robust})")
