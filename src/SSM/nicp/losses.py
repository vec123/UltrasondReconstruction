"""Data terms: how far the model surface is from the target surface.

A loss is a callable ``(predicted vertices, context) -> LossValue``. The
context carries the target, the current correspondences *if any*, the
coefficients and where the fit has got to, so a loss can decide for itself how
the two surfaces are associated. That single ``Optional`` is what separates the
two families:

    matched_l2   consumes a fixed pairing produced by a correspondence method
    chamfer/cpd  ignore the pairing and associate the surfaces themselves,
                 fresh at every iteration and in both directions

The config selects one by name and passes its keyword arguments:

    loss:
      name: chamfer
      kwargs:
        source_to_target: {weight: 1.0, base_sigma: 1.0}
        target_to_source: {weight: 5.0, base_sigma: 1.0, field_name: consistency_score}

Nearest-neighbour indices inside the correspondence-free losses are taken
without gradient and the distance is recomputed from the live tensors, which is
the standard treatment: the index is piecewise constant in the coefficients, so
it carries no gradient of its own.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from src.SSM.nicp.correspondence import Correspondences
from src.SSM.nicp.neighbours import build_tree, nearest
from src.SSM.nicp.registry import Registry, RegistryError
from src.SSM.nicp.shape import ShapeSpace
from src.SSM.nicp.weight_fields import ChamferSide

# The robust penalties are pure functions of a squared-distance tensor with
# nothing LDDMM-specific in them, and they are already documented and exercised
# there. Importing beats keeping a second copy of the same four kernels in step.
from src.resnet_lddmm.resnet_flow.losses.data_terms import ROBUST_KINDS, _robust_penalty

LOSSES = Registry("loss")
register_loss = LOSSES.register


def available_losses() -> List[str]:
    return LOSSES.names()


def build_loss(name: str, kwargs: Optional[Dict[str, Any]] = None) -> "Loss":
    return LOSSES.build(name, kwargs)


# --------------------------------------------------------------------------
# context and result
# --------------------------------------------------------------------------

@dataclass
class LossContext:
    """Everything a data term may look at besides the predicted vertices."""

    target: torch.Tensor                                   # (M, 3)
    space: ShapeSpace
    alpha: torch.Tensor                                    # (K,)
    #: the target as a cloud, so a data term can reach its per-point fields
    #: (a consistency score, a support count) and not only its coordinates
    target_cloud: Optional[Any] = None
    correspondences: Optional[Correspondences] = None
    target_normals: Optional[torch.Tensor] = None          # (M, 3)
    iteration: int = 0
    n_iterations: int = 1
    generator: Optional[torch.Generator] = None
    cache: Dict[str, Any] = field(default_factory=dict)    # per-fit scratch (target tree, sigma, ...)

    @property
    def progress(self) -> float:
        """0.0 on the first iteration, 1.0 on the last - drives annealing."""
        if self.n_iterations <= 1:
            return 0.0
        return min(1.0, self.iteration / float(self.n_iterations - 1))

    @property
    def target_tree(self):
        """cKDTree over the target, built once per fit (the target never moves)."""
        if "target_tree" not in self.cache:
            self.cache["target_tree"] = build_tree(self.target)
        return self.cache["target_tree"]

    def require_correspondences(self, loss_name: str) -> Correspondences:
        if self.correspondences is None:
            raise RegistryError(
                f"loss '{loss_name}' needs correspondences, but the correspondence method "
                f"produced none. Either use a correspondence-free loss (e.g. chamfer, cpd) "
                f"or set a correspondence method other than 'none'."
            )
        if self.correspondences.is_empty:
            raise RegistryError(
                f"loss '{loss_name}' got an empty match set; loosen the correspondence "
                f"thresholds."
            )
        return self.correspondences


@dataclass
class LossValue:
    """The scalar being minimised, plus a per-term breakdown for reporting."""

    total: torch.Tensor
    terms: Dict[str, float] = field(default_factory=dict)

    @property
    def item(self) -> float:
        return float(self.total.detach())


# --------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------

class Loss(ABC):
    name = "?"
    #: True when the loss cannot work without a pairing
    requires_correspondences = False
    #: True when the loss reads the target's per-point normals, which are then
    #: resolved once per fit instead of being computed on the off chance
    needs_target_normals = False

    @abstractmethod
    def __call__(self, pred: torch.Tensor, ctx: LossContext) -> LossValue:
        """``pred`` is (N, 3) and differentiable in the coefficients."""

    def describe(self) -> str:
        return repr(self)


def _reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if values.numel() == 0:
        return values.sum()          # 0.0, and keeps the graph connected
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")


def _check_ratio(ratio: float, label: str) -> float:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"{label} must be in [0, 1] (0 = use everything), got {ratio}")
    return ratio


def _subsample(points: torch.Tensor, ratio: float, generator):
    """A random ``ratio`` of the points, redrawn on every call.

    ``ratio`` of 0 (or 1) means "use everything" and returns the tensor
    untouched, so nothing is copied and the caller's cached structures over the
    full cloud stay valid. Returns ``(points, index)`` with ``index`` None when
    no sampling happened.

    Redrawing each iteration is deliberate: a fixed subset would fit that subset,
    while fresh draws make the loss an unbiased estimate of the full one and the
    noise averages out over the optimisation.
    """
    total = points.shape[0]
    if ratio <= 0.0 or ratio >= 1.0:
        return points, None

    n = max(1, int(round(total * ratio)))
    if n >= total:
        return points, None
    idx = torch.randperm(total, generator=generator, device=points.device)[:n]
    return points[idx], idx


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
    :class:`~src.SSM.nicp.weight_fields.ChamferSide`, whose keys match
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
            d2 = _robust_penalty(d2, self.robust_kernel, self.robust_breakpoint,
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


@register_loss("point_to_plane")
@dataclass(frozen=True)
class PointToPlaneLoss(Loss):
    """Matched distance measured along the target normal, not straight-line.

    Only the component of the residual perpendicular to the target surface is
    penalised, so the model is free to slide tangentially instead of being
    pulled at whichever point it happened to be paired with. On smooth regions
    that converges markedly faster than point-to-point.

    Like ``matched_l2`` this consumes a pairing rather than making one, so the
    rejection rules are the correspondence method's. Needs target normals.
    """

    weight: float = 1.0
    reduction: str = "mean"

    requires_correspondences = True
    needs_target_normals = True

    def __call__(self, pred: torch.Tensor, ctx: LossContext) -> LossValue:
        if ctx.target_normals is None:
            raise RegistryError("loss 'point_to_plane' needs target normals, none were supplied")

        corr = ctx.require_correspondences(self.name)
        device, dtype = pred.device, pred.dtype

        src = torch.as_tensor(corr.source_idx, dtype=torch.long, device=device)
        tgt = torch.as_tensor(corr.target_idx, dtype=torch.long, device=device)
        pair_weight = torch.as_tensor(corr.weight, dtype=dtype, device=device)

        delta = pred[src] - ctx.target[tgt]
        signed = (delta * ctx.target_normals[tgt]).sum(-1)
        weighted = pair_weight * signed.pow(2)

        value = weighted.sum() / pair_weight.sum() if self.reduction == "mean" else weighted.sum()
        total = self.weight * value
        return LossValue(total, {"point_to_plane": float(total.detach())})


@register_loss("p2plane_chamfer")
class P2PlaneChamferLoss(ChamferLoss):
    """``chamfer``'s per-direction weighting, with the residual measured to the
    target's tangent plane rather than to its nearest sample.

    Point-to-point Chamfer floors at roughly half the target's local sample
    spacing - a model vertex can never score better than the nearest scan
    *vertex* - and it pins tangentially, so vertices stick to samples instead of
    sliding along the surface. Projecting the residual onto the target normal
    removes both: only off-surface displacement is penalised, tangential motion
    is free, and the achievable residual is set by the surface rather than by
    how densely it happens to be sampled. The standard ICP point-to-plane
    argument, applied to a term that makes its own pairs.

    BOTH directions project onto the TARGET's normal, never the model's::

        source_to_target    ((x_i - t*) . n_t*)^2     t* = nearest target to x_i
        target_to_source    ((x* - t_j) . n_tj)^2     x* = nearest model vtx to t_j

    so the only normals needed are the target's, which are static: resolved once
    per fit rather than rebuilt from the moving model every iteration.
    Orientation is irrelevant - the projection is squared, so n and -n score
    alike and a target with inconsistent winding needs no repair.

    ``plane_weight`` blends against plain point-to-point. Pure point-to-plane
    (1.0) leaves tangential motion completely unconstrained, which is degenerate
    wherever the target is locally flat: the model can slide within the plane at
    no cost and nothing pulls it back. The default keeps a point-to-point anchor
    for exactly that reason.
    """

    needs_target_normals = True

    def __init__(self, source_to_target=None, target_to_source=None,
                 plane_weight: float = 0.8, p: int = 2,
                 source_ratio: float = 0.0, target_ratio: float = 0.0,
                 robust_kernel: Optional[str] = None,
                 robust_breakpoint: Optional[float] = None,
                 robust_slope: Optional[float] = None,
                 robust_tau: Optional[float] = None):
        super().__init__(source_to_target=source_to_target, target_to_source=target_to_source,
                         p=p, source_ratio=source_ratio, target_ratio=target_ratio,
                         robust_kernel=robust_kernel, robust_breakpoint=robust_breakpoint,
                         robust_slope=robust_slope, robust_tau=robust_tau)
        if not 0.0 <= plane_weight <= 1.0:
            raise ValueError(
                f"plane_weight must be in [0, 1], got {plane_weight}: it blends "
                f"point-to-point (0) against point-to-plane (1), it is not a scale"
            )
        self.plane_weight = plane_weight

    def __call__(self, pred: torch.Tensor, ctx: LossContext) -> LossValue:
        if ctx.target_normals is None:
            raise RegistryError(
                "loss 'p2plane_chamfer' needs the target's per-point normals, but none "
                "reached it. Failing rather than falling back to point-to-point on "
                "purpose: a silent fallback is indistinguishable from the term working."
            )

        source, source_idx = _subsample(pred, self.source_ratio, ctx.generator)
        target, target_idx = _subsample(ctx.target, self.target_ratio, ctx.generator)

        normals = ctx.target_normals
        if target_idx is not None:
            normals = normals[target_idx]      # gathered on the SAME draw as target
        # Unit-length defensively: a normal array that was scaled, interpolated
        # or averaged is no longer unit, and a non-unit n silently rescales the
        # projection per point - a weighting nobody configured.
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        total = torch.zeros((), dtype=pred.dtype, device=pred.device)
        terms: Dict[str, float] = {}
        st, ts = self.source_to_target, self.target_to_source
        a = self.plane_weight

        if st.active:
            tree = ctx.target_tree if target_idx is None else None
            _, idx = nearest(source.detach(), target, k=1, tree=tree)
            nn = idx[:, 0]
            offset = source - target[nn]
            d_point = offset.pow(2).sum(-1)
            d_plane = (offset * normals[nn]).sum(-1).pow(2)
            weight = self._field(st, "source_to_target", ctx.space.n_vertices, ctx, None)
            if weight is not None and source_idx is not None:
                weight = weight[source_idx]
            value = self._weighted_mean(self._penalty((1.0 - a) * d_point + a * d_plane), weight)
            total = total + st.precision * value
            terms["p2plane_source_to_target"] = float(value.detach())

        if ts.active:
            _, idx = nearest(target, source.detach(), k=1)
            offset = source[idx[:, 0]] - target
            d_point = offset.pow(2).sum(-1)
            d_plane = (offset * normals).sum(-1).pow(2)     # the target's own normal
            weight = self._field(ts, "target_to_source", ctx.target.shape[0],
                                 ctx, ctx.target_cloud)
            if weight is not None and target_idx is not None:
                weight = weight[target_idx]
            value = self._weighted_mean(self._penalty((1.0 - a) * d_point + a * d_plane), weight)
            total = total + ts.precision * value
            terms["p2plane_target_to_source"] = float(value.detach())

        terms["p2plane_chamfer"] = float(total.detach())
        terms["chamfer_n_source"] = float(source.shape[0])
        terms["chamfer_n_target"] = float(target.shape[0])
        return LossValue(total, terms)

    def describe(self) -> str:
        return super().describe().replace(
            "ChamferLoss(", f"P2PlaneChamferLoss(plane_weight={self.plane_weight:g}, ")


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
