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
      kwargs: {forward_weight: 1.0, backward_weight: 5.0, max_dist: 0.0026}

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

from src.nicp.correspondence import Correspondences
from src.nicp.neighbours import build_tree, nearest
from src.nicp.registry import Registry, RegistryError
from src.nicp.shape import ShapeSpace

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
@dataclass(frozen=True)
class ChamferLoss(Loss):
    """Bidirectional nearest-neighbour distance, each direction weighted.

    forward  (model -> target): every model vertex is pulled onto the target.
    backward (target -> model): every target point is explained by the model.

    **Unfiltered.** Which pairs are admitted is a property of the pairing, not
    of the objective, so ``max_dist`` and ``trim_ratio`` live on the
    correspondence methods instead - see ``symmetric``, which is this same
    bidirectional matching plus the rejection rules, and which additionally
    works with the closed-form solver. Reach for this loss when the two surfaces
    genuinely overlap, or as one term inside a ``composite``.

    ``p=2`` sums squared distances (the usual definition), ``p=1`` sums
    distances, which is less sensitive to a few far-away points.

    ``source_ratio`` and ``target_ratio`` thin the two clouds independently
    before every evaluation, which is where the cost is: the search is linear in
    both, so halving each quarters it. 0 uses everything.
    """

    forward_weight: float = 1.0
    backward_weight: float = 1.0
    p: int = 2
    reduction: str = "mean"
    source_ratio: float = 0.0     # fraction of the model vertices, 0 = all
    target_ratio: float = 0.0     # fraction of the target points, 0 = all

    def __post_init__(self):
        if self.p not in (1, 2):
            raise ValueError(f"p must be 1 or 2, got {self.p}")
        _check_ratio(self.source_ratio, "source_ratio")
        _check_ratio(self.target_ratio, "target_ratio")
        if self.forward_weight < 0 or self.backward_weight < 0:
            raise ValueError("direction weights must be >= 0")
        if self.forward_weight == 0 and self.backward_weight == 0:
            raise ValueError("both direction weights are 0; nothing would be optimised")
        _reduce(torch.zeros(1), self.reduction)

    def _distances(self, d2: torch.Tensor) -> torch.Tensor:
        return d2 if self.p == 2 else (d2 + 1e-24).sqrt()

    def __call__(self, pred: torch.Tensor, ctx: LossContext) -> LossValue:
        source, _ = _subsample(pred, self.source_ratio, ctx.generator)
        target, target_idx = _subsample(ctx.target, self.target_ratio, ctx.generator)

        total = torch.zeros((), dtype=pred.dtype, device=pred.device)
        terms: Dict[str, float] = {}

        if self.forward_weight > 0:
            # The target never moves, so its tree is built once per fit - but
            # only the full cloud has one; a fresh subsample needs its own.
            tree = ctx.target_tree if target_idx is None else None
            _, idx = nearest(source.detach(), target, k=1, tree=tree)
            d2 = (source - target[idx[:, 0]]).pow(2).sum(-1)
            forward = _reduce(self._distances(d2), self.reduction)
            total = total + self.forward_weight * forward
            terms["chamfer_forward"] = float(forward.detach())

        if self.backward_weight > 0:
            # the model moves every iteration, so this side is rebuilt each call
            _, idx = nearest(target, source.detach(), k=1)
            d2 = (target - source[idx[:, 0]]).pow(2).sum(-1)
            backward = _reduce(self._distances(d2), self.reduction)
            total = total + self.backward_weight * backward
            terms["chamfer_backward"] = float(backward.detach())

        terms["chamfer"] = float(total.detach())
        terms["chamfer_n_source"] = float(source.shape[0])
        terms["chamfer_n_target"] = float(target.shape[0])
        return LossValue(total, terms)


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
