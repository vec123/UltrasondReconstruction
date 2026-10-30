"""The loss registry, the context and value types, the ``Loss`` base class and the helpers the losses share."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from src.SSM.nicp.matching.correspondence import Correspondences
from src.SSM.nicp.matching.neighbours import build_tree
from src.core.registry import RegistryError, TypedRegistry
from src.SSM.nicp.shape import ShapeSpace


LOSSES = TypedRegistry("loss")


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
