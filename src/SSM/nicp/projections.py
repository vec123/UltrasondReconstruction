"""Hard constraints applied to the coefficients after each step.

A prior pulls, a projection stops: whatever the solver proposes is mapped back
onto the admissible set before the next iteration. That matters most in
correspondence-free fitting, where nothing rejects a bad association and the
fit can wander a long way out of the model before the data term notices.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from scipy.stats import chi2

from src.SSM.nicp.registry import Registry
from src.SSM.nicp.shape import ShapeSpace

PROJECTIONS = Registry("projection")
register_projection = PROJECTIONS.register


def available_projections() -> List[str]:
    return PROJECTIONS.names()


def build_projection(name: str, kwargs: Optional[Dict[str, Any]] = None) -> "Projection":
    return PROJECTIONS.build(name, kwargs)


class Projection(ABC):
    name = "?"

    @abstractmethod
    def __call__(self, alpha: torch.Tensor, space: ShapeSpace) -> torch.Tensor:
        """Returns the admissible coefficients closest to ``alpha``."""

    def describe(self) -> str:
        return repr(self)


def _rescale_to_ellipsoid(alpha: torch.Tensor, space: ShapeSpace, beta_sq: float) -> torch.Tensor:
    """Shrinks alpha onto the boundary of sum((alpha_i/sd_i)^2) <= beta_sq."""
    dist_sq = space.mahalanobis_sq(alpha)
    if float(dist_sq) <= beta_sq:
        return alpha
    return alpha * torch.sqrt(torch.as_tensor(beta_sq, dtype=alpha.dtype, device=alpha.device) / dist_sq)


@register_projection("none")
@dataclass(frozen=True)
class NoProjection(Projection):
    def __call__(self, alpha, space):
        return alpha


@register_projection("chi_clamp")
@dataclass(frozen=True)
class ChiClamp(Projection):
    """Reproduces ShapeInstance.clamp_coefficients.

    That method takes a probability argument and then ignores it: the bound is
    hardcoded to beta = 10000, with the chi-square line commented out. The
    constant is kept here as an explicit parameter so old runs can be
    reproduced, but it has no distributional meaning - prefer
    ``mahalanobis_clamp``, which derives the bound from the model dimension.
    """

    beta: float = 10000.0

    def __post_init__(self):
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")

    def __call__(self, alpha, space):
        return _rescale_to_ellipsoid(alpha, space, self.beta ** 2)


@register_projection("mahalanobis_clamp")
@dataclass(frozen=True)
class MahalanobisClamp(Projection):
    """Confines alpha to the chi-square ellipsoid that holds ``p`` of the model.

    sum((alpha_i/sd_i)^2 follows chi^2 with K degrees of freedom under the SSM,
    so the bound is chi2.ppf(p, K) - the constraint the original code intended.
    """

    p: float = 0.99

    def __post_init__(self):
        if not 0.0 < self.p < 1.0:
            raise ValueError(f"p must be in (0, 1), got {self.p}")

    def __call__(self, alpha, space):
        beta_sq = float(chi2.ppf(self.p, space.n_components))
        return _rescale_to_ellipsoid(alpha, space, beta_sq)
