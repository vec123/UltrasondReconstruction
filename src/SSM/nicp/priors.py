"""Priors on the shape coefficients: what keeps a fit from leaving the model.

All three are quadratic in alpha, so they add a diagonal to the normal
equations, ``(Phi_I^T Phi_I + D) dalpha = Phi_I^T r``, and the closed-form
solver can use them as well as the gradient one.

A note on scale, because it decides whether a weight does anything at all. The
eigenfunctions of this model are orthonormal, so ``Phi_I^T Phi_I`` has
eigenvalues below 1: an isotropic weight of 1 already dominates the data term
and anything past ~10 freezes the fit outright.

The mahalanobis weight would live nowhere near that range if taken literally -
on the baby model the eigenvalues run 5.3e-7 to 7.0e-4, so ``weight/lambda``
needs a weight around 3e-6 to be comparable, and ~30 restrains even the freest
mode 1e5 times harder than the data pulls on it. Its ``normalize`` flag (on by
default) divides that scale out, so both weights share the same [0, ~10] range
and mean the same thing on the dominant mode. Turn it off to recover the
literal MAP form, in which the weight is the scan's noise variance.

Those figures moved by a factor of ``n_vertices`` when the .mat's eigenvalues
were put into the same inner product as its eigenfunctions (see
:func:`src.model.load.rescale_eigenvalues`). Only the unnormalised form is
affected: ``normalize=True`` divides by ``lambda_max``, so a uniform rescaling
of the spectrum cancels and those weights mean exactly what they did before. To
read an unnormalised weight tuned before that fix, multiply it by 31027.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from src.SSM.nicp.registry import Registry
from src.SSM.nicp.shape import ShapeSpace

PRIORS = Registry("prior")
register_prior = PRIORS.register


def available_priors() -> List[str]:
    return PRIORS.names()


def build_prior(name: str, kwargs: Optional[Dict[str, Any]] = None) -> "Prior":
    return PRIORS.build(name, kwargs)


class Prior(ABC):
    name = "?"

    @abstractmethod
    def __call__(self, alpha: torch.Tensor, space: ShapeSpace) -> torch.Tensor:
        """Scalar penalty added to the data term."""

    def quadratic_diagonal(self, space: ShapeSpace) -> Optional[torch.Tensor]:
        """(K,) diagonal this prior contributes to the normal equations.

        ``None`` means "not expressible in closed form", which makes the
        normal-equations solver refuse the fit rather than quietly drop it.
        """
        return None

    def describe(self) -> str:
        return repr(self)


@register_prior("none")
@dataclass(frozen=True)
class NoPrior(Prior):
    """Unregularised: the fit is free to leave the plausible range of the model."""

    def __call__(self, alpha, space):
        return torch.zeros((), dtype=alpha.dtype, device=alpha.device)

    def quadratic_diagonal(self, space):
        return torch.zeros(space.n_components, dtype=space.dtype, device=space.device)


@register_prior("isotropic")
@dataclass(frozen=True)
class IsotropicPrior(Prior):
    """``weight * ||alpha||^2`` - the damping the original pipeline calls gamma.

    Treats every mode as equally likely, which the SSM does not: it is a
    Levenberg-Marquardt style step-size control rather than the model's own
    prior. Use ``mahalanobis`` for the statistically motivated one.
    """

    weight: float = 0.0

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(f"weight must be >= 0, got {self.weight}")

    def __call__(self, alpha, space):
        return self.weight * alpha.pow(2).sum()

    def quadratic_diagonal(self, space):
        return torch.full((space.n_components,), float(self.weight),
                          dtype=space.dtype, device=space.device)


@register_prior("mahalanobis")
@dataclass(frozen=True)
class MahalanobisPrior(Prior):
    """``sum(alpha_i^2 / lambda_i)``, scaled - the model's own Gaussian prior.

    Penalises each mode by how much variance the training set actually gave it,
    so rare modes are held back and dominant ones stay free. This is the term a
    MAP fit is supposed to carry.

    ``normalize`` decides what ``weight`` means, and the two scales are nowhere
    near each other:

    normalize=True (default)
        ``D_i = weight * lambda_max / lambda_i``. The dominant mode gets exactly
        the restraint ``isotropic`` would give at the same weight, and the rest
        are held proportionally harder. So the [0, ~10] feel of the isotropic
        weight carries over, while the mode-dependent shape - the actual point
        of this prior - is untouched.

    normalize=False
        ``D_i = weight / lambda_i``, the literal MAP form, in which ``weight``
        *is* the scan's noise variance in squared model units. Statistically
        exact and impossible to guess: with the baby model's eigenvalues around
        5e-7..7e-4, a workable value is near 3e-6, and anything as large as ~30
        restrains even the freest mode ~1e5 times harder than the data pulls on
        it, which freezes the fit at the mean shape. Weights tuned before the
        eigenvalue rescaling described in this module's docstring are 31027x
        smaller than their equivalent here.
    """

    weight: float = 0.0
    normalize: bool = True

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(f"weight must be >= 0, got {self.weight}")

    def _scale(self, space: ShapeSpace) -> float:
        """lambda_max under normalisation, so weight 1 matches isotropic 1 there."""
        return float(space.eigenvalues.max()) if self.normalize else 1.0

    def __call__(self, alpha, space):
        return self.weight * self._scale(space) * space.mahalanobis_sq(alpha)

    def quadratic_diagonal(self, space):
        return (self.weight * self._scale(space)) / space.eigenvalues
