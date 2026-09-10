"""Parameterised kernels: the prior half of a GPMM.

These are the ``k_smooth`` of §Gaussian Process Morphable Models - covariances
written down rather than estimated, which is what lets a model deform in ways
the training population never showed. Each is a function of distance alone, so
each becomes an *intrinsic* kernel for free when the domain reports geodesic
distances instead of Euclidean ones: nothing here knows which it was given.

All are scalar (``block_dim = 1``), i.e. they stand for ``k(p,p') * I_3``. How
that meets a matrix-valued kernel differs by operation and is decided in
``algebra.py`` - added, it is genuinely lifted; multiplied, it scales each 3x3
block instead, which is not the same thing.

**Choosing between them.** The length scale sets how far a deformation
propagates and is the parameter worth tuning first; ``scale`` sets how much
variance the term contributes and only matters relative to the other terms in a
sum. Smoothness is the second choice: ``gaussian`` samples are analytic, which
is smoother than any real anatomy and tends to look inflated; the Matern family
is the standard remedy, with ``matern32`` and ``matern52`` giving one and two
mean-square derivatives. ``wendland`` is the one with compact support, so it is
both the local-deformation kernel and the only one whose matrix is genuinely
sparse.

**Units.** Length scales are in the model's own coordinates, and the two models
in this project do not share them - the baby model spans about 0.013 across
while the fetus atlas is in millimetres. A sigma carried from one to the other
is meaningless unless both were put through ``normalize_ssm`` first.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Dict, Optional

import numpy as np

from src.SSM.kernels._array import array_namespace
from src.SSM.kernels.base import (
    DENSE,
    DIAGONAL,
    KERNELS,
    Kernel,
    KernelStructure,
    cross_distances,
    log_params,
    lowrank,
    pairwise_distances,
    read_log,
    register_kernel,
    resolve_indices,
)

SQRT3 = math.sqrt(3.0)
SQRT5 = math.sqrt(5.0)


@dataclass(frozen=True)
class Stationary(Kernel):
    """``k(p, p') = radial(||p - p'||)`` - everything shared by the family.

    Subclasses implement :meth:`radial` against the array namespace of their
    input, so the same line evaluates in numpy for a model build and in torch
    when ``src/SSM/hyper`` differentiates it (see ``_array.py``).
    """

    block_dim = 1

    @property
    def structure(self) -> KernelStructure:
        return DENSE

    @abstractmethod
    def radial(self, d, xp):
        """``radial(0)`` must equal this kernel's ``scale``."""

    def block(self, domain, rows=None, cols=None) -> np.ndarray:
        rows = resolve_indices(domain, rows)
        cols = resolve_indices(domain, cols)
        d = pairwise_distances(domain, rows, cols)
        return self.radial(d, array_namespace(d))

    def diag(self, domain, rows=None) -> np.ndarray:
        rows = resolve_indices(domain, rows)
        return np.full(rows.size, float(self.scale), dtype=np.float64)

    def supports_cross(self) -> bool:
        return True

    def cross(self, domain, X, cols=None) -> np.ndarray:
        cols = resolve_indices(domain, cols)
        d = cross_distances(domain, X, cols)
        return self.radial(d, array_namespace(d))


# --------------------------------------------------------------------------
# the smooth family
# --------------------------------------------------------------------------

@register_kernel("gaussian")
@dataclass(frozen=True)
class Gaussian(Stationary):
    """``scale * exp(-d^2 / 2 sigma^2)`` - the squared-exponential.

    Note the ``2`` in the denominator: this is the Gaussian-process convention,
    not the ``exp(-d^2/sigma^2)`` Deformetrica uses for its LDDMM deformation
    kernel (``kwd`` in an atlas ``params.json``). A width carried over from
    there means ``sigma_here = kwd / sqrt(2)``.

    Its samples are analytic - infinitely differentiable - which is stronger
    smoothness than any anatomy has. It is the right first thing to try and
    frequently the wrong thing to keep; ``matern52`` is the usual replacement.
    """

    sigma: float = 1.0
    scale: float = 1.0

    def __post_init__(self):
        _require_positive(sigma=self.sigma)
        _require_non_negative(scale=self.scale)

    def radial(self, d, xp):
        return self.scale * xp.exp(-(d ** 2) / (2.0 * self.sigma ** 2))

    def free_parameters(self) -> Dict[str, float]:
        return log_params(sigma=self.sigma, scale=self.scale)

    def with_parameters(self, theta) -> "Gaussian":
        return replace(
            self,
            sigma=read_log(theta, "sigma", self.sigma),
            scale=read_log(theta, "scale", self.scale),
        )


@register_kernel("matern12")
@dataclass(frozen=True)
class Matern12(Stationary):
    """``scale * exp(-d / rho)`` - Matern with nu = 1/2, the exponential kernel.

    Continuous but nowhere differentiable: samples are rough. Useful as the
    high-frequency term of a sum, rarely on its own for a surface.
    """

    rho: float = 1.0
    scale: float = 1.0

    def __post_init__(self):
        _require_positive(rho=self.rho)
        _require_non_negative(scale=self.scale)

    def radial(self, d, xp):
        return self.scale * xp.exp(-d / self.rho)

    def free_parameters(self) -> Dict[str, float]:
        return log_params(rho=self.rho, scale=self.scale)

    def with_parameters(self, theta) -> "Matern12":
        return replace(
            self,
            rho=read_log(theta, "rho", self.rho),
            scale=read_log(theta, "scale", self.scale),
        )


@register_kernel("matern32")
@dataclass(frozen=True)
class Matern32(Stationary):
    """``scale * (1 + sqrt3 d/rho) exp(-sqrt3 d/rho)`` - nu = 3/2.

    Once mean-square differentiable. The mildest departure from Gaussian that
    stops samples looking inflated.
    """

    rho: float = 1.0
    scale: float = 1.0

    def __post_init__(self):
        _require_positive(rho=self.rho)
        _require_non_negative(scale=self.scale)

    def radial(self, d, xp):
        t = SQRT3 * d / self.rho
        return self.scale * (1.0 + t) * xp.exp(-t)

    def free_parameters(self) -> Dict[str, float]:
        return log_params(rho=self.rho, scale=self.scale)

    def with_parameters(self, theta) -> "Matern32":
        return replace(
            self,
            rho=read_log(theta, "rho", self.rho),
            scale=read_log(theta, "scale", self.scale),
        )


@register_kernel("matern52")
@dataclass(frozen=True)
class Matern52(Stationary):
    """``scale * (1 + sqrt5 d/rho + 5d^2/3rho^2) exp(-sqrt5 d/rho)`` - nu = 5/2.

    Twice mean-square differentiable. The usual default for a surface prior:
    smooth enough that curvature is well defined, rough enough to be honest.
    """

    rho: float = 1.0
    scale: float = 1.0

    def __post_init__(self):
        _require_positive(rho=self.rho)
        _require_non_negative(scale=self.scale)

    def radial(self, d, xp):
        t = SQRT5 * d / self.rho
        return self.scale * (1.0 + t + (t ** 2) / 3.0) * xp.exp(-t)

    def free_parameters(self) -> Dict[str, float]:
        return log_params(rho=self.rho, scale=self.scale)

    def with_parameters(self, theta) -> "Matern52":
        return replace(
            self,
            rho=read_log(theta, "rho", self.rho),
            scale=read_log(theta, "scale", self.scale),
        )


@register_kernel("rational_quadratic")
@dataclass(frozen=True)
class RationalQuadratic(Stationary):
    """``scale * (1 + d^2 / 2 alpha sigma^2)^-alpha``.

    A scale mixture of Gaussians: one term that behaves like several length
    scales at once, with ``alpha`` controlling the spread of the mixture. As
    ``alpha -> inf`` it becomes ``gaussian(sigma)``. Worth reaching for when a
    sum of two Gaussians is doing better than one but you would rather not
    tune two widths.
    """

    sigma: float = 1.0
    alpha: float = 1.0
    scale: float = 1.0

    def __post_init__(self):
        _require_positive(sigma=self.sigma, alpha=self.alpha)
        _require_non_negative(scale=self.scale)

    def radial(self, d, xp):
        return self.scale * (1.0 + (d ** 2) / (2.0 * self.alpha * self.sigma ** 2)) ** (-self.alpha)

    def free_parameters(self) -> Dict[str, float]:
        return log_params(sigma=self.sigma, alpha=self.alpha, scale=self.scale)

    def with_parameters(self, theta) -> "RationalQuadratic":
        return replace(
            self,
            sigma=read_log(theta, "sigma", self.sigma),
            alpha=read_log(theta, "alpha", self.alpha),
            scale=read_log(theta, "scale", self.scale),
        )


@register_kernel("wendland")
@dataclass(frozen=True)
class Wendland(Stationary):
    """``scale * (1 - r)^4 (4r + 1)`` for ``r = d/radius <= 1``, zero beyond.

    Wendland's C^2 function, positive definite in R^3. The one kernel here whose
    matrix is genuinely **sparse**, which makes it the affordable route to local
    deformation at 31027 vertices where a Gaussian's dense 7.7 GB is not.

    Two things follow from compact support and are easy to get wrong:

    * The radius is a hard cutoff, not a decay rate. Set it to the size of the
      region you want to move, not to the ``sigma`` of a Gaussian you are
      replacing - matching the two by eye lands the radius roughly ``3 sigma``.
    * Too small a radius makes ``K`` numerically block-diagonal, and the
      resulting model deforms each patch independently with visible seams. If
      samples look faceted, the radius is below the mesh's own feature scale.

    ``structure`` reports ``SPARSE``, which describes the mathematics; ``block``
    still returns a dense array with zeros in it. Consumers go through
    ``matmat``/``block`` and are unaffected. A sparse assembly is a worthwhile
    later optimisation, not a correctness matter.
    """

    radius: float = 1.0
    scale: float = 1.0

    def __post_init__(self):
        _require_positive(radius=self.radius)
        _require_non_negative(scale=self.scale)

    @property
    def structure(self) -> KernelStructure:
        from src.SSM.kernels.base import SPARSE

        return SPARSE

    def radial(self, d, xp):
        r = d / self.radius
        inside = xp.maximum(1.0 - r, 0.0)
        return self.scale * (inside ** 4) * (4.0 * r + 1.0)

    def free_parameters(self) -> Dict[str, float]:
        return log_params(radius=self.radius, scale=self.scale)

    def with_parameters(self, theta) -> "Wendland":
        return replace(
            self,
            radius=read_log(theta, "radius", self.radius),
            scale=read_log(theta, "scale", self.scale),
        )


# --------------------------------------------------------------------------
# the degenerate two
# --------------------------------------------------------------------------

@register_kernel("white")
@dataclass(frozen=True)
class White(Kernel):
    """``variance`` on the diagonal, zero off it - the nugget.

    Two distinct jobs, and it is worth knowing which one a given value is doing:

    * **Jitter.** A tiny value (1e-8 relative to the trace) makes a kernel that
      is positive semi-definite in theory positive *definite* in floating point,
      so Cholesky and the eigensolvers stop failing on near-null directions.
    * **Model noise.** The ``epsilon`` of Eq. ``ppca_gen_model``: the variance a
      truncated model does not represent. On the baby model that is knowable
      rather than guessable - its 163 modes retain 99.01% of the variance, so
      the missing 0.99% is what belongs here.

    ``analytic_part`` returns ``None`` even though ``k(x*, x_j) = 0`` is a
    perfectly good closed form. It is a deliberate override: a white kernel
    carries no information away from the sample points, so using it as the
    kernel to extend against would silently produce a basis that is zero
    everywhere off the mesh. A model that wants continuous evaluation needs a
    genuinely spatial term, and failing loudly in ``extension.py`` is better
    than returning zeros.
    """

    variance: float = 1.0
    block_dim = 1

    def __post_init__(self):
        _require_non_negative(variance=self.variance)

    @property
    def structure(self) -> KernelStructure:
        return DIAGONAL

    def block(self, domain, rows=None, cols=None) -> np.ndarray:
        rows = resolve_indices(domain, rows)
        cols = resolve_indices(domain, cols)
        return self.variance * (rows[:, None] == cols[None, :]).astype(np.float64)

    def diag(self, domain, rows=None) -> np.ndarray:
        rows = resolve_indices(domain, rows)
        return np.full(rows.size, float(self.variance), dtype=np.float64)

    def supports_cross(self) -> bool:
        return True

    def cross(self, domain, X, cols=None) -> np.ndarray:
        cols = resolve_indices(domain, cols)
        n_new = np.asarray(X, dtype=np.float64).reshape(-1, 3).shape[0]
        return np.zeros((n_new, cols.size), dtype=np.float64)

    def analytic_part(self) -> Optional[Kernel]:
        return None

    def free_parameters(self) -> Dict[str, float]:
        return log_params(variance=self.variance)

    def with_parameters(self, theta) -> "White":
        return replace(self, variance=read_log(theta, "variance", self.variance))


@register_kernel("constant")
@dataclass(frozen=True)
class Constant(Kernel):
    """``variance`` everywhere - a rank-one kernel.

    Models a shared offset: under it every point moves together, so on its own
    it generates pure translation. Its use is as a term in a sum, where it lets
    the model absorb a global shift without spending a smooth mode on it.
    """

    variance: float = 1.0
    block_dim = 1

    def __post_init__(self):
        _require_non_negative(variance=self.variance)

    @property
    def structure(self) -> KernelStructure:
        return lowrank(1)

    def block(self, domain, rows=None, cols=None) -> np.ndarray:
        rows = resolve_indices(domain, rows)
        cols = resolve_indices(domain, cols)
        return np.full((rows.size, cols.size), float(self.variance), dtype=np.float64)

    def diag(self, domain, rows=None) -> np.ndarray:
        rows = resolve_indices(domain, rows)
        return np.full(rows.size, float(self.variance), dtype=np.float64)

    def factor(self, domain, rows=None) -> np.ndarray:
        rows = resolve_indices(domain, rows)
        return np.full((rows.size, 1), math.sqrt(self.variance), dtype=np.float64)

    def matmat(self, domain, V, rows=None, cols=None, chunk_bytes=None):
        rows = resolve_indices(domain, rows)
        cols = resolve_indices(domain, cols)
        V = np.asarray(V)
        squeeze = V.ndim == 1
        if squeeze:
            V = V[:, None]
        total = V.sum(axis=0, keepdims=True)
        out = np.repeat(self.variance * total, rows.size, axis=0)
        return out[:, 0] if squeeze else out

    def supports_cross(self) -> bool:
        return True

    def cross(self, domain, X, cols=None) -> np.ndarray:
        cols = resolve_indices(domain, cols)
        n_new = np.asarray(X, dtype=np.float64).reshape(-1, 3).shape[0]
        return np.full((n_new, cols.size), float(self.variance), dtype=np.float64)

    def free_parameters(self) -> Dict[str, float]:
        return log_params(variance=self.variance)

    def with_parameters(self, theta) -> "Constant":
        return replace(self, variance=read_log(theta, "variance", self.variance))


# --------------------------------------------------------------------------

def _require_positive(**values):
    for key, value in values.items():
        if not value > 0:
            raise ValueError(f"{key} must be positive, got {value}")


def _require_non_negative(**values):
    for key, value in values.items():
        if value < 0:
            raise ValueError(f"{key} must be >= 0, got {value}")
