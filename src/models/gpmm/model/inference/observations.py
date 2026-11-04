"""Noisy observations of some of a model's vertices - what a posterior conditions on.

A registration that deforms the model's own mean shape (resnet_lddmm on the
baby mean, an NICP fit, ...) hands back one position per model vertex, in
correspondence by index. Not every one of them is worth believing: where the
scan has no data, the registration only extrapolates. A per-vertex error (the
recon-to-target distance, typically) says which is which, and a
:class:`NoiseModel` turns that error into an observation standard deviation.

Both readings of "trust only the good vertices" are one noise model each:

    cutoff        sigma below a threshold, inf (= unobserved) above it
    logistic      sigma rising smoothly to sigma_max around a threshold;
                  width -> 0 recovers the cutoff
    linear        sigma + slope * error
    exponential   sigma * exp(error / scale)

An infinite std drops the vertex: it is left to the model's prior rather than
observed with zero weight, which is the same answer at a lower cost.

The noise model works in the error's own units (mm for recon_target_distance),
which are also the units of the observed positions. :meth:`Observations.transformed`
carries both into the model frame together, so a sigma in a config never has
to be restated in model units.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit

from src.shared.core.registry import TypedRegistry

NOISE_MODELS = TypedRegistry("noise_model")
register_noise_model = NOISE_MODELS.register


def available_noise_models() -> List[str]:
    return NOISE_MODELS.names()


def build_noise_model(name: str, kwargs: Optional[Dict[str, Any]] = None) -> "NoiseModel":
    return NOISE_MODELS.build(name, kwargs)


class NoiseModel(ABC):
    """Per-vertex observation std from a per-vertex error, in the error's units."""

    @abstractmethod
    def std(self, error: np.ndarray) -> np.ndarray:
        """(R,) std for (R,) errors; ``inf`` marks a vertex as unobserved."""

    def variance(self, error: np.ndarray) -> np.ndarray:
        return self.std(error) ** 2

    def describe(self) -> str:
        return repr(self)


def _positive(**values):
    for name, value in values.items():
        if not value > 0:
            raise ValueError(f"{name} must be > 0, got {value}")


@register_noise_model("cutoff")
@dataclass(frozen=True)
class CutoffNoise(NoiseModel):
    """``sigma`` where ``error < threshold``, unobserved elsewhere."""

    sigma: float = 0.5
    threshold: float = 2.0

    def __post_init__(self):
        _positive(sigma=self.sigma, threshold=self.threshold)

    def std(self, error):
        error = np.asarray(error, dtype=np.float64)
        return np.where(error < self.threshold, self.sigma, np.inf)


@register_noise_model("logistic")
@dataclass(frozen=True)
class LogisticNoise(NoiseModel):
    """A soft cutoff, blended in the weight ``1 / std^2``:

        1 / std^2 = (1 - s) / sigma^2 + s / sigma_max^2,
        s = sigmoid((error - threshold) / width)

    Every vertex is observed, but past ``threshold`` its weight falls from
    ``1 / sigma^2`` to ``1 / sigma_max^2`` within a few ``width``; width -> 0
    is the cutoff. Blending the weight rather than the std is what keeps the
    good side at ``sigma``: a std blend leaves ``s * sigma_max`` there, which
    for sigma_max = 100 is millimetres even far below the threshold. Choose
    ``sigma_max`` well above the prior's own std (~1-3 mm on the baby models),
    so a vertex there barely moves the posterior; ``inf`` drops the vertices
    whose weight underflows.
    """

    sigma: float = 0.5
    sigma_max: float = 100.0
    threshold: float = 2.0
    width: float = 0.5

    def __post_init__(self):
        _positive(sigma=self.sigma, threshold=self.threshold, width=self.width)
        if not self.sigma_max >= self.sigma:
            raise ValueError(f"sigma_max must be >= sigma, got {self.sigma_max} < {self.sigma}")

    def std(self, error):
        error = np.asarray(error, dtype=np.float64)
        rise = expit((error - self.threshold) / self.width)
        weight = (1.0 - rise) / self.sigma ** 2 + rise / self.sigma_max ** 2
        with np.errstate(divide="ignore"):
            return 1.0 / np.sqrt(weight)


@register_noise_model("linear")
@dataclass(frozen=True)
class LinearNoise(NoiseModel):
    """``sigma + slope * error``."""

    sigma: float = 0.5
    slope: float = 1.0

    def __post_init__(self):
        _positive(sigma=self.sigma)
        if self.slope < 0:
            raise ValueError(f"slope must be >= 0, got {self.slope}")

    def std(self, error):
        return self.sigma + self.slope * np.asarray(error, dtype=np.float64)


@register_noise_model("exponential")
@dataclass(frozen=True)
class ExponentialNoise(NoiseModel):
    """``sigma * exp(error / scale)``; overflows to inf, i.e. unobserved."""

    sigma: float = 0.5
    scale: float = 1.0

    def __post_init__(self):
        _positive(sigma=self.sigma, scale=self.scale)

    def std(self, error):
        with np.errstate(over="ignore"):
            return self.sigma * np.exp(np.asarray(error, dtype=np.float64) / self.scale)


@dataclass
class Observations:
    """Observed positions of R model vertices, each with an isotropic noise variance.

    ``indices`` address the model's own vertex ordering, so the design matrix
    of a fit is just :meth:`~src.models.gpmm.model.model.StatisticalShapeModel.phi_rows`.
    ``variances`` are in the squared units of ``targets`` and must be finite:
    a vertex nobody trusts is left out, not given an infinite variance.
    """

    indices: np.ndarray     # (R,) model vertex ids
    targets: np.ndarray     # (R, 3) observed positions
    variances: np.ndarray   # (R,) per-vertex isotropic noise variance

    def __post_init__(self):
        self.indices = np.asarray(self.indices, dtype=np.int64).reshape(-1)
        self.targets = np.asarray(self.targets, dtype=np.float64).reshape(-1, 3)
        self.variances = np.asarray(self.variances, dtype=np.float64).reshape(-1)
        if not len(self.indices) == len(self.targets) == len(self.variances):
            raise ValueError(
                f"{len(self.indices)} indices, {len(self.targets)} targets and "
                f"{len(self.variances)} variances - they must match"
            )
        if np.any(~np.isfinite(self.variances)) or np.any(self.variances <= 0):
            raise ValueError("variances must be finite and > 0; leave unobserved vertices out")
        if len(np.unique(self.indices)) != len(self.indices):
            raise ValueError("each vertex may be observed only once")

    @classmethod
    def from_errors(cls, points, errors, noise_model: NoiseModel, indices=None) -> "Observations":
        """Observes every vertex whose noise std is finite.

        ``points`` (R, 3) and ``errors`` (R,) are by vertex; ``indices``
        defaults to ``arange(R)``, i.e. ``points`` is the whole model mesh.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        errors = np.asarray(errors, dtype=np.float64).reshape(-1)
        indices = np.arange(len(points)) if indices is None else np.asarray(indices).reshape(-1)
        if not len(points) == len(errors) == len(indices):
            raise ValueError(f"{len(points)} points, {len(errors)} errors, {len(indices)} indices")

        std = noise_model.std(errors)
        keep = np.isfinite(std) & np.isfinite(errors)
        return cls(indices=indices[keep], targets=points[keep], variances=std[keep] ** 2)

    @property
    def n(self) -> int:
        return len(self.indices)

    @property
    def stds(self) -> np.ndarray:
        return np.sqrt(self.variances)

    def subset(self, mask) -> "Observations":
        mask = np.asarray(mask)
        return Observations(self.indices[mask], self.targets[mask], self.variances[mask])

    def thinned(self, spacing: float) -> "Observations":
        """A subset with no two targets closer than ``spacing``, best-trusted first.

        Neighbouring vertices of a registration share nearly the same error, so
        the full set counts one error many times over; keeping about one
        observation per ``spacing`` brings them closer to the independent
        measurements the posterior assumes. Greedy Poisson-disk: in order of
        increasing variance, a target is kept unless a kept one lies within
        ``spacing``. In the units of ``targets``; ``spacing <= 0`` keeps all.
        """
        if spacing <= 0 or not self.n:
            return self
        tree = cKDTree(self.targets)
        kept = np.zeros(self.n, dtype=bool)
        blocked = np.zeros(self.n, dtype=bool)
        for i in np.argsort(self.variances, kind="stable"):
            if blocked[i]:
                continue
            kept[i] = True
            blocked[tree.query_ball_point(self.targets[i], spacing)] = True
        return self.subset(kept)

    def transformed(self, transform) -> "Observations":
        """The same observations in another frame, for a ``SimilarityTransform``.

        Positions move with the transform; a variance scales by ``scale**2``,
        since a similarity stretches every length by ``scale``.
        """
        return Observations(
            indices=self.indices,
            targets=transform.apply_points(self.targets),
            variances=self.variances * float(transform.scale) ** 2,
        )

    def precision(self, extra_variance: float = 0.0) -> np.ndarray:
        """(3R,) per-coordinate weights ``1 / (variance + extra_variance)``.

        Vertex-major, to line up with ``phi_rows(indices)``. ``extra_variance``
        is where a model's own unrepresented ``noise`` joins the observation's.
        """
        return np.repeat(1.0 / (self.variances + float(extra_variance)), 3)

    def mask(self, n_vertices: int) -> np.ndarray:
        """(n_vertices,) bool, True where a vertex is observed."""
        mask = np.zeros(int(n_vertices), dtype=bool)
        mask[self.indices] = True
        return mask

    def describe(self) -> str:
        if not self.n:
            return "Observations(n=0)"
        std = self.stds
        return (f"Observations(n={self.n}, std min/median/max="
                f"{std.min():.4g}/{np.median(std):.4g}/{std.max():.4g})")
