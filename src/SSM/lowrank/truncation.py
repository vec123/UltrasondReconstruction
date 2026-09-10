"""The result of a decomposition, and how many modes of it to keep.

Every approximator in this package returns a :class:`Decomposition` -
``(Psi, Lambda)`` plus enough provenance to say how it was obtained and how far
it can be trusted. A GPMM's generative model is Eq. ``sample_gmm_discrete``,

    u = Psi Lambda^{1/2} z,      z ~ N(0, I_L)

so these two arrays *are* the model, and truncating them is choosing how much
of the process to represent.

:class:`Truncation` collects the three criteria in one object rather than
letting each approximator invent its own. They compose by taking the strictest,
which is the convention ``AtlasConfig`` already uses for its ``n_components`` /
``variance_threshold`` pair - so a config that sets both means the same thing
here as it does there.

The discarded variance is not thrown away silently: it is reported as
``residual_variance``, which is the ``epsilon`` of Eq. ``ppca_gen_model`` and
the right value for a ``white`` nugget if the truncated model is to stay honest
about what it cannot represent.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class Decomposition:
    """``K ~ Psi diag(values) Psi^T``, ordered by descending variance.

    ``basis`` is ``(N, L)`` with ``N = block_dim * n_vertices``, in the
    vertex-major layout the rest of ``src/SSM`` uses. ``values`` are variances,
    not standard deviations - the same convention as
    ``StatisticalShapeModel.eigenvalues``, so a decomposition drops straight
    into a model with no rescaling.

    ``basis`` is orthonormal in whichever inner product the operator was
    decomposed in. When a mass matrix was applied (the usual case on a mesh,
    see :func:`src.SSM.lowrank.operator.mass_weighted`) that is the
    ``M``-inner product: ``Psi^T M Psi = I``, not ``Psi^T Psi = I``.
    ``inner_product`` records which, because the difference is invisible in the
    arrays and changes what ``project`` and every orthogonality check mean.
    """

    basis: np.ndarray                       # (N, L)
    values: np.ndarray                      # (L,) variances, descending
    method: str = "?"
    inner_product: str = "euclidean"        # euclidean | mass
    residual_variance: float = 0.0          # trace not represented; Eq. ppca_gen_model
    total_variance: Optional[float] = None  # trace of K, when the method knows it
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.basis = np.ascontiguousarray(np.asarray(self.basis, dtype=np.float64))
        self.values = np.ascontiguousarray(np.asarray(self.values, dtype=np.float64).reshape(-1))
        if self.basis.shape[1] != self.values.size:
            raise ValueError(
                f"{self.basis.shape[1]} basis columns for {self.values.size} eigenvalues"
            )

    @property
    def rank(self) -> int:
        return self.values.size

    @property
    def n_rows(self) -> int:
        return self.basis.shape[0]

    @property
    def retained_variance(self) -> float:
        return float(self.values.sum())

    @property
    def variance_share(self) -> np.ndarray:
        """Fraction of the *represented* variance in each mode."""
        total = self.retained_variance
        return self.values / total if total > 0 else np.zeros_like(self.values)

    @property
    def explained(self) -> float:
        """Fraction of the operator's total variance this decomposition holds."""
        total = self.total_variance
        if total is None:
            total = self.retained_variance + self.residual_variance
        return float(self.retained_variance / total) if total > 0 else 1.0

    def factor(self) -> np.ndarray:
        """``Psi Lambda^{1/2}`` - the ``W`` of ``u = W z``, Eq. ``sample_gmm_discrete``.

        Also the whitened basis: in these coordinates the prior is the identity,
        which is what makes a gradient fit well conditioned and turns the
        Mahalanobis penalty into ``||z||^2``.
        """
        return self.basis * np.sqrt(np.maximum(self.values, 0.0))[None, :]

    def reconstruct(self) -> np.ndarray:
        """The dense ``Psi Lambda Psi^T``. Tests and small domains only."""
        return self.factor() @ self.factor().T

    def truncated(self, truncation: "Truncation") -> "Decomposition":
        return truncation.apply(self)

    def describe(self) -> str:
        return (
            f"{self.method}: rank {self.rank}, {self.explained:.2%} of variance, "
            f"leading mode {self.variance_share[0]:.1%}"
            if self.rank
            else f"{self.method}: empty"
        )


@dataclass(frozen=True)
class Truncation:
    """How many modes to keep. The strictest active criterion wins.

    ``n_components``
        A hard cap on the number of modes.
    ``variance_threshold``
        Keep the leading modes carrying this fraction of the represented
        variance. ``1.0`` keeps everything.
    ``min_eigenvalue``
        Drop modes below an absolute variance. The one to use for numerical
        hygiene: modes at the level of the eigensolver's own error are noise,
        and dividing by their square root in ``project`` amplifies exactly that.

    All three default to "keep everything", so an omitted truncation is a
    no-op rather than a silent 99% cut.
    """

    n_components: Optional[int] = None
    variance_threshold: float = 1.0
    min_eigenvalue: float = 0.0

    def __post_init__(self):
        if self.n_components is not None and self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}")
        if not 0.0 < self.variance_threshold <= 1.0:
            raise ValueError(
                f"variance_threshold must be in (0, 1], got {self.variance_threshold}"
            )
        if self.min_eigenvalue < 0:
            raise ValueError(f"min_eigenvalue must be >= 0, got {self.min_eigenvalue}")

    def count(self, values: np.ndarray) -> int:
        """How many leading modes survive, given descending ``values``."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        keep = values.size

        if self.min_eigenvalue > 0:
            keep = min(keep, int(np.count_nonzero(values >= self.min_eigenvalue)))

        if self.variance_threshold < 1.0:
            total = values.sum()
            if total > 0:
                cumulative = np.cumsum(values) / total
                # searchsorted finds the first index at or past the threshold;
                # +1 turns it into a count, so the kept modes *reach* it rather
                # than stopping just short.
                keep = min(keep, int(np.searchsorted(cumulative, self.variance_threshold) + 1))

        if self.n_components is not None:
            keep = min(keep, int(self.n_components))

        return max(keep, 1) if values.size else 0

    def apply(self, decomposition: Decomposition) -> Decomposition:
        """Cuts the decomposition, moving the discarded variance to the residual."""
        keep = self.count(decomposition.values)
        if keep >= decomposition.rank:
            return decomposition

        dropped = float(decomposition.values[keep:].sum())
        return replace(
            decomposition,
            basis=np.ascontiguousarray(decomposition.basis[:, :keep]),
            values=np.ascontiguousarray(decomposition.values[:keep]),
            residual_variance=decomposition.residual_variance + dropped,
            info={**decomposition.info, "truncated_from": decomposition.rank},
        )


#: Keeps every mode; the default wherever a truncation is optional.
KEEP_ALL = Truncation()
