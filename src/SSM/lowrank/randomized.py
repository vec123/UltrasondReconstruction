"""Randomised range finding - §Solving the Eigenvalue Problem, made matrix-free.

The theory's construction verbatim: draw a Gaussian test matrix ``Omega``,
push it through the operator to find where the operator's action lives, and
solve the small problem there.

    Y = K Omega          Y = Q R          B = Q^T K          U = Q U_B

Two departures from the text, both deliberate.

**Rayleigh-Ritz instead of the SVD of B.** For a symmetric operator, forming
``C = Q^T K Q`` (``k x k``) and taking ``eigh(C)`` gives the same eigenpairs as
the SVD of the ``k x n`` matrix ``B``, at the same number of matrix products
and ``O(k^3)`` instead of ``O(k^2 n)``. It also returns a symmetric
decomposition directly, rather than singular values that then have to be
argued back into eigenvalues.

**Power iterations.** ``Y = K Omega`` captures the dominant subspace well only
when the spectrum decays sharply. Replacing it with ``(K K)^q K Omega`` raises
the ratio between wanted and unwanted eigenvalues to the power ``2q + 1``, which
is what makes the method usable on a kernel with a slowly decaying spectrum -
and a sum containing a ``matern12`` or a ``white`` term decays slowly. Two
iterations is the standard default and costs four extra products.

Each product is re-orthonormalised. Without it the columns of ``Y`` collapse
onto the leading eigenvector within two or three iterations in float64, and the
resulting basis silently spans far less than it claims.

**Where it fits.** This is the method that needs only ``matmat``, so it is the
one that works on any operator - including a sum whose terms have different
structures, where no factor exists and no column can be cheaply selected. Its
cost is ``(2q + 1)`` passes over the operator, so it is the right choice when a
product is cheap (a low-rank or sparse kernel) and the wrong one when each
product means forming an ``n^2`` block - there, ``pivoted_cholesky`` touches
``r`` columns instead of ``k`` full passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.SSM.lowrank.operator import (
    Approximator,
    Decomposition,
    register_approximator,
    sorted_eigenpairs,
)


@register_approximator("randomized")
@dataclass(frozen=True)
class Randomized(Approximator):
    """Halko's randomised range finder, then Rayleigh-Ritz on the subspace.

    ``n_components``
        How many eigenpairs are wanted.
    ``oversampling``
        Extra random directions, discarded at the end. They are what makes the
        estimate of the wanted subspace reliable rather than marginal; 10 is
        the standard default and rarely needs changing.
    ``power_iterations``
        ``q`` above. Raise it when the spectrum is flat - if the returned
        eigenvalues do not decay, that is the symptom.
    ``seed``
        Fixes the test matrix. The method is stochastic, so two runs differ
        slightly; a seed makes a build reproducible, which matters because the
        basis is cached and reused.
    """

    n_components: int = 100
    oversampling: int = 10
    power_iterations: int = 2
    seed: Optional[int] = None
    needs_indexing = False

    def __post_init__(self):
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}")
        if self.oversampling < 0:
            raise ValueError(f"oversampling must be >= 0, got {self.oversampling}")
        if self.power_iterations < 0:
            raise ValueError(f"power_iterations must be >= 0, got {self.power_iterations}")

    def decompose(self, operator) -> Decomposition:
        n = operator.shape[0]
        sketch = min(n, int(self.n_components) + int(self.oversampling))

        rng = np.random.default_rng(self.seed)
        Omega = rng.standard_normal((n, sketch))

        Y = np.asarray(operator.matmat(Omega))
        Q, _ = np.linalg.qr(Y)

        for _ in range(int(self.power_iterations)):
            # Re-orthonormalise between every product: in float64 the columns
            # otherwise collapse onto the leading eigenvector after two or
            # three passes, and the loss is invisible in the output.
            Z, _ = np.linalg.qr(np.asarray(operator.matmat(Q)))
            Q, _ = np.linalg.qr(np.asarray(operator.matmat(Z)))

        KQ = np.asarray(operator.matmat(Q))
        C = Q.T @ KQ
        C = 0.5 * (C + C.T)

        values, vectors = np.linalg.eigh(C)
        values, vectors = sorted_eigenpairs(values, vectors)

        keep = min(int(self.n_components), values.size)
        values = values[:keep]
        basis = np.ascontiguousarray(Q @ vectors[:, :keep])

        total = None
        if hasattr(operator, "diag"):
            total = float(np.asarray(operator.diag()).sum())

        residual = 0.0 if total is None else max(total - float(values.sum()), 0.0)
        return Decomposition(
            basis=basis,
            values=values,
            method="randomized",
            total_variance=total,
            residual_variance=residual,
            info={"sketch": sketch, "power_iterations": int(self.power_iterations)},
        )
