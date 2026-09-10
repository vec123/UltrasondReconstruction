"""Exact decompositions: the reference the approximate ones are judged against.

Two routes, both from §Sampling new shapes, and the theory proves them
equivalent:

* the Cholesky ``K = L L^T`` with ``W = L``, and
* the Mercer expansion ``u = Psi Lambda^{1/2} z``.

At fetus scale (17229 coordinates, a 2.4 GB matrix) either runs and they should
agree to machine precision - which is what makes this module the yardstick for
``randomized``, ``nystroem`` and ``pivoted_cholesky``. At baby scale (93081
coordinates, 69 GB) neither runs, which is why the other three exist. So the
choice between the theory's two routes is not a preference: past a few thousand
vertices the eigen route is the only branch that executes at all.

:class:`ExactFactor` is the third case and the important one in practice. A
``pca`` or ``laplace_beltrami`` kernel *arrives* factored - the decomposition is
what it stores - so there is nothing to solve. Running an eigensolver over it
would be slower and strictly less accurate than reading the factor and
orthogonalising it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg as sla

from src.SSM.lowrank.operator import (
    Approximator,
    Decomposition,
    register_approximator,
    sorted_eigenpairs,
)


@register_approximator("dense_eigh")
@dataclass(frozen=True)
class DenseEigh(Approximator):
    """Forms the matrix and solves it exactly.

    ``n_components`` asks ``eigh`` for only the leading eigenpairs, which is
    cheaper than a full decomposition but does not change the memory needed to
    form the matrix - and forming it is the binding constraint. Treat this as
    "small domains and tests".
    """

    n_components: Optional[int] = None
    needs_indexing = False

    def decompose(self, operator) -> Decomposition:
        if not hasattr(operator, "to_dense"):
            raise TypeError(
                f"dense_eigh needs an operator that can be formed; "
                f"{type(operator).__name__} cannot. Use 'randomized' or 'pivoted_cholesky'."
            )

        A = np.asarray(operator.to_dense(), dtype=np.float64)
        A = 0.5 * (A + A.T)          # kill the asymmetry rounding leaves behind
        total = float(np.trace(A))

        if self.n_components is not None and self.n_components < A.shape[0]:
            lo = A.shape[0] - int(self.n_components)
            values, vectors = sla.eigh(A, subset_by_index=[lo, A.shape[0] - 1])
        else:
            values, vectors = sla.eigh(A)

        values, vectors = sorted_eigenpairs(values, vectors)
        return Decomposition(
            basis=vectors,
            values=values,
            method="dense_eigh",
            total_variance=total,
            residual_variance=max(total - float(values.sum()), 0.0),
        )


@register_approximator("exact_factor")
@dataclass(frozen=True)
class ExactFactor(Approximator):
    """Uses the operator's own factor, when it has one.

    A kernel whose structure is ``LOWRANK`` carries ``F`` with ``K = F F^T``
    explicitly: ``pca`` holds ``Phi sqrt(Lambda)``, ``laplace_beltrami`` holds
    the filtered spectrum, and a sum of low-rank kernels concatenates theirs.
    The eigendecomposition then costs one thin SVD of ``F`` - ``O(N r^2)``,
    with ``r`` the rank rather than the dimension - and is exact.

    Worth checking for before anything else: on the baby model this turns what
    would be a 69 GB eigenproblem into an SVD of a 93081 x 163 matrix.
    """

    needs_indexing = False

    def decompose(self, operator) -> Decomposition:
        F = operator.factor() if hasattr(operator, "factor") else None
        if F is None:
            raise TypeError(
                "exact_factor needs an operator with an exact low-rank factor; this one "
                "has none (structure is not LOWRANK). Use 'pivoted_cholesky' or "
                "'randomized'."
            )

        F = np.asarray(F, dtype=np.float64)
        left, singular, _ = np.linalg.svd(F, full_matrices=False)
        values = singular ** 2
        total = float(values.sum())

        values, vectors = sorted_eigenpairs(values, left)
        return Decomposition(
            basis=vectors,
            values=values,
            method="exact_factor",
            total_variance=total,
            residual_variance=0.0,
            info={"factor_rank": F.shape[1]},
        )


def cholesky_factor(operator, jitter: float = 0.0) -> np.ndarray:
    """``L`` with ``K = L L^T`` - the theory's ``W = L`` of §Sampling new shapes.

    Kept because the equivalence with ``Psi Lambda^{1/2}`` is a checkable
    identity and worth checking: the two generate the same distribution, so a
    disagreement means one of them is wrong. Not a production path - a Cholesky
    factor is as large as the matrix, so this is bounded by the same limit that
    rules out ``dense_eigh``.

    ``jitter`` is added to the diagonal when the matrix is positive
    *semi*-definite, which a truncated or rank-deficient kernel always is:
    Cholesky needs strict positivity and fails on the null space otherwise.
    """
    A = np.asarray(operator.to_dense(), dtype=np.float64)
    A = 0.5 * (A + A.T)
    if jitter:
        A = A + jitter * np.eye(A.shape[0])
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            f"{exc}. The kernel is positive semi-definite rather than definite - add a "
            f"'white' term, or pass jitter=<small multiple of the trace>."
        ) from exc
