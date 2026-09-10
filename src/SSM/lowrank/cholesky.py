"""Pivoted Cholesky - the default at baby scale.

Greedy low-rank factorisation that touches the kernel only through its diagonal
and one block of columns per step. That is what makes it the method of choice
where the matrix cannot be formed and a matrix-vector product is expensive:

    baby model, 3n = 93081
      dense K                     69 GB          impossible
      randomized, k = 300         300 full passes over the operator
      pivoted Cholesky, r = 300   300 columns    223 MB

Each step picks the vertex carrying the most unexplained variance, evaluates the
kernel between it and everything else, and subtracts what that explains. The
remaining diagonal is the error, exactly - so unlike the randomised method this
one *knows* how wrong it is at every step, and ``tolerance`` is a real
guarantee on the trace rather than a hope about the spectrum.

**Block pivoting.** For a matrix-valued kernel the three coordinates of a vertex
are eliminated together rather than one at a time. They share a kernel column
evaluation, so this costs a third of the kernel work of scalar pivoting; it also
keeps the factorisation from splitting a vertex's 3x3 block across steps, which
is what would happen otherwise and would let a partially-eliminated vertex be
re-selected.

**When not to use it.** Greedy pivoting follows the diagonal, so it is at its
weakest when the diagonal is uninformative - a stationary kernel has a constant
diagonal, and the first pivots are then arbitrary. It recovers within a few
steps because the *residual* diagonal does become informative, but on a purely
stationary kernel with a slowly decaying spectrum ``randomized`` is the better
estimator. On anything containing a data or localised term - where the diagonal
varies over the surface, which is the realistic case - this is the stronger
method.
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

#: A pivot below this fraction of the trace carries only rounding.
_EXHAUSTED = 1e-14


@register_approximator("pivoted_cholesky")
@dataclass(frozen=True)
class PivotedCholesky(Approximator):
    """Greedy ``K ~ L L^T``, then a thin SVD of ``L`` for the eigenpairs.

    ``max_rank``
        Hard cap on the number of columns, and the memory bound: ``L`` is
        ``N x max_rank``.
    ``tolerance``
        Stop once the unexplained trace falls below this fraction of the total.
        ``1e-6`` is a tight fit; ``1e-3`` is usually indistinguishable in a
        sampled shape and much cheaper.
    ``jitter``
        Added to a pivot block before its Cholesky. A pivot block can be
        numerically indefinite once its variance is nearly exhausted, which is
        precisely when it is about to be discarded anyway.
    """

    max_rank: int = 300
    tolerance: float = 1e-6
    jitter: float = 1e-12
    needs_indexing = True

    def __post_init__(self):
        if self.max_rank < 1:
            raise ValueError(f"max_rank must be >= 1, got {self.max_rank}")
        if self.tolerance < 0:
            raise ValueError(f"tolerance must be >= 0, got {self.tolerance}")

    def decompose(self, operator) -> Decomposition:
        block_dim = int(getattr(operator, "block_dim", 1))
        N = operator.shape[0]
        n_vertices = N // block_dim

        residual_diag = np.asarray(operator.diag(), dtype=np.float64).copy()
        total = float(residual_diag.sum())
        if total <= 0:
            raise ValueError("the operator has zero trace - there is nothing to decompose")

        max_columns = min(int(self.max_rank), N)
        L = np.zeros((N, max_columns), dtype=np.float64)
        pivots: list[int] = []
        used = np.zeros(n_vertices, dtype=bool)
        filled = 0
        remaining = total

        while filled + block_dim <= max_columns:
            # The stopping test is on the *total* unexplained trace, not on the
            # largest pivot. With n vertices the average vertex carries 1/n of
            # the trace, so comparing a single pivot against a fraction of the
            # total would stop before the first step on any large mesh - at
            # n = 31027 no vertex clears even a 1e-4 threshold.
            if remaining <= self.tolerance * total:
                break

            per_vertex = residual_diag.reshape(n_vertices, block_dim).sum(axis=1)
            per_vertex[used] = -np.inf
            vertex = int(np.argmax(per_vertex))

            # Separately: a pivot at the level of rounding is exhausted, and
            # eliminating it would divide noise by noise.
            if per_vertex[vertex] <= _EXHAUSTED * total:
                break

            coords = block_dim * vertex + np.arange(block_dim)

            # One kernel column block, minus what the existing factor explains.
            column = np.asarray(operator.block(None, [vertex]), dtype=np.float64)
            column = column.reshape(N, block_dim)
            if filled:
                column = column - L[:, :filled] @ L[coords, :filled].T

            pivot_block = column[coords]
            pivot_block = 0.5 * (pivot_block + pivot_block.T)
            if self.jitter:
                pivot_block = pivot_block + self.jitter * total * np.eye(block_dim)

            try:
                chol = np.linalg.cholesky(pivot_block)
            except np.linalg.LinAlgError:
                # Exhausted rather than broken: this vertex has no variance
                # left that the factor does not already carry.
                used[vertex] = True
                residual_diag[coords] = 0.0
                continue

            new = sla.solve_triangular(chol, column.T, lower=True).T
            L[:, filled : filled + block_dim] = new
            filled += block_dim

            residual_diag = np.maximum(residual_diag - (new ** 2).sum(axis=1), 0.0)
            used[vertex] = True
            residual_diag[coords] = 0.0
            remaining = float(residual_diag.sum())
            pivots.append(vertex)

        if filled == 0:
            raise ValueError(
                f"pivoted_cholesky produced no columns: tolerance={self.tolerance} is "
                f"already satisfied by the untouched operator, or max_rank="
                f"{self.max_rank} is below block_dim={block_dim}."
            )

        L = L[:, :filled]
        left, singular, _ = np.linalg.svd(L, full_matrices=False)
        values, basis = sorted_eigenpairs(singular ** 2, left)

        return Decomposition(
            basis=basis,
            values=values,
            method="pivoted_cholesky",
            total_variance=total,
            residual_variance=float(residual_diag.sum()),
            info={
                "pivots": np.asarray(pivots, dtype=np.int64),
                "columns": filled,
                "relative_residual": float(residual_diag.sum() / total),
            },
        )


def pivoted_cholesky_factor(operator, max_rank: int = 300, tolerance: float = 1e-6):
    """``(L, pivots, relative_residual)`` without the SVD.

    The factor alone is enough to *sample* - ``u = L z`` has covariance
    ``L L^T`` - so when a model is only going to generate shapes, this skips the
    decomposition entirely. The eigenpairs are needed for anything that reasons
    per mode: truncation by variance share, the Mahalanobis distance, a slider
    per mode.
    """
    approximator = PivotedCholesky(max_rank=max_rank, tolerance=tolerance)
    result = approximator.decompose(operator)
    return result.factor(), result.info["pivots"], result.info["relative_residual"]
