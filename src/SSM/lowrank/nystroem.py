"""Inducing-point approximation - §Solving the Eigenvalue Problem, sampled.

Pick ``m`` of the ``n`` points, evaluate the kernel only against those, and
extend:

    K ~ K_nm K_mm^{-1} K_mn

The theory arrives here from the Nystrom quadrature of the integral eigenvalue
problem, which is the same construction read the other way round: approximating
``lambda psi(p) = integral k(p', p) psi(p') d mu(p')`` by a sum over ``m``
sample points *is* this factorisation.

Three reasons it earns its place beside ``pivoted_cholesky``, which is often
cheaper for a plain decomposition:

* **It comes with the extension formula.** :func:`nystrom_extend` evaluates the
  eigenfunctions at points that were never in the domain, which is what makes a
  GPMM resolution-independent. No other method here produces it.
* **It is cheap on a Hadamard product.** The ``m x m`` block of a product is
  the elementwise product of the two ``m x m`` blocks, so the corner of the
  algebra that costs ``163`` dense ``n x n`` products per matvec costs nothing
  extra here. On the baby model that is the difference between minutes and
  hours.
* **It is the same computation the hyperparameter bound needs.** The
  inducing-point variational bound uses exactly ``K_mm``, ``K_nm`` and
  ``diag(K_nn)``, so learning kernel parameters reuses this rather than
  duplicating it.

**Choosing m.** Cost is ``O(N m^2)`` in time and ``O(N m)`` in memory - on the
baby model at ``m = 1000`` vertices the cross block is 2.2 GB in float64, so it
is memory that binds first. Accuracy depends on how well ``m`` points cover the
surface at the kernel's length scale: roughly, ``m`` should be a few times
``area / sigma^2``. A short length scale needs many inducing points and is the
case where ``pivoted_cholesky`` wins instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from src.SSM.lowrank.operator import (
    Approximator,
    Decomposition,
    register_approximator,
    sorted_eigenpairs,
)

SELECTIONS = ("farthest", "random", "uniform")


def _unwrap(operator):
    """Reach the underlying operator through any weighting wrappers."""
    seen = operator
    while hasattr(seen, "operator"):
        seen = seen.operator
    return seen


def _points_of(operator) -> Optional[np.ndarray]:
    """The operator's own reference coordinates, if they can be reached."""
    inner = _unwrap(operator)
    domain = getattr(inner, "domain", None)
    vertices = getattr(inner, "vertices", None)
    if domain is None or vertices is None:
        return None
    return np.asarray(domain.points)[vertices]


def select_inducing(operator, n_inducing: int, selection: str = "farthest", seed=None) -> np.ndarray:
    """Vertex positions to use as inducing points, within the operator's own space.

    ``farthest``
        Farthest-point sampling, so the inducing set covers the surface rather
        than clustering where the mesh happens to be dense. This is the choice
        that matters: a random subset of a non-uniform mesh over-samples the
        detailed regions and leaves the smooth ones unsupported, and the
        approximation is visibly worse there.
    ``random`` / ``uniform``
        A random subset, and every ``n/m``-th vertex. Cheap, and reasonable on
        an already-uniform mesh. ``uniform`` is deterministic, which makes it
        useful for a reproducible test.
    """
    inner = _unwrap(operator)
    n = getattr(inner, "n_vertices", None) or operator.shape[0]
    m = int(min(n_inducing, n))

    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {list(SELECTIONS)}, got {selection!r}")

    if selection == "uniform":
        return np.unique(np.linspace(0, n - 1, m).astype(np.int64))

    if selection == "random":
        return np.sort(np.random.default_rng(seed).choice(n, size=m, replace=False))

    points = _points_of(operator)
    if points is None:
        raise ValueError(
            "selection='farthest' needs the operator's coordinates, which this operator "
            "does not expose. Pass inducing=<indices> explicitly, or use "
            "selection='random'/'uniform'."
        )

    from src.geometry.geometry import farthest_point_sampling

    # Returns a boolean mask rather than indices, and the caller wants indices.
    mask = farthest_point_sampling(np.asarray(points, dtype=np.float64), m)
    return np.flatnonzero(mask)


@register_approximator("nystroem")
@dataclass(frozen=True)
class Nystroem(Approximator):
    """Decomposition from ``m`` inducing points, and the extension that follows.

    ``jitter`` is relative to the trace of ``K_mm`` and is not optional in
    practice: the inducing block of a smooth kernel is severely
    ill-conditioned - a Gaussian at a long length scale has a numerically
    singular ``K_mm`` for any interesting ``m`` - and its inverse appears
    directly in the construction. ``rank_tolerance`` is the second guard,
    discarding inducing directions whose eigenvalue is below a fraction of the
    largest, which is what keeps ``K_mm^{-1/2}`` from amplifying rounding into
    the basis.
    """

    n_inducing: int = 512
    selection: str = "farthest"
    inducing: Optional[Sequence[int]] = None
    jitter: float = 1e-10
    rank_tolerance: float = 1e-12
    seed: Optional[int] = None
    needs_indexing = True

    def __post_init__(self):
        if self.n_inducing < 1:
            raise ValueError(f"n_inducing must be >= 1, got {self.n_inducing}")
        if self.jitter < 0:
            raise ValueError(f"jitter must be >= 0, got {self.jitter}")

    def inducing_indices(self, operator) -> np.ndarray:
        if self.inducing is not None:
            return np.ascontiguousarray(np.asarray(self.inducing, dtype=np.int64).reshape(-1))
        return select_inducing(operator, self.n_inducing, self.selection, self.seed)

    def decompose(self, operator) -> Decomposition:
        idx = self.inducing_indices(operator)

        K_mm = np.asarray(operator.block(idx, idx), dtype=np.float64)
        K_mm = 0.5 * (K_mm + K_mm.T)
        if self.jitter:
            K_mm = K_mm + self.jitter * float(np.trace(K_mm)) / max(K_mm.shape[0], 1) * np.eye(K_mm.shape[0])

        values_m, vectors_m = np.linalg.eigh(K_mm)
        keep = values_m > self.rank_tolerance * max(values_m.max(), 0.0)
        if not np.any(keep):
            raise np.linalg.LinAlgError(
                "the inducing block has no numerically positive eigenvalue - the kernel is "
                "zero on the inducing set, or jitter is too small."
            )
        whitener = vectors_m[:, keep] / np.sqrt(values_m[keep])[None, :]

        K_nm = np.asarray(operator.block(None, idx), dtype=np.float64)
        L = K_nm @ whitener                      # K ~ L L^T

        left, singular, _ = np.linalg.svd(L, full_matrices=False)
        values, basis = sorted_eigenpairs(singular ** 2, left)

        total = float(np.asarray(operator.diag()).sum())
        return Decomposition(
            basis=basis,
            values=values,
            method="nystroem",
            total_variance=total,
            residual_variance=max(total - float(values.sum()), 0.0),
            info={"n_inducing": int(idx.size), "inducing": idx, "selection": self.selection},
        )


def nystrom_extend(cross: np.ndarray, basis: np.ndarray, values: np.ndarray,
                   rank_tolerance: float = 1e-12) -> np.ndarray:
    """Eigenfunctions at new points: ``psi_i(x*) = k(x*)^T psi_i / lambda_i``.

    §Practical notes on continuous inference. ``cross`` is
    ``k(X_new, domain)`` with shape ``(M, N)``, ``basis``/``values`` the
    decomposition on the domain; the result is ``(M, L)``, the basis evaluated
    at the new points.

    Note what this needs and what it therefore cannot do. ``cross`` requires
    ``k`` at coordinates that were never sampled, so it exists only for a
    kernel with an analytic form - a ``pca`` or ``laplace_beltrami`` term
    supplies no such value, which is exactly the gap
    ``src/SSM/model/extension.py`` closes by regressing onto the analytic part
    instead. This function is the linear algebra; the decision about *which*
    kernel to extend with is not made here.

    Modes with eigenvalues at the level of the solver's own error are set to
    zero rather than divided through: ``1/lambda`` for a mode that is numerical
    noise turns that noise into the dominant feature of the extension.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.asarray(cross, dtype=np.float64) @ np.asarray(basis, dtype=np.float64)

    safe = values > rank_tolerance * max(values.max(initial=0.0), 0.0)
    scaled = np.zeros_like(out)
    scaled[:, safe] = out[:, safe] / values[safe][None, :]
    return scaled
