"""What the eigensolvers see, and the interface they are selected through.

The decomposition layer never touches a :class:`~src.SSM.kernels.base.Kernel`.
It works against an *operator* - something with a shape, a matrix-vector
product, and a declared structure - which buys two things:

* the same four algorithms decompose any positive semi-definite operator, a
  hand-built matrix in a test as readily as a ``pca * gaussian`` kernel on
  31027 vertices;
* nothing here has to know that ``K`` is 69 GB on the baby model and must never
  be formed. :class:`KernelOperator` answers ``matmat`` by chunking, and the
  algorithms are written to ask only for that.

Two protocols rather than one, because the algorithms genuinely need different
things. :class:`LinearOperator` (``matmat``, ``shape``, ``structure``) is enough
for ``randomized``. :class:`IndexableOperator` adds ``block`` and ``diag``,
which ``nystroem`` and ``pivoted_cholesky`` require because they select
*columns* rather than projecting onto random directions. Declaring that split
means an operator that can only do matvecs still works with what it can, and
fails clearly rather than mysteriously with the rest.

**The mass matrix.** §Solving the Eigenvalue Problem discretises the integral
operator assuming points of uniform density, ``d mu = 1/n``, giving
``(1/n) K psi = lambda psi``. Mesh vertices are not uniform - the baby model's
31027 of them bunch wherever the original remeshing put detail - and under that
assumption every eigenfunction is biased toward the dense regions. The correct
discretisation carries the mass matrix ``M`` of barycentric vertex areas:

    K M psi = lambda psi ,    psi^T M psi = 1

which :func:`mass_weighted` implements by symmetrising to ``M^{1/2} K M^{1/2}``
- an ordinary symmetric eigenproblem, so every algorithm here is unchanged -
and :func:`to_mass_orthonormal` undoes on the way out. It reduces to the
theory's form exactly when ``M = (1/n) I``, so this is a generalisation rather
than a disagreement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from src.core.registry import TypedRegistry
from src.SSM.kernels.base import KernelStructure, expand_indices, resolve_indices
from src.SSM.lowrank.truncation import KEEP_ALL, Decomposition, Truncation

APPROXIMATORS = TypedRegistry("approximator")
register_approximator = APPROXIMATORS.register


def available_approximators() -> List[str]:
    return APPROXIMATORS.names()


def build_approximator(name: str, kwargs: Optional[Dict[str, Any]] = None) -> "Approximator":
    return APPROXIMATORS.build(name, kwargs)


# --------------------------------------------------------------------------
# protocols
# --------------------------------------------------------------------------

class LinearOperator(Protocol):
    """A symmetric positive semi-definite operator, applied but never formed."""

    @property
    def shape(self) -> tuple: ...

    @property
    def structure(self) -> KernelStructure: ...

    def matmat(self, V: np.ndarray) -> np.ndarray: ...


class IndexableOperator(LinearOperator, Protocol):
    """Additionally addressable by index - what column-selecting methods need."""

    def block(self, rows=None, cols=None) -> np.ndarray: ...

    def diag(self) -> np.ndarray: ...


# --------------------------------------------------------------------------
# concrete operators
# --------------------------------------------------------------------------

class KernelOperator:
    """A kernel evaluated on a domain, as an operator.

    ``vertices`` optionally restricts the operator to a subset - which is how a
    fit is run at a decimated resolution and reconstructed at full resolution,
    and how the inducing-point methods address their sample. Row and column
    indices passed to :meth:`block` are positions *within* that subset, so a
    caller never has to hold two index spaces at once.
    """

    def __init__(self, kernel, domain, vertices=None):
        self.kernel = kernel
        self.domain = domain
        self.vertices = resolve_indices(domain, vertices)
        self.block_dim = kernel.block_dim

    @property
    def n_vertices(self) -> int:
        return self.vertices.size

    @property
    def shape(self) -> tuple:
        size = self.block_dim * self.vertices.size
        return (size, size)

    @property
    def structure(self) -> KernelStructure:
        return self.kernel.structure

    def _select(self, index):
        return self.vertices if index is None else self.vertices[np.asarray(index, dtype=np.int64)]

    def matmat(self, V: np.ndarray) -> np.ndarray:
        return self.kernel.matmat(self.domain, V, self.vertices, self.vertices)

    def block(self, rows=None, cols=None) -> np.ndarray:
        return self.kernel.block(self.domain, self._select(rows), self._select(cols))

    def diag(self) -> np.ndarray:
        return self.kernel.diag(self.domain, self.vertices)

    def factor(self) -> Optional[np.ndarray]:
        """The kernel's own exact factor, when it has one.

        Worth checking before running any algorithm here: ``pca`` and
        ``laplace_beltrami`` arrive already decomposed, and re-deriving what
        they carry would be both slower and less accurate.
        """
        return self.kernel.factor(self.domain, self.vertices)

    def to_dense(self) -> np.ndarray:
        return self.kernel.dense(self.domain, self.vertices, self.vertices)

    def __repr__(self) -> str:
        return f"KernelOperator({self.kernel.describe()}, n={self.vertices.size})"


class DenseOperator:
    """A matrix already in memory. Reference implementation and test fixture."""

    def __init__(self, matrix, structure: Optional[KernelStructure] = None):
        from src.SSM.kernels.base import DENSE

        self.matrix = np.ascontiguousarray(np.asarray(matrix, dtype=np.float64))
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"expected a square matrix, got shape {self.matrix.shape}")
        self._structure = structure or DENSE
        self.block_dim = 1

    @property
    def shape(self) -> tuple:
        return self.matrix.shape

    @property
    def structure(self) -> KernelStructure:
        return self._structure

    def matmat(self, V):
        return self.matrix @ np.asarray(V)

    def block(self, rows=None, cols=None):
        n = self.matrix.shape[0]
        rows = np.arange(n) if rows is None else np.asarray(rows, dtype=np.int64)
        cols = np.arange(n) if cols is None else np.asarray(cols, dtype=np.int64)
        return self.matrix[np.ix_(rows, cols)]

    def diag(self):
        return np.diag(self.matrix).copy()

    def factor(self):
        return None

    def to_dense(self):
        return self.matrix


class MassWeighted:
    """``M^{1/2} A M^{1/2}`` - the symmetrised generalised problem.

    ``weights`` are ``sqrt(mass)`` already expanded to coordinate space; build
    them with :func:`mass_weights` rather than by hand, since the vertex-major
    expansion is easy to get subtly wrong and the symptom is a basis that is
    almost, but not quite, orthonormal.
    """

    def __init__(self, operator, weights):
        self.operator = operator
        self.weights = np.ascontiguousarray(np.asarray(weights, dtype=np.float64).reshape(-1))
        if self.weights.size != operator.shape[0]:
            raise ValueError(
                f"{self.weights.size} weights for an operator of size {operator.shape[0]}"
            )
        self.block_dim = getattr(operator, "block_dim", 1)

    @property
    def shape(self) -> tuple:
        return self.operator.shape

    @property
    def structure(self) -> KernelStructure:
        # A diagonal congruence changes neither rank nor sparsity pattern.
        return self.operator.structure

    def _w(self, index=None):
        if index is None:
            return self.weights
        return self.weights[expand_indices(np.asarray(index, dtype=np.int64), self.block_dim)]

    def matmat(self, V):
        V = np.asarray(V)
        scaled = self.weights[:, None] * V if V.ndim == 2 else self.weights * V
        out = np.asarray(self.operator.matmat(scaled))
        return self.weights[:, None] * out if out.ndim == 2 else self.weights * out

    def block(self, rows=None, cols=None):
        inner = self.operator.block(rows, cols)
        return self._w(rows)[:, None] * inner * self._w(cols)[None, :]

    def diag(self):
        return (self.weights ** 2) * np.asarray(self.operator.diag())

    def factor(self):
        F = self.operator.factor() if hasattr(self.operator, "factor") else None
        return None if F is None else self.weights[:, None] * F

    def to_dense(self):
        A = self.operator.to_dense()
        return self.weights[:, None] * A * self.weights[None, :]


def mass_weights(mass, block_dim: int = 1) -> np.ndarray:
    """``sqrt(mass)``, expanded vertex-major to coordinate space.

    ``mass`` is the per-vertex quadrature measure - barycentric vertex areas,
    normalised to sum to one. A domain that does not supply one is being
    treated as uniformly sampled, which is the theory's ``d mu = 1/n``.
    """
    m = np.asarray(mass, dtype=np.float64).reshape(-1)
    if np.any(m < 0):
        raise ValueError("mass must be non-negative")
    w = np.sqrt(m)
    return np.repeat(w, block_dim) if block_dim > 1 else w


def mass_weighted(operator, mass=None, block_dim: Optional[int] = None):
    """Wraps ``operator`` for the generalised problem; a no-op when ``mass`` is None."""
    if mass is None:
        return operator
    dim = getattr(operator, "block_dim", 1) if block_dim is None else block_dim
    return MassWeighted(operator, mass_weights(mass, dim))


def to_mass_orthonormal(decomposition: Decomposition, weights) -> Decomposition:
    """Undoes the symmetrisation: ``Psi = M^{-1/2} U``, so ``Psi^T M Psi = I``.

    Zero-mass vertices would divide by zero. They mean a degenerate triangle
    fan rather than anything meaningful, so their basis rows are set to zero -
    a vertex with no area contributes nothing to the quadrature and should
    contribute nothing to the mode.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    safe = w > 0
    basis = np.zeros_like(decomposition.basis)
    basis[safe] = decomposition.basis[safe] / w[safe][:, None]
    decomposition.basis = basis
    decomposition.inner_product = "mass"
    return decomposition


# --------------------------------------------------------------------------
# the approximator interface
# --------------------------------------------------------------------------

class Approximator(ABC):
    """One way of extracting ``(Psi, Lambda)`` from a PSD operator."""

    name = "?"
    #: Whether this method needs ``block``/``diag`` as well as ``matmat``.
    needs_indexing = False

    @abstractmethod
    def decompose(self, operator) -> Decomposition:
        """Leading eigenpairs, ordered by descending variance."""

    def describe(self) -> str:
        return repr(self)


def _check_operator(operator, approximator: Approximator) -> None:
    if approximator.needs_indexing and not hasattr(operator, "block"):
        raise TypeError(
            f"approximator '{approximator.name}' selects kernel columns and needs an "
            f"operator with block(); {type(operator).__name__} provides only matmat(). "
            f"Use 'randomized', which does not."
        )


def sorted_eigenpairs(values: np.ndarray, vectors: np.ndarray, clip: bool = True):
    """Descending order, with negative eigenvalues clipped to zero.

    A PSD operator has no negative eigenvalues; the ones a solver returns are
    rounding at the level of its own tolerance. Left alone they propagate into
    ``sqrt(Lambda)`` as nans, which surface far from here as a model that
    samples garbage.
    """
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    if clip:
        values = np.maximum(values, 0.0)
    return np.ascontiguousarray(values), np.ascontiguousarray(vectors)


def decompose(
    operator,
    approximator="dense_eigh",
    kwargs: Optional[Dict[str, Any]] = None,
    truncation: Truncation = KEEP_ALL,
    mass=None,
) -> Decomposition:
    """Build a decomposition end to end: mass, method, truncation.

    The single entry point the model builder uses, so the mass symmetrisation
    and its inverse happen in exactly one place and cannot be half-applied.
    """
    if isinstance(approximator, str):
        approximator = build_approximator(approximator, kwargs)

    weighted = mass_weighted(operator, mass)
    _check_operator(weighted, approximator)

    result = approximator.decompose(weighted)
    if mass is not None:
        result = to_mass_orthonormal(result, weighted.weights)

    return truncation.apply(result)
