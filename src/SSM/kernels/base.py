"""The kernel interface every covariance in a GPMM is built from.

Implements the covariance side of §Gaussian Process Morphable Models in
``theory/shape_deformation_models.tex``: a shape is ``f(p) = u_bar(p) + u(p)``
with ``u ~ GP(0, k)``, and evaluating ``k`` at ``n`` points gives the block
matrix ``K in R^{3n x 3n}`` of Eq. ``GP_deformation_dist``. Everything in this
package exists to let that ``k`` be *written down* - as a data kernel, a
parameterised kernel, or any sum/product of them - rather than only estimated
from a population.

Three facts shape the interface.

**The matrix cannot be formed.** On the baby model ``3n = 93081``, so ``K`` in
float64 is 69 GB. ``matmat`` is therefore the primitive and ``block``/``dense``
exist for small domains and tests. The default ``matmat`` here chunks over row
blocks against a memory budget, so a subclass gets a working matrix-free
product from implementing ``block`` alone, and overrides it only when its
structure allows something better (a low-rank kernel applies its factor twice
and never touches a dense block at all).

**Structure has to survive composition.** A sum of two low-rank kernels is
low-rank; a Hadamard product with a compactly supported kernel is sparse. The
low-rank layer picks its algorithm from :attr:`Kernel.structure`, so the
combinators in ``algebra.py`` propagate it rather than collapsing everything to
dense. :func:`sum_structure` and :func:`product_structure` hold that lattice in
one place. Both are deliberately conservative: an unrecognised combination
degrades to ``DENSE``, which costs at worst a suboptimal choice of eigensolver -
never a wrong answer, since ``matmat`` remains exact whatever the label says.

**Not every kernel can be evaluated off the sample points.** A parameterised
kernel is a formula and can be evaluated anywhere. The PCA kernel is a tabulated
second moment: ``k_PCA(x*, x') = (1/(N-1)) sum_i u^i(x*) u^i(x')^T`` needs each
subject's deformation *at the new point*, and the subjects are known only at the
mesh vertices. That asymmetry is a property of empirical covariance, not a gap
to paper over, so it is in the type: :meth:`Kernel.cross` evaluates against
arbitrary coordinates and is optional, :meth:`Kernel.analytic_part` returns the
sub-kernel for which it is defined, and the continuous extension of §Practical
notes builds on that part alone (see ``src/SSM/model/extension.py``). A kernel
whose ``analytic_part`` is ``None`` is perfectly usable - it simply cannot be
sampled off its own vertices without an explicit interpolation strategy.

Conventions, fixed here and relied on everywhere else:

* ``block_dim`` is 1 for a scalar kernel ``k(p,p') in R`` and 3 for a
  matrix-valued one ``k(p,p') in R^{3x3}``. A scalar kernel represents
  ``k(p,p') * I_3`` when it meets a matrix-valued one - but see
  ``algebra.Product``, where that lift would be *wrong* and block scaling is
  meant instead.
* ``rows`` and ``cols`` are integer arrays of **vertex** indices into the
  domain, or ``None`` for "all of them".
* Coordinates are vertex-major: entry ``3*i + c`` is component ``c`` of vertex
  ``i``. This matches ``mean.reshape(-1, 3)`` and
  ``Phi.reshape(n, 3, K)`` as used throughout ``src/SSM``.
"""

from __future__ import annotations

import inspect
import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from src.core.registry import TypedRegistry

KERNELS = TypedRegistry("kernel")
register_kernel = KERNELS.register

#: Working set a chunked ``matmat`` may allocate for one row block, in bytes.
#: 256 MB keeps the baby model's 31027-column blocks at ~1000 rows a chunk.
DEFAULT_CHUNK_BYTES = 256 * 1024 ** 2


def available_kernels() -> List[str]:
    return KERNELS.names()


def build_kernel(spec) -> "Kernel":
    """Builds a kernel from a nested spec, recursing into ``children``.

    ``spec`` is a mapping (or any object carrying the same attributes) of the
    form ``{"name": str, "kwargs": {...}, "children": [spec, ...]}``. Children
    are built first and passed to the parent as its ``children`` argument, which
    is what makes the composite in ``algebra.py`` configurable from YAML without
    the schema knowing any kernel's name.

    Unknown keyword arguments are rejected by the registry against the class's
    own signature, so a typo fails here rather than producing a kernel with
    silent defaults.
    """
    if isinstance(spec, Kernel):
        return spec

    get = spec.get if hasattr(spec, "get") else lambda key, default=None: getattr(spec, key, default)

    name = get("name")
    if not name:
        raise ValueError(f"kernel spec has no 'name': {spec!r}")

    kwargs = dict(get("kwargs") or {})
    children = [build_kernel(child) for child in (get("children") or [])]

    if children:
        # The combinators take their operands under two different names: the
        # n-ary ones (sum, product) as `children`, the unary ones (scale,
        # localized, symmetry, isotropic_lift) as `child`. Reading the
        # constructor rather than hard-coding the split keeps a new combinator
        # from having to be added here as well.
        accepted = inspect.signature(KERNELS.get(name)).parameters
        if "children" in accepted:
            kwargs["children"] = children
        elif "child" in accepted:
            if len(children) != 1:
                raise ValueError(
                    f"kernel '{name}' wraps a single child, got {len(children)}"
                )
            kwargs["child"] = children[0]
        else:
            raise ValueError(f"kernel '{name}' takes no child kernels")

    return KERNELS.build(name, kwargs)


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------

class Structure(Enum):
    """How a kernel matrix is shaped, which decides how it is decomposed."""

    DENSE = "dense"
    LOWRANK = "lowrank"
    SPARSE = "sparse"
    DIAGONAL = "diagonal"


class KernelStructure:
    """``Structure`` plus the rank, when there is one."""

    __slots__ = ("kind", "rank")

    def __init__(self, kind: Structure, rank: Optional[int] = None):
        if kind is Structure.LOWRANK and rank is None:
            raise ValueError("a LOWRANK structure must state its rank")
        self.kind = kind
        self.rank = None if rank is None else int(rank)

    @property
    def is_lowrank(self) -> bool:
        return self.kind is Structure.LOWRANK

    def __eq__(self, other):
        return (
            isinstance(other, KernelStructure)
            and self.kind is other.kind
            and self.rank == other.rank
        )

    def __hash__(self):
        return hash((self.kind, self.rank))

    def __repr__(self):
        return f"{self.kind.value}" + (f"(r={self.rank})" if self.rank is not None else "")


DENSE = KernelStructure(Structure.DENSE)
SPARSE = KernelStructure(Structure.SPARSE)
DIAGONAL = KernelStructure(Structure.DIAGONAL)


def lowrank(rank: int) -> KernelStructure:
    return KernelStructure(Structure.LOWRANK, rank)


def sum_structure(a: KernelStructure, b: KernelStructure) -> KernelStructure:
    """Structure of ``A + B``.

    Exact where the sum genuinely preserves a form, ``DENSE`` otherwise. The
    conservative cases are real: ``diagonal + lowrank`` is a
    diagonal-plus-low-rank matrix, which is cheap to apply but is not itself
    low-rank, and calling it low-rank would let the decomposition truncate away
    the diagonal part.
    """
    if a.kind is b.kind:
        if a.is_lowrank:
            return lowrank(a.rank + b.rank)
        return KernelStructure(a.kind)
    pair = {a.kind, b.kind}
    if pair == {Structure.DIAGONAL, Structure.SPARSE}:
        return SPARSE
    return DENSE


def product_structure(a: KernelStructure, b: KernelStructure) -> KernelStructure:
    """Structure of the Hadamard product ``A * B``.

    A zero in either factor is a zero in the product, which is what makes
    ``diagonal`` and ``sparse`` absorbing. ``lowrank * lowrank`` is the
    Khatri-Rao case: exact, at rank ``r_a * r_b``, which grows fast enough that
    ``algebra.Product`` guards it.
    """
    if a.kind is Structure.DIAGONAL or b.kind is Structure.DIAGONAL:
        return DIAGONAL
    if a.kind is Structure.SPARSE or b.kind is Structure.SPARSE:
        return SPARSE
    if a.is_lowrank and b.is_lowrank:
        return lowrank(a.rank * b.rank)
    return DENSE


# --------------------------------------------------------------------------
# domain
# --------------------------------------------------------------------------

class KernelDomain(Protocol):
    """What a kernel needs from the domain it is evaluated on.

    Deliberately smaller than ``src.SSM.model.domain.ReferenceDomain``, which
    also carries the quadrature measure, the landmarks and the triangulation:
    none of those belong to kernel evaluation, and keeping them out means the
    kernels can be exercised against any object with ``points``.
    """

    @property
    def points(self) -> np.ndarray:
        """(n, 3) reference coordinates - the mean shape, never a deformation."""


def domain_size(domain) -> int:
    n = getattr(domain, "n", None)
    return int(n) if n is not None else int(np.asarray(domain.points).shape[0])


def resolve_indices(domain, index) -> np.ndarray:
    """``None`` -> every vertex; anything else -> a contiguous int array."""
    if index is None:
        return np.arange(domain_size(domain), dtype=np.int64)
    return np.ascontiguousarray(np.asarray(index, dtype=np.int64).reshape(-1))


def expand_indices(index: np.ndarray, block_dim: int) -> np.ndarray:
    """Vertex indices -> coordinate indices, vertex-major.

    ``[2, 5]`` at ``block_dim=3`` becomes ``[6, 7, 8, 15, 16, 17]`` - the row
    layout of ``mean``, ``Phi`` and every other ``(3N, ...)`` array in this
    project.
    """
    if block_dim == 1:
        return index
    return (block_dim * index[:, None] + np.arange(block_dim)[None, :]).reshape(-1)


def pairwise_distances(domain, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """(|rows|, |cols|) distances, using the domain's own metric when it has one.

    A ``ReferenceDomain`` answers this itself, so a geodesic domain returns
    geodesic distances and every stationary kernel becomes an intrinsic one with
    no change. A bare object carrying only ``points`` falls back to Euclidean,
    which is what makes the kernels testable without building a mesh.
    """
    method = getattr(domain, "pairwise_distances", None)
    if method is not None:
        return np.asarray(method(rows, cols))

    points = np.asarray(domain.points)
    return _euclidean(points[rows], points[cols], same=_same_indices(rows, cols))


def cross_distances(domain, X: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """(|X|, |cols|) distances from arbitrary coordinates to domain vertices."""
    method = getattr(domain, "cross_distances", None)
    if method is not None:
        return np.asarray(method(X, cols))

    points = np.asarray(domain.points)
    return _euclidean(np.asarray(X, dtype=np.float64).reshape(-1, 3), points[cols])


def _euclidean(A: np.ndarray, B: np.ndarray, same: bool = False) -> np.ndarray:
    """Pairwise Euclidean distance via the expanded square.

    The expanded form ``|a|^2 + |b|^2 - 2 a.b`` is the only one that fits in
    memory at these sizes - the direct form needs an ``(m, n, 3)`` intermediate,
    three times the result - but it cancels, and two consequences follow.

    It can go slightly negative for coincident points, and ``sqrt`` of that is a
    nan that travels silently into the eigendecomposition. Clipping at zero is
    the standard guard.

    Worse, the error is *absolute* rather than relative: cancellation leaves
    ~1e-16 in the squared distance, so a true distance of zero comes back as
    ~1e-8. On a self-distance that is not a rounding detail, it is a wrong
    diagonal - and since ``Stationary.diag`` returns ``radial(0)`` exactly,
    ``block`` and ``diag`` would then disagree about the same entries, which
    pivoted Cholesky reads alternately. ``same=True`` sets the diagonal to zero
    exactly, which is what every caller comparing an index set with itself
    wants. Distinct-but-near-coincident points keep the ~1e-8 floor; that is
    below any length scale a kernel is worth evaluating at, and getting rid of
    it would cost the memory the expanded form exists to save.
    """
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * (A @ B.T)
    if same:
        np.fill_diagonal(sq, 0.0)
    return np.sqrt(np.maximum(sq, 0.0))


def _same_indices(rows: np.ndarray, cols: np.ndarray) -> bool:
    """True when the two index sets are the same, so the diagonal is a self-distance."""
    return rows is cols or (rows.shape == cols.shape and np.array_equal(rows, cols))


# --------------------------------------------------------------------------
# the interface
# --------------------------------------------------------------------------

class Kernel(ABC):
    """A covariance function, evaluable on a domain and composable with others.

    Subclasses must supply :meth:`block` and :attr:`structure`. Everything else
    has a working default: :meth:`matmat` chunks over ``block``, :meth:`diag`
    reads its diagonal, :meth:`dense` is ``block`` with a size guard. Override
    them where structure allows better - a low-rank kernel should override
    :meth:`matmat`, :meth:`factor` and :meth:`diag`, and never build a block.

    Instances are treated as immutable value objects: :meth:`with_parameters`
    returns a new kernel rather than mutating. That is what makes a kernel
    hashable as part of a build-cache key.
    """

    name = "?"
    block_dim = 1

    # -- structure ---------------------------------------------------------

    @property
    @abstractmethod
    def structure(self) -> KernelStructure:
        """How this kernel's matrix is shaped; see :func:`sum_structure`."""

    # -- evaluation --------------------------------------------------------

    @abstractmethod
    def block(self, domain, rows=None, cols=None) -> np.ndarray:
        """Dense ``(block_dim*|rows|, block_dim*|cols|)`` block.

        Implementations evaluate exactly what is asked for and do not chunk -
        chunking is the caller's business, and :meth:`matmat` already does it.
        """

    def matmat(self, domain, V, rows=None, cols=None, chunk_bytes: Optional[int] = None):
        """``K(rows, cols) @ V``, without ever holding ``K``.

        ``V`` is ``(block_dim*|cols|, k)``; the result is
        ``(block_dim*|rows|, k)``. The default walks row blocks sized to
        ``chunk_bytes``, which is the whole reason a 93081-square kernel is
        usable at all.
        """
        rows = resolve_indices(domain, rows)
        cols = resolve_indices(domain, cols)
        V = np.asarray(V)
        if V.ndim == 1:
            V = V[:, None]
            squeeze = True
        else:
            squeeze = False

        expected = self.block_dim * cols.size
        if V.shape[0] != expected:
            raise ValueError(
                f"{self.name}: V has {V.shape[0]} rows, expected {expected} "
                f"({self.block_dim} x {cols.size} columns of K)"
            )

        budget = DEFAULT_CHUNK_BYTES if chunk_bytes is None else int(chunk_bytes)
        per_row = max(1, self.block_dim * cols.size * 8)
        step = max(1, min(rows.size, budget // per_row // max(1, self.block_dim)))

        out = np.empty((self.block_dim * rows.size, V.shape[1]), dtype=np.float64)
        for start in range(0, rows.size, step):
            stop = min(start + step, rows.size)
            piece = self.block(domain, rows[start:stop], cols)
            out[self.block_dim * start : self.block_dim * stop] = piece @ V

        return out[:, 0] if squeeze else out

    def diag(self, domain, rows=None) -> np.ndarray:
        """``(block_dim*|rows|,)`` diagonal of ``K``.

        Needed on its own by the trace term of the inducing-point bound in
        ``src/SSM/hyper``, and by the pivoted-Cholesky pivot search - both of
        which must never touch a full block.
        """
        rows = resolve_indices(domain, rows)
        out = np.empty(self.block_dim * rows.size, dtype=np.float64)
        step = max(1, DEFAULT_CHUNK_BYTES // max(1, self.block_dim ** 2 * 8) // 64)
        for start in range(0, rows.size, step):
            stop = min(start + step, rows.size)
            piece = rows[start:stop]
            block = self.block(domain, piece, piece)
            out[self.block_dim * start : self.block_dim * stop] = np.diag(block)
        return out

    def factor(self, domain, rows=None) -> Optional[np.ndarray]:
        """``(block_dim*|rows|, r)`` with ``K = F F^T``, or ``None``.

        Only an exactly low-rank kernel returns one. When it does, the low-rank
        layer skips the eigensolver entirely: the factor *is* the decomposition,
        up to an orthogonalisation.
        """
        return None

    def dense(self, domain, rows=None, cols=None, max_entries: int = 50_000_000) -> np.ndarray:
        """The full block, with a guard against forming something enormous.

        For tests, small domains and the reference eigensolver. The guard is not
        paranoia: ``dense`` on the baby model asks for 69 GB, and the failure
        without it is an unexplained kill rather than a message.
        """
        rows = resolve_indices(domain, rows)
        cols = resolve_indices(domain, cols)
        entries = (self.block_dim ** 2) * rows.size * cols.size
        if entries > max_entries:
            raise MemoryError(
                f"{self.name}.dense() would allocate {entries:,} entries "
                f"({entries * 8 / 1024 ** 3:.1f} GB). Use matmat(), or raise "
                f"max_entries if this is deliberate."
            )
        return self.block(domain, rows, cols)

    # -- evaluation off the domain ----------------------------------------

    def supports_cross(self) -> bool:
        """True when :meth:`cross` is defined, i.e. the kernel is a formula."""
        return False

    def cross(self, domain, X, cols=None) -> np.ndarray:
        """``k(X, domain[cols])`` for arbitrary coordinates ``X``.

        Defined only for kernels with an analytic form. The tabulated ones -
        ``pca``, ``laplace_beltrami`` - raise, which is the honest answer: their
        value at a new point is not determined by what they store.
        """
        raise NotImplementedError(
            f"kernel '{self.name}' is tabulated on its domain and has no value at "
            f"arbitrary coordinates. Extend it with the analytic part of the model "
            f"kernel (see src/SSM/model/extension.py), or pick a kernel that has one."
        )

    def analytic_part(self) -> Optional["Kernel"]:
        """The sub-kernel evaluable anywhere, or ``None``.

        This is the seam the continuous formulation of §Practical notes hangs
        on. ``Sum(pca, gaussian).analytic_part()`` is ``gaussian``: the
        extension operator regresses onto *that*, which is what keeps it
        non-circular - regressing onto a kernel that contains ``pca`` would
        require the very values being extended.
        """
        return self if self.supports_cross() else None

    # -- parameters --------------------------------------------------------

    def free_parameters(self) -> Dict[str, float]:
        """Unconstrained (log-space) parameters, for search or gradients.

        Log space because every parameter here is a positive scale: it keeps an
        optimiser from proposing a negative length scale, and it matches the
        multiplicative way these quantities actually behave.
        """
        return {}

    def with_parameters(self, theta: Dict[str, float]) -> "Kernel":
        """A new kernel with ``theta`` applied. Never mutates."""
        return self

    # -- composition -------------------------------------------------------

    def __add__(self, other) -> "Kernel":
        from src.SSM.kernels.algebra import Sum

        return Sum([self, _as_kernel(other)])

    __radd__ = __add__

    def __mul__(self, other) -> "Kernel":
        from src.SSM.kernels.algebra import Product, Scale

        if np.isscalar(other):
            return Scale(self, float(other))
        return Product([self, _as_kernel(other)])

    __rmul__ = __mul__

    # -- reporting ---------------------------------------------------------

    def describe(self) -> str:
        # free_parameters() is log-space, which is right for an optimiser and
        # wrong for a log line - a sigma of 1.5 reading as 0.405 is worse than
        # no report at all. Undo it for display only.
        params = ", ".join(
            f"{key[4:] if key.startswith('log_') else key}="
            f"{(math.exp(value) if key.startswith('log_') else value):.4g}"
            for key, value in sorted(self.free_parameters().items())
        )
        return f"{self.name}({params}) [{self.structure}]"

    def __repr__(self) -> str:
        return self.describe()


def _as_kernel(value) -> Kernel:
    if isinstance(value, Kernel):
        return value
    raise TypeError(f"cannot combine a Kernel with {type(value).__name__}")


# --------------------------------------------------------------------------
# helpers shared by the concrete kernels
# --------------------------------------------------------------------------

def log_params(**values) -> Dict[str, float]:
    """``{"sigma": 3.0} -> {"log_sigma": 1.0986}``, skipping non-positive values."""
    out = {}
    for key, value in values.items():
        if value is not None and value > 0:
            out[f"log_{key}"] = math.log(float(value))
    return out


def read_log(theta: Dict[str, float], key: str, current: float) -> float:
    """``theta`` may carry only some parameters; the rest keep their value."""
    name = f"log_{key}"
    return math.exp(float(theta[name])) if name in theta else current
