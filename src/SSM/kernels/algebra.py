"""Combinators - the part that makes a GPMM more than a covariance.

§Gaussian Process Morphable Models builds its total covariance as

    k(p, p') = k_PCA(p, p') + k_smooth(p, p')      or      k_PCA * k_smooth

and the theory notes the third variant explicitly: when ``k_smooth`` is scalar
the product is the **block-scaled** ``K_ij = k_smooth(p_i, p_j) * K_PCA,ij``,
not a Hadamard product against a lifted ``k_smooth I_3``. Those are different
matrices - the lift zeroes the off-diagonal entries inside every 3x3 block, and
using it would quietly destroy the directional structure the data kernel exists
to carry. :class:`Product` distinguishes the two cases; see its docstring.

Each combinator is itself a :class:`~src.SSM.kernels.base.Kernel`, so an
expression nests to any depth and needs no special evaluator:

    pca + 0.3 * Gaussian(sigma=15)
    Localized(Matern52(rho=8), weights) + White(1e-8)
    Symmetry(pca * Gaussian(sigma=20), mirror)

What the two forms buy is different, and worth choosing deliberately:

**Sum** *adds* directions. The 163 modes of a PCA model span a fixed subspace;
anything outside it is unreachable, however implausible the target. Adding a
smooth kernel makes the model able to reach shapes the population never showed,
and incidentally makes ``K`` non-singular so a posterior stops being degenerate.

**Product** *localises*. It decorrelates distant vertices while keeping the
population's local correlation structure, which is what breaks the "every mode
is global" property of PCA. That is the useful one for fitting partial data, and
it is also the expensive one - see the rank note on :class:`Product`.

The rank consequence is worth stating once, because it surprises people: a
smooth kernel has numerical rank of order ``area / sigma^2``, so a product with
it multiplies the number of modes rather than adding to it. Localising a
163-mode model at a tenth of its extent gives an object of rank ~10^4 before
truncation. That is the freedom being bought, not a defect - but the truncation
level, the regularisation weight and the chi-square bound all mean different
things afterwards.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.SSM.kernels.base import (
    Kernel,
    KernelStructure,
    Structure,
    expand_indices,
    lowrank,
    product_structure,
    register_kernel,
    resolve_indices,
    sum_structure,
)

#: Refuse to build a Khatri-Rao factor past this many columns. A product of two
#: rank-200 kernels is rank 40000, which is larger than the matrix it factors on
#: any domain worth using it on; past this the generic chunked path is cheaper.
MAX_EXACT_PRODUCT_RANK = 20_000


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _namespaced(children: Sequence[Kernel]) -> Dict[str, float]:
    """Child parameters, prefixed by position: ``0.log_sigma``, ``1.log_scale``."""
    out: Dict[str, float] = {}
    for index, child in enumerate(children):
        for key, value in child.free_parameters().items():
            out[f"{index}.{key}"] = value
    return out


def _split_namespaced(theta: Dict[str, float], count: int) -> List[Dict[str, float]]:
    """Inverse of :func:`_namespaced`; unprefixed keys are ignored."""
    parts: List[Dict[str, float]] = [{} for _ in range(count)]
    for key, value in theta.items():
        head, _, rest = key.partition(".")
        if not rest or not head.isdigit():
            continue
        index = int(head)
        if 0 <= index < count:
            parts[index][rest] = value
    return parts


def _repeat_blocks(S: np.ndarray, factor: int = 3) -> np.ndarray:
    """``S[i,j]`` -> a ``factor x factor`` constant block, for block scaling."""
    return np.repeat(np.repeat(S, factor, axis=0), factor, axis=1)


def _row_khatri_rao(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Khatri-Rao: ``out[i, p*rb + q] = A[i,p] * B[i,q]``.

    Why this is the factor of a Hadamard product: with ``A = F_a F_a^T`` and
    ``B = F_b F_b^T``,

        (A * B)_ij = (sum_p F_a[i,p] F_a[j,p]) (sum_q F_b[i,q] F_b[j,q])
                   = sum_{p,q} (F_a[i,p] F_b[i,q]) (F_a[j,p] F_b[j,q])

    so the products of the entries, taken row by row, are exactly the factor of
    the product. Exact, not an approximation - only large.
    """
    return (A[:, :, None] * B[:, None, :]).reshape(A.shape[0], -1)


def _lift_if_needed(children: Sequence[Kernel]) -> List[Kernel]:
    """Wrap scalar children in :class:`IsotropicLift` when a sibling is matrix-valued.

    Only correct for a **sum**: ``k_s + K`` does mean ``k_s I_3 + K``. A product
    must not come through here - see :class:`Product`.
    """
    children = list(children)
    if any(child.block_dim == 3 for child in children):
        children = [IsotropicLift(c) if c.block_dim == 1 else c for c in children]
    return children


def _check_children(children, who: str) -> List[Kernel]:
    children = list(children or [])
    if len(children) < 1:
        raise ValueError(f"{who} needs at least one child kernel")
    for child in children:
        if not isinstance(child, Kernel):
            raise TypeError(f"{who} children must be Kernels, got {type(child).__name__}")
    return children


# --------------------------------------------------------------------------
# sum
# --------------------------------------------------------------------------

@register_kernel("sum")
class Sum(Kernel):
    """``k_1 + k_2 + ...`` - the union of what the terms can express.

    Scalar terms are lifted to ``k I_3`` automatically when any sibling is
    matrix-valued, so ``pca + gaussian`` needs no ceremony.

    Structure is exact where the sum preserves one: two low-rank terms give a
    low-rank sum at the combined rank, which is why ``pca + laplace_beltrami``
    stays cheap. ``diagonal + lowrank`` degrades to ``DENSE`` deliberately - it
    is a diagonal-plus-low-rank matrix, cheap to apply but not itself low-rank,
    and labelling it low-rank would let a truncation discard the diagonal.
    """

    def __init__(self, children: Sequence[Kernel]):
        children = _check_children(children, "sum")
        self.children = _lift_if_needed(children)
        self.block_dim = max(child.block_dim for child in self.children)

    @property
    def structure(self) -> KernelStructure:
        result = self.children[0].structure
        for child in self.children[1:]:
            result = sum_structure(result, child.structure)
        return result

    def block(self, domain, rows=None, cols=None) -> np.ndarray:
        total = self.children[0].block(domain, rows, cols)
        for child in self.children[1:]:
            total = total + child.block(domain, rows, cols)
        return total

    def matmat(self, domain, V, rows=None, cols=None, chunk_bytes=None):
        total = self.children[0].matmat(domain, V, rows, cols, chunk_bytes)
        for child in self.children[1:]:
            total = total + child.matmat(domain, V, rows, cols, chunk_bytes)
        return total

    def diag(self, domain, rows=None) -> np.ndarray:
        total = self.children[0].diag(domain, rows)
        for child in self.children[1:]:
            total = total + child.diag(domain, rows)
        return total

    def factor(self, domain, rows=None) -> Optional[np.ndarray]:
        factors = [child.factor(domain, rows) for child in self.children]
        if any(f is None for f in factors):
            return None
        return np.hstack(factors)

    def supports_cross(self) -> bool:
        return all(child.supports_cross() for child in self.children)

    def cross(self, domain, X, cols=None) -> np.ndarray:
        total = self.children[0].cross(domain, X, cols)
        for child in self.children[1:]:
            total = total + child.cross(domain, X, cols)
        return total

    def analytic_part(self) -> Optional[Kernel]:
        """The sum of whichever terms have one - the usual case being a
        ``pca + gaussian`` model, whose analytic part is the ``gaussian``.

        This is what the continuous extension regresses onto, and dropping the
        tabulated terms is exactly what keeps it non-circular.
        """
        parts = [p for p in (child.analytic_part() for child in self.children) if p is not None]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else Sum(parts)

    def free_parameters(self) -> Dict[str, float]:
        return _namespaced(self.children)

    def with_parameters(self, theta) -> "Sum":
        parts = _split_namespaced(theta, len(self.children))
        return Sum([c.with_parameters(p) for c, p in zip(self.children, parts)])

    def describe(self) -> str:
        return "(" + " + ".join(c.describe() for c in self.children) + f") [{self.structure}]"


# --------------------------------------------------------------------------
# product
# --------------------------------------------------------------------------

@register_kernel("product")
class Product(Kernel):
    """Hadamard or block-scaled product - the localising combinator.

    **The two cases, and why they are not the same.** With a scalar ``s`` and a
    matrix-valued ``K``, the theory's block-scaled product is

        (s * K)_ij = s(p_i, p_j) * K_ij          (a scalar times a 3x3 block)

    Lifting ``s`` to ``s I_3`` and taking a full Hadamard product would instead
    give ``s_ij * K_ij * I_3`` - the block's off-diagonal entries multiplied by
    zero. That discards the correlation between x, y and z displacements, which
    on a PCA kernel is most of what it knows. So :class:`Product` block-scales
    when the operands' ``block_dim`` differ and takes an elementwise product
    only when they agree. This is the one place where the automatic lifting used
    by :class:`Sum` would be a silent correctness bug.

    **Cost.** ``lowrank * lowrank`` is exact via Khatri-Rao and cheap to apply,
    but at rank ``r_a * r_b`` - guarded by :data:`MAX_EXACT_PRODUCT_RANK`, past
    which the generic chunked path is used instead. ``lowrank * dense`` has no
    exact factor and costs a full ``n^2`` block per row chunk; on the baby model
    that is the expensive corner of the whole design. The remedies are at the
    approximator level, not here: truncate the smooth factor to a modest rank
    first so the product becomes ``lowrank * lowrank``, or decompose the product
    by Nystrom, whose ``m x m`` block is just the elementwise product of the two
    ``m x m`` blocks and so costs nothing extra.
    """

    def __init__(self, children: Sequence[Kernel]):
        children = _check_children(children, "product")
        self.children = children
        self.block_dim = max(child.block_dim for child in children)

    @property
    def structure(self) -> KernelStructure:
        result = self.children[0].structure
        for child in self.children[1:]:
            result = product_structure(result, child.structure)
        return result

    # -- pairwise combination ---------------------------------------------

    @staticmethod
    def _combine_blocks(A: np.ndarray, dim_a: int, B: np.ndarray, dim_b: int):
        """Elementwise when the block dimensions agree, block-scaled otherwise."""
        if dim_a == dim_b:
            return A * B, dim_a
        if dim_a == 1:
            return _repeat_blocks(A) * B, dim_b
        return A * _repeat_blocks(B), dim_a

    def block(self, domain, rows=None, cols=None) -> np.ndarray:
        current = self.children[0].block(domain, rows, cols)
        dim = self.children[0].block_dim
        for child in self.children[1:]:
            current, dim = self._combine_blocks(current, dim, child.block(domain, rows, cols), child.block_dim)
        return current

    def diag(self, domain, rows=None) -> np.ndarray:
        """The diagonal of a product needs care across a block-dimension change.

        A scalar factor's diagonal has one entry per vertex; a matrix-valued
        one's has three. The scalar entries are repeated so the two line up.
        """
        current = self.children[0].diag(domain, rows)
        dim = self.children[0].block_dim
        for child in self.children[1:]:
            other = child.diag(domain, rows)
            if dim == child.block_dim:
                current = current * other
            elif dim == 1:
                current, dim = np.repeat(current, 3) * other, child.block_dim
            else:
                current = current * np.repeat(other, 3)
        return current

    def factor(self, domain, rows=None) -> Optional[np.ndarray]:
        factors = []
        dims = []
        for child in self.children:
            F = child.factor(domain, rows)
            if F is None:
                return None
            factors.append(F)
            dims.append(child.block_dim)

        current, dim = factors[0], dims[0]
        for F, child_dim in zip(factors[1:], dims[1:]):
            if dim == 1 and child_dim == 3:
                current, dim = np.repeat(current, 3, axis=0), 3
            elif dim == 3 and child_dim == 1:
                F = np.repeat(F, 3, axis=0)
            if current.shape[1] * F.shape[1] > MAX_EXACT_PRODUCT_RANK:
                return None
            current = _row_khatri_rao(current, F)
        return current

    def supports_cross(self) -> bool:
        return all(child.supports_cross() for child in self.children)

    def cross(self, domain, X, cols=None) -> np.ndarray:
        current = self.children[0].cross(domain, X, cols)
        dim = self.children[0].block_dim
        for child in self.children[1:]:
            current, dim = self._combine_blocks(current, dim, child.cross(domain, X, cols), child.block_dim)
        return current

    def analytic_part(self) -> Optional[Kernel]:
        """All or nothing: one tabulated factor makes the whole product tabulated.

        Unlike a sum, a product cannot be split - ``k_PCA * k_smooth`` has no
        value at a new point just because ``k_smooth`` does.
        """
        parts = [child.analytic_part() for child in self.children]
        if any(p is None for p in parts):
            return None
        return Product(parts)

    def free_parameters(self) -> Dict[str, float]:
        return _namespaced(self.children)

    def with_parameters(self, theta) -> "Product":
        parts = _split_namespaced(theta, len(self.children))
        return Product([c.with_parameters(p) for c, p in zip(self.children, parts)])

    def describe(self) -> str:
        return "(" + " * ".join(c.describe() for c in self.children) + f") [{self.structure}]"


# --------------------------------------------------------------------------
# unary modifiers
# --------------------------------------------------------------------------

@register_kernel("scale")
class Scale(Kernel):
    """``c * k`` with ``c >= 0`` - the mixing weight of a term in a sum.

    Kept as a wrapper rather than folded into each kernel's own ``scale`` so
    that a composite can be re-weighted as a whole: ``0.3 * (pca * gaussian)``
    is one parameter, not two buried in different leaves.
    """

    def __init__(self, child: Kernel, scale: float = 1.0):
        if scale < 0:
            raise ValueError(f"scale must be >= 0, got {scale}")
        self.child = child
        self.scale = float(scale)
        self.block_dim = child.block_dim

    @property
    def structure(self) -> KernelStructure:
        return self.child.structure

    def block(self, domain, rows=None, cols=None):
        return self.scale * self.child.block(domain, rows, cols)

    def matmat(self, domain, V, rows=None, cols=None, chunk_bytes=None):
        return self.scale * self.child.matmat(domain, V, rows, cols, chunk_bytes)

    def diag(self, domain, rows=None):
        return self.scale * self.child.diag(domain, rows)

    def factor(self, domain, rows=None):
        F = self.child.factor(domain, rows)
        return None if F is None else F * math.sqrt(self.scale)

    def supports_cross(self) -> bool:
        return self.child.supports_cross()

    def cross(self, domain, X, cols=None):
        return self.scale * self.child.cross(domain, X, cols)

    def analytic_part(self) -> Optional[Kernel]:
        part = self.child.analytic_part()
        return None if part is None else Scale(part, self.scale)

    def free_parameters(self) -> Dict[str, float]:
        out = {f"0.{k}": v for k, v in self.child.free_parameters().items()}
        if self.scale > 0:
            out["log_scale"] = math.log(self.scale)
        return out

    def with_parameters(self, theta) -> "Scale":
        inner = _split_namespaced(theta, 1)[0]
        scale = math.exp(theta["log_scale"]) if "log_scale" in theta else self.scale
        return Scale(self.child.with_parameters(inner), scale)

    def describe(self) -> str:
        return f"{self.scale:.4g} * {self.child.describe()}"


@register_kernel("localized")
class Localized(Kernel):
    """``w(x) k(x, y) w(y)`` - a kernel restricted to a region.

    Positive semi-definiteness survives, because this is ``D_w K D_w`` with
    ``D_w`` diagonal. Structure survives too: a diagonal congruence changes no
    rank and destroys no sparsity, so localising a low-rank kernel keeps it
    low-rank at the same rank. That makes this the cheap way to get regional
    freedom - much cheaper than a product with a compactly supported kernel,
    and more controllable, since the region is stated rather than implied by a
    radius.

    Where the weights come from: ``src/weight_field/`` already builds
    landmark-centred fields with per-axis decay for exactly this purpose. Pass
    the resulting per-vertex array as ``weights``.

    ``weight_fn`` is optional and only matters for continuous evaluation: the
    weights are tabulated per vertex, so ``w(x*)`` at an arbitrary point is
    undefined unless a callable supplies it. Without it this kernel reports no
    analytic part, which is honest rather than restrictive - it just means a
    localised model needs an unlocalised term somewhere in the sum if it is to
    be sampled off its own vertices.
    """

    def __init__(self, child: Kernel, weights, weight_fn=None):
        self.child = child
        self.weights = np.ascontiguousarray(np.asarray(weights, dtype=np.float64).reshape(-1))
        if np.any(self.weights < 0):
            raise ValueError("localisation weights must be >= 0")
        self.weight_fn = weight_fn
        self.block_dim = child.block_dim

    def _w(self, index: np.ndarray) -> np.ndarray:
        w = self.weights[index]
        return np.repeat(w, 3) if self.block_dim == 3 else w

    @property
    def structure(self) -> KernelStructure:
        return self.child.structure

    def block(self, domain, rows=None, cols=None):
        rows = resolve_indices(domain, rows)
        cols = resolve_indices(domain, cols)
        K = self.child.block(domain, rows, cols)
        return self._w(rows)[:, None] * K * self._w(cols)[None, :]

    def matmat(self, domain, V, rows=None, cols=None, chunk_bytes=None):
        rows = resolve_indices(domain, rows)
        cols = resolve_indices(domain, cols)
        V = np.asarray(V)
        squeeze = V.ndim == 1
        if squeeze:
            V = V[:, None]
        scaled = self._w(cols)[:, None] * V
        out = np.asarray(self.child.matmat(domain, scaled, rows, cols, chunk_bytes))
        out = self._w(rows)[:, None] * out
        return out[:, 0] if squeeze else out

    def diag(self, domain, rows=None):
        rows = resolve_indices(domain, rows)
        return (self._w(rows) ** 2) * self.child.diag(domain, rows)

    def factor(self, domain, rows=None):
        rows = resolve_indices(domain, rows)
        F = self.child.factor(domain, rows)
        return None if F is None else self._w(rows)[:, None] * F

    def supports_cross(self) -> bool:
        return self.weight_fn is not None and self.child.supports_cross()

    def cross(self, domain, X, cols=None):
        if self.weight_fn is None:
            raise NotImplementedError(
                "localized kernel has tabulated weights, so w(x*) is undefined at a new "
                "point. Pass weight_fn=... to make it continuous, or keep an unlocalised "
                "term in the sum to extend against."
            )
        cols = resolve_indices(domain, cols)
        K = self.child.cross(domain, X, cols)
        wx = np.asarray(self.weight_fn(X), dtype=np.float64).reshape(-1)
        if self.block_dim == 3:
            wx = np.repeat(wx, 3)
        return wx[:, None] * K * self._w(cols)[None, :]

    def analytic_part(self) -> Optional[Kernel]:
        part = self.child.analytic_part()
        if part is None or self.weight_fn is None:
            return None
        return Localized(part, self.weights, self.weight_fn)

    def free_parameters(self) -> Dict[str, float]:
        return {f"0.{k}": v for k, v in self.child.free_parameters().items()}

    def with_parameters(self, theta) -> "Localized":
        inner = _split_namespaced(theta, 1)[0]
        return Localized(self.child.with_parameters(inner), self.weights, self.weight_fn)

    def describe(self) -> str:
        return (
            f"localized({self.child.describe()}, w in "
            f"[{self.weights.min():.3g}, {self.weights.max():.3g}])"
        )


@register_kernel("symmetry")
class Symmetry(Kernel):
    """``(K + P K P^T) / 2`` - a process that respects a bilateral symmetry.

    ``P`` maps a vertex to its mirror partner and reflects the displacement,
    so a sample drawn from this kernel is as likely as its mirror image. On a
    face model that is a strong and correct prior: it does not force symmetry
    on any one sample, it removes the model's ability to prefer one side.

    PSD is immediate - ``P K P^T`` is a congruence of a PSD matrix, and the sum
    of two PSD matrices is PSD - so no eigenvalue clipping is needed. The rank
    doubles, which is the cost.

    ``mirror`` is the vertex permutation (``mirror[i]`` is ``i``'s partner) and
    ``reflection`` the 3x3 matrix applied to the displacement, by default the
    reflection in the x = 0 plane. Both are tabulated per vertex, so this offers
    no analytic part.
    """

    def __init__(self, child: Kernel, mirror, reflection=None):
        self.child = child
        self.mirror = np.ascontiguousarray(np.asarray(mirror, dtype=np.int64).reshape(-1))
        self.reflection = (
            np.diag([-1.0, 1.0, 1.0]) if reflection is None
            else np.asarray(reflection, dtype=np.float64).reshape(3, 3)
        )
        self.block_dim = child.block_dim

    @property
    def structure(self) -> KernelStructure:
        base = self.child.structure
        if base.is_lowrank:
            return lowrank(2 * base.rank)
        return base if base.kind is Structure.DENSE else KernelStructure(Structure.DENSE)

    def _reflect_block(self, K: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
        if self.block_dim == 1:
            return K
        R = self.reflection
        K4 = K.reshape(n_rows, 3, n_cols, 3)
        return np.einsum("ac,rcsd,bd->rasb", R, K4, R).reshape(3 * n_rows, 3 * n_cols)

    def block(self, domain, rows=None, cols=None):
        rows = resolve_indices(domain, rows)
        cols = resolve_indices(domain, cols)
        direct = self.child.block(domain, rows, cols)
        mirrored = self.child.block(domain, self.mirror[rows], self.mirror[cols])
        return 0.5 * (direct + self._reflect_block(mirrored, rows.size, cols.size))

    def diag(self, domain, rows=None):
        rows = resolve_indices(domain, rows)
        direct = self.child.diag(domain, rows)
        mirrored = self.child.diag(domain, self.mirror[rows])
        if self.block_dim == 1:
            return 0.5 * (direct + mirrored)
        # Only the diagonal of R B R^T is needed: sum_c R[a,c]^2 B[c,c].
        squared = (self.reflection ** 2)
        blocks = mirrored.reshape(-1, 3) @ squared.T
        return 0.5 * (direct + blocks.reshape(-1))

    def factor(self, domain, rows=None):
        rows = resolve_indices(domain, rows)
        F = self.child.factor(domain, rows)
        if F is None:
            return None
        Fm = self.child.factor(domain, self.mirror[rows])
        if self.block_dim == 3:
            Fm = np.einsum("ac,rck->rak", self.reflection, Fm.reshape(rows.size, 3, -1))
            Fm = Fm.reshape(3 * rows.size, -1)
        return np.hstack([F, Fm]) / math.sqrt(2.0)

    def analytic_part(self) -> Optional[Kernel]:
        return None

    def free_parameters(self) -> Dict[str, float]:
        return {f"0.{k}": v for k, v in self.child.free_parameters().items()}

    def with_parameters(self, theta) -> "Symmetry":
        inner = _split_namespaced(theta, 1)[0]
        return Symmetry(self.child.with_parameters(inner), self.mirror, self.reflection)

    def describe(self) -> str:
        return f"symmetry({self.child.describe()})"


@register_kernel("isotropic_lift")
class IsotropicLift(Kernel):
    """A scalar kernel as ``k(p, p') I_3`` - the same prior on x, y and z.

    Inserted automatically by :class:`Sum`, so it rarely appears in a spec.
    Written explicitly it is a statement that the three coordinates are
    independent and identically distributed, which is what a scalar kernel
    means and is *not* what a data kernel says.

    The lift is also why a scalar kernel can be decomposed on the ``n x n``
    problem rather than ``3n x 3n``: each eigenvalue simply gains multiplicity
    three. The low-rank layer reads that off ``block_dim`` on the child.
    """

    def __init__(self, child: Kernel):
        if child.block_dim != 1:
            raise ValueError(f"isotropic_lift takes a scalar kernel, got block_dim={child.block_dim}")
        self.child = child
        self.block_dim = 3

    @property
    def structure(self) -> KernelStructure:
        base = self.child.structure
        return lowrank(3 * base.rank) if base.is_lowrank else base

    def block(self, domain, rows=None, cols=None):
        return np.kron(self.child.block(domain, rows, cols), np.eye(3))

    def matmat(self, domain, V, rows=None, cols=None, chunk_bytes=None):
        """Three scalar products, not one of triple the size.

        ``(S kron I) V`` acts on each coordinate independently, so reshaping to
        ``(n, 3k)`` and applying ``S`` once is exactly equivalent and costs a
        third of the work of building the lifted matrix.
        """
        cols_idx = resolve_indices(domain, cols)
        V = np.asarray(V)
        squeeze = V.ndim == 1
        if squeeze:
            V = V[:, None]
        k = V.shape[1]
        reshaped = V.reshape(cols_idx.size, 3 * k)
        out = self.child.matmat(domain, reshaped, rows, cols, chunk_bytes)
        out = np.asarray(out).reshape(-1, k)
        return out[:, 0] if squeeze else out

    def diag(self, domain, rows=None):
        return np.repeat(self.child.diag(domain, rows), 3)

    def factor(self, domain, rows=None):
        F = self.child.factor(domain, rows)
        return None if F is None else np.kron(F, np.eye(3))

    def supports_cross(self) -> bool:
        return self.child.supports_cross()

    def cross(self, domain, X, cols=None):
        return np.kron(self.child.cross(domain, X, cols), np.eye(3))

    def analytic_part(self) -> Optional[Kernel]:
        part = self.child.analytic_part()
        return None if part is None else IsotropicLift(part)

    def free_parameters(self) -> Dict[str, float]:
        return self.child.free_parameters()

    def with_parameters(self, theta) -> "IsotropicLift":
        return IsotropicLift(self.child.with_parameters(theta))

    def describe(self) -> str:
        return f"lift({self.child.describe()})"
