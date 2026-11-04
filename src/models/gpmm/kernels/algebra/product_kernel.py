"""Product of kernels: localises, keeping the local correlation structure."""

from __future__ import annotations

from typing import Dict, Optional, Sequence
import numpy as np
from src.models.gpmm.kernels.base import Kernel, KernelStructure, product_structure, register_kernel

from .common import _check_children, _namespaced, _split_namespaced


#: Refuse to build a Khatri-Rao factor past this many columns. A product of two
#: rank-200 kernels is rank 40000, which is larger than the matrix it factors on
#: any domain worth using it on; past this the generic chunked path is cheaper.
MAX_EXACT_PRODUCT_RANK = 20_000


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
