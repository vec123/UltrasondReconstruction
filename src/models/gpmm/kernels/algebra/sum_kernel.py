"""Sum of kernels: adds directions the population never showed."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np
from src.models.gpmm.kernels.base import Kernel, KernelStructure, register_kernel, sum_structure

from .common import _check_children, _namespaced, _split_namespaced
from .lift import IsotropicLift


def _lift_if_needed(children: Sequence[Kernel]) -> List[Kernel]:
    """Wrap scalar children in :class:`IsotropicLift` when a sibling is matrix-valued.

    Only correct for a **sum**: ``k_s + K`` does mean ``k_s I_3 + K``. A product
    must not come through here - see :class:`Product`.
    """
    children = list(children)
    if any(child.block_dim == 3 for child in children):
        children = [IsotropicLift(c) if c.block_dim == 1 else c for c in children]
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
