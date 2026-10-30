"""Lifts a scalar kernel to a vector-valued one with an identity block."""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
from src.SSM.kernels.base import Kernel, KernelStructure, lowrank, register_kernel, resolve_indices


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
