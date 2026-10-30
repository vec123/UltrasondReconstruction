"""A kernel made symmetric under a mirror map."""

from __future__ import annotations

import math
from typing import Dict, Optional
import numpy as np
from src.SSM.kernels.base import Kernel, KernelStructure, Structure, lowrank, register_kernel, resolve_indices

from .common import _split_namespaced


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
