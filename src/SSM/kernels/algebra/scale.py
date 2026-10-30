"""A kernel multiplied by a constant."""

from __future__ import annotations

import math
from typing import Dict, Optional
from src.SSM.kernels.base import Kernel, KernelStructure, register_kernel

from .common import _split_namespaced


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
