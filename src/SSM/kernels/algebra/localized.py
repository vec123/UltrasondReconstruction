"""A kernel localised by per-vertex weights."""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
from src.SSM.kernels.base import Kernel, KernelStructure, register_kernel, resolve_indices

from .common import _split_namespaced


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
