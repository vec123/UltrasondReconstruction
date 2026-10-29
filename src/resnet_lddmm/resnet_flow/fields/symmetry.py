"""Mirror symmetry of a velocity field, exact by construction or priced as a cheaper part.

With ``M`` the reflection in the face's mid-plane, any field ``g`` splits as

    g(x) = sym(x) + anti(x),   sym(x)  = (g(x) + M g(Mx)) / 2   (sym(Mx)  =  M sym(x))
                               anti(x) = (g(x) - M g(Mx)) / 2   (anti(Mx) = -M anti(x))

A stationary symmetric field gives a symmetric flow, phi(Mx) = M phi(x): the
symmetry is a pointwise property of the field, so no second integration and no
left/right vertex correspondence is needed. The two modes:

  * ``symmetric``: the field IS ``sym`` -- every flow it makes is mirror-symmetric.
  * ``split``: the field is ``g`` as it was, and ``split`` returns ``sym`` as the
    priced part "symmetric", so the kinetic term charges it at
    ``discount * kinetic_weight`` and ``anti`` at the full price: symmetric motion is
    cheap, asymmetric motion possible but dearer.

Both evaluate ``g`` once on ``[x, Mx]`` (2N points). The plane is placed once in the
training frame from the template's landmarks (``runner._place_symmetry``) and kept in
buffers, so every checkpoint carries it and apply needs neither landmarks nor template.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.resnet_lddmm.resnet_flow.config import SYMMETRY_MODES
from src.resnet_lddmm.resnet_flow.fields.base import VelocityField


class MirrorPlane(nn.Module):
    """The plane {x : n.x = d}, |n| = 1, and its reflection M x = x - 2 (n.x - d) n."""

    def __init__(self):
        super().__init__()
        self.register_buffer("normal", torch.zeros(3))
        self.register_buffer("offset", torch.zeros(()))
        self.placed = False

    def place(self, normal: Tensor, offset: float) -> None:
        """Set the plane: ``normal`` [3] (normalised here) and ``offset`` d, in the training frame."""
        normal = torch.as_tensor(normal, dtype=self.normal.dtype)
        self.normal = (normal / normal.norm()).to(self.normal.device)
        self.offset = torch.tensor(float(offset), dtype=self.offset.dtype, device=self.offset.device)
        self.placed = True

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        normal = state_dict.get(prefix + "normal")
        self.placed = normal is not None and bool(normal.abs().sum() > 0)

    def distance(self, x: Tensor) -> Tensor:
        """[..., 3] -> [...], signed distance n.x - d."""
        return x @ self.normal.to(x.dtype) - self.offset.to(x.dtype)

    def reflect_points(self, x: Tensor) -> Tensor:
        """[..., 3] -> [..., 3], M x."""
        return x - 2.0 * self.distance(x)[..., None] * self.normal.to(x.dtype)

    def reflect_vectors(self, v: Tensor) -> Tensor:
        """[..., 3] -> [..., 3], the linear part of M (a velocity reflects without the offset)."""
        n = self.normal.to(v.dtype)
        return v - 2.0 * (v @ n)[..., None] * n


class SymmetricField(VelocityField):
    """A pointwise field made mirror-symmetric (``symmetric``) or split with a cheaper symmetric part (``split``)."""

    def __init__(self, network: VelocityField, mode: str = "symmetric"):
        """Initialize the field.

        Args:
            network: the pointwise field g (its velocity at a point depends on that point only)
            mode: "symmetric" (the field is sym) or "split" (the field is g, sym priced apart)
        """
        super().__init__()
        if mode not in SYMMETRY_MODES:
            raise ValueError(f"symmetry mode must be one of {SYMMETRY_MODES}, got {mode!r}")
        self.network = network
        self.mode = mode
        self.plane = MirrorPlane()

    def halves(self, x: Tensor, step: Optional[int] = None,
               code: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """(sym, anti), both [B, N, 3]: g evaluated once on [x, Mx].

        Raises:
            RuntimeError: the plane was never placed (the reflection would be garbage).
        """
        if not self.plane.placed:
            raise RuntimeError("the mirror plane was never placed: call runner._place_symmetry "
                               "(training) or load a checkpoint that carries it (apply)")
        n = x.shape[1]
        values = self.network(torch.cat([x, self.plane.reflect_points(x)], dim=1), step=step, code=code)
        direct, mirrored = values[:, :n], self.plane.reflect_vectors(values[:, n:])
        return 0.5 * (direct + mirrored), 0.5 * (direct - mirrored)

    def split(self, x: Tensor, step: Optional[int] = None,
              code: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        """(velocity, priced parts): (sym, {}) when symmetric, (g, {"symmetric": sym}) when split."""
        sym, anti = self.halves(x, step=step, code=code)
        if self.mode == "symmetric":
            return sym, {}
        return sym + anti, {"symmetric": sym}

    def forward(self, x: Tensor, step: Optional[int] = None, code: Optional[Tensor] = None) -> Tensor:
        return self.split(x, step=step, code=code)[0]
