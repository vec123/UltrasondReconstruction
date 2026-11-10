"""A velocity field that is exactly zero at the zero code, so the zero code IS the template.

    v(x, t, z) = B_t(x [, z]) . (A z),    B_t(x) in R^{3 x r},  A: R^{n_z} -> R^r (no bias)

The inner network gives a basis of r velocity fields per point and step, as one
output of 3 * r entries (blocks.VelocityBlock out_dim). The code selects a
combination of them. Because the code enters through the product with ``A z``,
v = 0 at z = 0 for every x and t: the flow of the zero code is the identity, and
the zero code maps the template to itself, by construction and without a loss term.

Two modes, decided by the conditioning the builder gives the inner network
(runner.model._build_atlas_field):

  * ``linear``: the inner network reads no code (NoConditioning), so
    v = sum_k (A z)_k b_k(x, t) is linear in z. The deformation, the flow of v, is
    linear in z to first order only, because the points move along the flow.
  * ``nonlinear``: the inner network keeps the run's conditioning and reads z, so
    v is non-linear in z, but still exactly zero at z = 0.

Two ranks:

  * full (``rank`` None): r = n_z and A is the identity: one basis field per code entry.
  * low (``rank`` an int r < n_z): A is a learned n_z -> r map, so the network
    outputs 3 * r instead of 3 * n_z entries per point (less memory and time).

The zero code is the template; the template is the population centre only when
the codes have a mean near zero. This field does not enforce that.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.models.lddmm.resnet_flow.fields.base import VelocityField


def basis_rank(n_z: int, rank: Optional[int]) -> int:
    """The number r of basis fields: n_z for the full rank (None), else ``rank``.

    Raises:
        ValueError: n_z < 1, or rank outside 1 <= rank < n_z.
    """
    if n_z < 1:
        raise ValueError(f"an atlas field needs a code of at least one entry, got n_z = {n_z}")
    if rank is None:
        return int(n_z)
    if not 1 <= int(rank) < n_z:
        raise ValueError(f"atlas rank must be 1 <= rank < n_z = {n_z}, got {rank}; "
                         f"use rank: full for r = n_z")
    return int(rank)


class AtlasField(VelocityField):
    """``B(x [, z]) . (A z)``: a basis network contracted with the (mixed) code."""

    def __init__(self, network: VelocityField, n_z: int, rank: Optional[int] = None):
        """Initialize the field.

        Args:
            network: the basis network, built with out_dim = 3 * basis_rank(n_z, rank).
                Its conditioning decides the mode: NoConditioning gives the linear one.
            n_z: the code size.
            rank: None for the full rank (A = identity), else r < n_z (A learned).
        """
        super().__init__()
        self.network = network
        self.n_z = int(n_z)
        self.rank = basis_rank(n_z, rank)
        # No parameters for the full rank, so its state_dict is the network's alone.
        self.mix = nn.Identity() if rank is None else nn.Linear(self.n_z, self.rank, bias=False)

    def forward(self, x: Tensor, step: Optional[int] = None, code: Optional[Tensor] = None) -> Tensor:
        """[B, N, 3] positions and [B, n_z] code -> [B, N, 3] velocity.

        Raises:
            ValueError: no code, or a code of the wrong size.
        """
        if code is None:
            raise ValueError("an atlas field needs a code: its velocity is B(x) . (A z)")
        if code.shape[-1] != self.n_z:
            raise ValueError(f"atlas field reads a code of {self.n_z} entries, got {code.shape[-1]}")
        basis = self.network(x, step=step, code=code)                    # [B, N, 3 * r]
        basis = basis.view(*x.shape[:-1], 3, self.rank)                 # [B, N, 3, r]
        return torch.einsum("bndr,br->bnd", basis, self.mix(code).to(basis.dtype))
