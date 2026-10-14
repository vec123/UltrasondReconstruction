"""Designed deformation modes added to a learned field, at a discounted kinetic cost.

    v(x, t) = v_net(x, t) + sum_m h_m(t) phi_m(x)

``phi_m`` are fixed fields designed in the deformation-mode editor
(``src/deformation_modes``) - e.g. "widen the alae, the cheek beside them
slides but does not bulge" - and ``h_m(t)`` one learned amplitude per mode and
Euler step. The point is the price: the kinetic term charges only the
network's part (``Trajectory.free_velocities``), and the modes' part is charged
separately at a discount (``losses.kinetic.ModeKineticLoss``), so moving along
a designed mode is cheap and any other motion costs what it always did.

Each ``phi_m`` is a Gaussian kernel expansion (``KernelExpansion``), placed once
in the training frame by ``runner._place_modes`` and kept in buffers, so every
checkpoint carries the exact fields its amplitudes were trained with and apply
needs neither the spec nor the template.
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.resnet_lddmm.resnet_flow.fields.base import VelocityField


class ModeBasis(nn.Module):
    """The fixed fields ``phi_1..phi_M``, evaluated at any points.

    phi_m(x) = sum_c exp(-|x - c|^2 / 2 sigma_m^2) (a_mc + M_mc (x - c))

    Modes with fewer kernel centres than the largest are zero-padded (a zero
    term adds nothing wherever its centre is). The buffers are sized when the
    basis is placed, or when a checkpoint is loaded into it.
    """

    def __init__(self, n_modes: int):
        super().__init__()
        if n_modes < 1:
            raise ValueError(f"a ModeBasis needs at least one mode, got {n_modes}")
        self.n_modes = int(n_modes)
        self.register_buffer("centers", torch.zeros(n_modes, 0, 3))
        self.register_buffer("offsets", torch.zeros(n_modes, 0, 3))      # a - M c
        self.register_buffer("matrices", torch.zeros(n_modes, 0, 9))     # M, row-major
        self.register_buffer("sigma", torch.ones(n_modes))
        self.placed = False

    def place(self, expansions: Sequence) -> None:
        """Set the fields from one ``KernelExpansion`` per mode, in the training frame.

        Raises:
            ValueError: the number of expansions is not ``n_modes``.
        """
        if len(expansions) != self.n_modes:
            raise ValueError(f"basis holds {self.n_modes} modes, got {len(expansions)} fields")
        size = max(len(e.centers) for e in expansions)
        dtype = self.sigma.dtype
        centers = torch.zeros(self.n_modes, size, 3, dtype=dtype)
        offsets = torch.zeros(self.n_modes, size, 3, dtype=dtype)
        matrices = torch.zeros(self.n_modes, size, 9, dtype=dtype)
        for m, e in enumerate(expansions):
            c = torch.as_tensor(e.centers, dtype=torch.float64)
            a = torch.as_tensor(e.vectors, dtype=torch.float64)
            mat = torch.as_tensor(e.matrices, dtype=torch.float64)
            n = len(c)
            centers[m, :n] = c.to(dtype)
            offsets[m, :n] = (a - torch.einsum("cij,cj->ci", mat, c)).to(dtype)
            matrices[m, :n] = mat.reshape(n, 9).to(dtype)
        device = self.sigma.device
        self.centers = centers.to(device)
        self.offsets = offsets.to(device)
        self.matrices = matrices.to(device)
        self.sigma = torch.tensor([float(e.sigma) for e in expansions], dtype=dtype, device=device)
        self.placed = True

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # A checkpoint's buffers ARE a placement, of their own size: resize to
        # them before the default loader insists on the unplaced (empty) shapes.
        for name in ("centers", "offsets", "matrices", "sigma"):
            key = prefix + name
            if key in state_dict:
                setattr(self, name, torch.empty_like(state_dict[key], device=self.sigma.device))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self.placed = f"{prefix}centers" in state_dict and state_dict[f"{prefix}centers"].shape[1] > 0

    def forward(self, x: Tensor) -> Tensor:
        """[B, N, 3] -> [B, N, M, 3], every mode's field at every point.

        Raises:
            RuntimeError: the basis was never placed (the modes would silently be zero).
        """
        if not self.placed:
            raise RuntimeError("deformation modes were never placed: call place() (training) "
                               "or load a checkpoint that carries them (apply)")
        centers = self.centers.to(x.dtype)
        # |x - c|^2 by expansion: never materialises [B, N, M, C, 3].
        sq = ((x * x).sum(-1)[..., None, None] - 2.0 * torch.einsum("bnd,mcd->bnmc", x, centers)
              + (centers * centers).sum(-1)).clamp_min(0.0)
        k = torch.exp(-0.5 * sq / self.sigma.to(x.dtype)[:, None] ** 2)                 # [B, N, M, C]
        constant = torch.einsum("bnmc,mcd->bnmd", k, self.offsets.to(x.dtype))
        linear = torch.einsum("bnmc,mce->bnme", k, self.matrices.to(x.dtype))
        linear = linear.reshape(*linear.shape[:-1], 3, 3)
        return constant + torch.einsum("bnmij,bnj->bnmi", linear, x)


class ModalField(VelocityField):
    """A learned field plus designed modes with learned amplitudes, one per mode and step.

    ``amplitude[k, m]`` is mode m's speed at step k, in the units the basis was
    normalised to (``runner._place_modes``: 1 = peak speed of 1 mm per unit
    time), so ``amplitude[:, m].mean()`` is roughly the peak displacement the
    mode contributes over the whole flow. Starts at zero: the flow is the
    network's alone until the data asks for a mode.
    """

    def __init__(self, network: VelocityField, names: Sequence[str], num_steps: int):
        super().__init__()
        self.network = network
        self.names: List[str] = list(names)
        self.basis = ModeBasis(len(self.names))
        self.amplitude = nn.Parameter(torch.zeros(int(num_steps), len(self.names)))

    def split(self, x: Tensor, step: Optional[int] = None,
              code: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """(total velocity, the modes' part of it), both [B, N, 3].

        Integrators record the second so the kinetic cost can charge it apart
        from the network's part.
        """
        amplitude = self.amplitude[0 if step is None else step].to(x.dtype)
        modal = torch.einsum("bnmd,m->bnd", self.basis(x), amplitude)
        return self.network(x, step=step, code=code) + modal, modal

    def forward(self, x: Tensor, step: Optional[int] = None, code: Optional[Tensor] = None) -> Tensor:
        return self.split(x, step=step, code=code)[0]


def network_of(field: VelocityField) -> VelocityField:
    """The learned network inside ``field`` - itself, unless modes are wrapped around it."""
    return field.network if isinstance(field, ModalField) else field
