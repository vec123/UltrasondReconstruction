"""Multi-resolution field: a global coarse field plus a fine field confined to one region.

    v(x) = v_coarse(x) + m(x) * v_fine(x)

One global Fourier schedule has to serve every region at once: opening the
bands late fits the overall shape well but leaves small features underfit, and
opening them early fits the features but lets fine bands distort the whole
face. Splitting the field decouples the two -- the coarse field keeps its own
(slow or no) schedule everywhere, and the fine field, with more bands and a fast
schedule of its own, acts only inside the mask.

The mask is a function of SPACE, not of template vertex: the field is evaluated
at the points' current positions along the trajectory, so a per-vertex field
could not be looked up there. It is 1 on a ball of radius ``r_in`` around
``center``, falls to 0 with a C2 "smootherstep" at ``r_out``, and is exactly 0
beyond -- so outside it the model IS the coarse model, and fine-band folding
risk is confined to the region. Placed once from the template's own landmarks
(``runner._place_local_mask``); ``center`` and the radii are buffers, so every
checkpoint carries them and apply reproduces the same region.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from src.resnet_lddmm.resnet_flow.conditioning.base import Conditioning
from src.resnet_lddmm.resnet_flow.fields.base import VelocityField
from src.resnet_lddmm.resnet_flow.fields.fourier import FourierTimeVaryingField
from src.resnet_lddmm.resnet_flow.fields.time_varying import TimeVaryingField


def _smootherstep(s: Tensor) -> Tensor:
    """6s^5 - 15s^4 + 10s^3: 0 -> 1 on [0, 1] with zero slope and curvature at both ends."""
    return s.pow(3) * (s * (6.0 * s - 15.0) + 10.0)


class SmoothBallMask(nn.Module):
    """m(x) = 1 within ``r_in`` of ``center``, C2 falloff to exactly 0 at ``r_out``.

    Falloff ``1 - S(t)``, ``S(t) = 6t^5 - 15t^4 + 10t^3``,
    ``t = clamp((d - r_in)/(r_out - r_in), 0, 1)``: value, slope and curvature
    are continuous at both radii, so the masked field stays as smooth as the
    field it masks. Each half is evaluated from its accurate side (S is
    symmetric, ``1 - S(t) = S(1 - t)``): the direct form cancels in float32
    near the outer edge, the mirrored one overshoots 1 near the inner edge.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("center", torch.zeros(3))
        self.register_buffer("r_in", torch.tensor(0.0))
        self.register_buffer("r_out", torch.tensor(0.0))
        self.placed = False

    def place(self, center: Tensor, r_in: float, r_out: float) -> None:
        """Set the region, in the frame the field is evaluated in (normalised).

        Raises:
            ValueError: r_in < 0 or r_out <= r_in.
        """
        if r_in < 0 or r_out <= r_in:
            raise ValueError(f"mask needs 0 <= r_in < r_out, got r_in={r_in}, r_out={r_out}")
        self.center.copy_(torch.as_tensor(center, dtype=self.center.dtype))
        self.r_in.fill_(float(r_in))
        self.r_out.fill_(float(r_out))
        self.placed = True

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # A checkpoint's buffers ARE a placement: apply never calls place().
        self.placed = f"{prefix}r_out" in state_dict and float(state_dict[f"{prefix}r_out"]) > 0

    def forward(self, x: Tensor) -> Tensor:
        """[..., 3] -> [..., 1] mask values in [0, 1].

        Raises:
            RuntimeError: the mask was never placed, which would silently turn
                the fine field off.
        """
        if not self.placed:
            raise RuntimeError("local mask was never placed: call place() (training) or "
                               "load a checkpoint that carries it (apply)")
        center = self.center.to(dtype=x.dtype)
        # sqrt(|x-c|^2 + eps^2): differentiable at the centre, where |x-c| is not.
        d = ((x - center).pow(2).sum(-1, keepdim=True) + 1e-24).sqrt()
        t = ((d - self.r_in) / (self.r_out - self.r_in)).clamp(0.0, 1.0)
        return torch.where(t < 0.5, 1.0 - _smootherstep(t), _smootherstep(1.0 - t))


class LocalRefinementField(VelocityField):
    """``v_coarse(x) + m(x) * v_fine(x)``, one block per time step in both parts.

    The coarse part is a plain ``TimeVaryingField`` when ``coarse_n_e`` is 0,
    else a ``FourierTimeVaryingField``; the fine part is always a
    ``FourierTimeVaryingField``. Both project through zero-initialised layers,
    so the flow is the identity at initialisation. They share ``conditioning``
    (one module, registered under both).
    """

    def __init__(
        self,
        num_blocks: int = 10,
        mlp: Sequence[int] = (128,),
        activation: str = "relu",
        conditioning: Optional[Conditioning] = None,
        coarse_n_e: int = 0,
        coarse_annealed: bool = False,
        fine_mlp: Sequence[int] = (128, 128),
        fine_n_e: int = 5,
        fine_annealed: bool = True,
    ):
        """Initialize both fields and an unplaced mask.

        Args:
            num_blocks: time steps K, shared by both parts.
            mlp: coarse hidden widths.
            activation: "relu" | "leaky_relu", both parts.
            conditioning: optional conditioning module, shared.
            coarse_n_e: coarse Fourier bands; 0 = plain xyz (TimeVaryingField).
            coarse_annealed: coarse bands start closed (needs coarse_n_e >= 1).
            fine_mlp: fine hidden widths.
            fine_n_e: fine Fourier bands, >= 1.
            fine_annealed: fine bands start closed, for a FourierAnnealing to open.
        """
        super().__init__()
        if coarse_n_e > 0:
            self.coarse = FourierTimeVaryingField(num_blocks, mlp, activation, conditioning,
                                                  n_e=coarse_n_e, annealed=coarse_annealed)
        else:
            if coarse_annealed:
                raise ValueError("coarse_annealed needs coarse_n_e >= 1: there are no bands to open")
            self.coarse = TimeVaryingField(num_blocks, mlp, activation, conditioning)
        self.fine = FourierTimeVaryingField(num_blocks, fine_mlp, activation, conditioning,
                                            n_e=fine_n_e, annealed=fine_annealed)
        self.mask = SmoothBallMask()

    def forward(
        self, x: Tensor, step: Optional[int] = None, code: Optional[Tensor] = None
    ) -> Tensor:
        """Coarse velocity everywhere plus the masked fine velocity.

        Args:
            x: [B, N, 3] point positions
            step: which block (time step) to use, required
            code: [B, N_z] per-shape code (ignored by NoConditioning)

        Returns:
            [B, N, 3] velocity
        """
        return self.coarse(x, step=step, code=code) + self.mask(x) * self.fine(x, step=step, code=code)
