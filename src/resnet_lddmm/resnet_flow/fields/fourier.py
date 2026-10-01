"""Fourier-feature time-varying field, with optional coarse-to-fine band annealing.

A ReLU MLP on raw xyz is spectrally biased: it fits the large, smooth part of a
deformation quickly and small features (a nose, in a face normalised to unit
extent) only slowly. Feeding it ``[x, sin(pi 2^k x), cos(pi 2^k x)]`` for
k = 0..n_e-1 hands it those shorter periods directly -- band k has period
``2 / 2^k`` in the normalised frame, so with the face's largest extent at 1:

    n_e = 3  finest period 0.5    (~half the face)
    n_e = 4  finest period 0.25   (nose-sized)
    n_e = 5  finest period 0.125  (alae, tip)
    n_e >= 6 approaches scan-noise scale

Annealing (the BARF window, Lin et al. 2021) switches band k on smoothly as a
progress value ``alpha`` passes k:

    w_k(alpha) = (1 - cos(pi * clamp(alpha - k, 0, 1))) / 2,   alpha in [0, n_e]

At alpha = 0 every band is zero and the field sees plain xyz -- the current
``time_varying`` model's input, up to dead weights -- so training fits the
coarse shape first and the finer bands fade in on a schedule set by
``callbacks_fourier.FourierAnnealing``. ``alpha`` is a buffer: it is saved in
every checkpoint, so apply reproduces exactly the bandwidth that was trained.

Higher bands make the velocity field steeper, and with a fixed number of Euler
steps a steep enough step folds the surface. Nothing in the kinetic term charges
steepness (it charges speed), so watch diag/det_min and diag/flips_count.
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from src.resnet_lddmm.resnet_flow.conditioning.base import Conditioning, NoConditioning
from src.resnet_lddmm.resnet_flow.fields.base import VelocityField
from src.resnet_lddmm.resnet_flow.fields.blocks import FourierFeatures, VelocityBlock


class AnnealedFourierFeatures(FourierFeatures):
    """``FourierFeatures`` with each band scaled by the BARF window at ``alpha``.

    Same output layout as the parent -- ``[h, sin (d-major), cos (d-major)]`` --
    so at ``alpha = n_e`` the two are identical, and at ``alpha = 0`` only the
    ``h`` passthrough is non-zero.
    """

    def __init__(self, n_e: int, alpha: Optional[float] = None):
        """Initialize the encoding.

        Args:
            n_e: number of frequency bands, >= 1.
            alpha: initial progress in [0, n_e]; None means n_e (every band fully
                on, i.e. no annealing).

        Raises:
            ValueError: n_e < 1, or alpha outside [0, n_e].
        """
        if n_e < 1:
            raise ValueError(f"n_e must be >= 1, got {n_e}; use kind: time_varying for no bands")
        super().__init__(n_e)
        self.n_e = n_e
        self.register_buffer("alpha", torch.tensor(float(n_e)))
        self.set_alpha(n_e if alpha is None else alpha)

    def set_alpha(self, alpha: float) -> None:
        """Set the annealing progress, in [0, n_e].

        Raises:
            ValueError: alpha outside [0, n_e].
        """
        if not 0.0 <= alpha <= self.n_e:
            raise ValueError(f"alpha must be in [0, {self.n_e}], got {alpha}")
        self.alpha.fill_(float(alpha))

    def band_weights(self) -> Tensor:
        """[n_e] window weight per band, each in [0, 1]."""
        k = torch.arange(self.n_e, device=self.alpha.device, dtype=self.alpha.dtype)
        return (1.0 - torch.cos(math.pi * (self.alpha - k).clamp(0.0, 1.0))) / 2.0

    def forward(self, h: Tensor) -> Tensor:
        """[..., d] -> [..., (2*n_e + 1)*d], bands scaled by their window weight."""
        features = super().forward(h)
        d = h.shape[-1]
        band = self.band_weights().to(features.dtype).repeat(d)   # d-major, like the parent
        scale = torch.cat([features.new_ones(d), band, band])
        return features * scale


class FourierTimeVaryingField(VelocityField):
    """``TimeVaryingField`` with its xyz input passed through ``AnnealedFourierFeatures``.

    Block l is time step l, exactly as in ``TimeVaryingField``; only the input
    changes, from ``[x, z]`` to ``[gamma(x), z]``. The conditioning features z are
    NOT encoded -- they are a code, not a position. Each block's projection is
    zero-initialised by ``VelocityBlock``, so the flow is still the identity at
    initialisation.
    """

    def __init__(
        self,
        num_blocks: int = 10,
        mlp: Sequence[int] = (512,),
        activation: str = "relu",
        conditioning: Optional[Conditioning] = None,
        n_e: int = 4,
        annealed: bool = False,
    ):
        """Initialize the field.

        Args:
            num_blocks: time steps K, one block each.
            mlp: hidden widths per block, as for TimeVaryingField.
            activation: "relu" | "leaky_relu".
            conditioning: optional conditioning module (default NoConditioning).
            n_e: number of Fourier bands, >= 1.
            annealed: start with every band off (alpha = 0), for a
                FourierAnnealing callback to open; False starts fully open.
        """
        super().__init__()
        self.conditioning = conditioning or NoConditioning()
        self.encoding = AnnealedFourierFeatures(n_e, alpha=0.0 if annealed else None)
        in_dim = 3 * (2 * n_e + 1) + self.conditioning.dim
        self.blocks = nn.ModuleList(
            VelocityBlock(in_dim, mlp, activation) for _ in range(num_blocks)
        )

    def forward(
        self, x: Tensor, step: Optional[int] = None, code: Optional[Tensor] = None
    ) -> Tensor:
        """Compute velocity using the block for the given step.

        Args:
            x: [B, N, 3] point positions
            step: which block to use (required, as for TimeVaryingField)
            code: [B, N_z] per-shape code (ignored by NoConditioning)

        Returns:
            [B, N, 3] velocity
        """
        zbar = self.conditioning(x, code)
        features = self.encoding(x)
        h = features if zbar is None else torch.cat([features, zbar], dim=-1)
        return self.blocks[step](h)
