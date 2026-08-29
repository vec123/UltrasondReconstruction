"""TimeVaryingField: ResNet-LDDMM velocity field (one block per time step)."""

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from src.resnet_lddmm.fields.base import VelocityField
from src.resnet_lddmm.fields.blocks import VelocityBlock, FourierFeatures, mlp, last_linear
from src.resnet_lddmm.conditioning.base import Conditioning, NoConditioning


class TimeVaryingField(VelocityField):
    """ResNet-LDDMM: block l is time step l.
    """

    def __init__(
        self,
        num_blocks: int = 10,
        width: int = 512,
        activation: str = "relu",
        conditioning: Optional[Conditioning] = None,
    ):
        super().__init__()
        self.conditioning = conditioning or NoConditioning()
        in_dim = 3 + self.conditioning.dim
        self.blocks = nn.ModuleList(
            VelocityBlock(in_dim, width, activation) for _ in range(num_blocks)
        )

    def forward(
        self, x: Tensor, step: Optional[int] = None, code: Optional[Tensor] = None
    ) -> Tensor:
        """Compute velocity using the block for the given step.

        Args:
            x: [B, N, 3] point positions
            step: which block to use (must be provided for TimeVaryingField)
            code: [B, N_z] per-shape code (ignored by NoConditioning)

        Returns:
            [B, N, 3] velocity
        """
        # Get per-point conditioning features (None under NoConditioning)
        zbar = self.conditioning(x, code)
        # Concatenate position with conditioning if present
        h = x if zbar is None else torch.cat([x, zbar], dim=-1)
        # Apply the block for this step
        return self.blocks[step](h)


class StationaryField(VelocityField):
    """AD-SVFD: FA-NN(x ⊕ z̄) → FourierFeatures → DF-NN(feats ⊕ z̄) → v.

    One net, reused at every step (step ignored) — ~278k params at defaults.
    STEPS T23.
    """

    def __init__(
        self,
        fa: tuple = (64, 64, 64),
        df: tuple = (256, 256, 256, 256, 256),
        fourier_n_e: int = 3,
        activation: str = "leaky_relu",
        conditioning: Optional[Conditioning] = None,
    ):
        """Initialize stationary field with FA-NN → Fourier → DF-NN pipeline.

        Args:
            fa: layer widths for feature activation MLP
            df: layer widths for deformation field MLP
            fourier_n_e: Fourier encoding parameter (n_e)
            activation: activation function ("relu" | "leaky_relu")
            conditioning: optional conditioning module (default: NoConditioning)
        """
        super().__init__()
        self.conditioning = conditioning or NoConditioning()
        c = self.conditioning.dim

        # FA-NN: [3 + c] -> ... -> fa[-1]
        self.fa = mlp([3 + c, *fa], act=activation)

        # Fourier features on FA output
        self.fourier = FourierFeatures(fourier_n_e)
        fpe_dim = (2 * fourier_n_e + 1) * fa[-1]

        # DF-NN: [fpe_dim + c] -> ... -> 3
        self.df = mlp([fpe_dim + c, *df, 3], act=activation, final_bias=False)

        # Identity at init: zero-init the final layer
        nn.init.zeros_(last_linear(self.df).weight)

    def forward(
        self, x: Tensor, step: Optional[int] = None, code: Optional[Tensor] = None
    ) -> Tensor:
        """Compute velocity via FA-NN → Fourier → DF-NN pipeline.

        Args:
            x: [B, N, 3] point positions
            step: ignored (stationary field uses same velocity at all times)
            code: [B, N_z] per-shape code (ignored by NoConditioning)

        Returns:
            [B, N, 3] velocity
        """
        # Get per-point conditioning features (None under NoConditioning)
        zbar = self.conditioning(x, code)

        # FA input: [x ⊕ z̄]
        h = x if zbar is None else torch.cat([x, zbar], dim=-1)

        # FA-NN → Fourier features
        feats = self.fourier(self.fa(h))

        # DF input: [Fourier(FA) ⊕ z̄]
        if zbar is not None:
            feats = torch.cat([feats, zbar], dim=-1)

        # DF-NN → velocity
        return self.df(feats)
