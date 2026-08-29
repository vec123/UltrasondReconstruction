"""Conditioning strategies: concatenation and FiLM-style modulation."""

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from src.resnet_lddmm.conditioning.base import Conditioning


class ConcatConditioning(Conditioning):
    """Simple concatenation: broadcast code to every point.

    Per-point features are the code repeated N times.
    """

    def __init__(self, n_z: int = 256):
        super().__init__()
        self.dim = n_z

    def forward(self, x: Tensor, code: Optional[Tensor]) -> Optional[Tensor]:
        """Broadcast code to every point.

        Args:
            x: [B, N, 3] point positions
            code: [B, N_z] per-shape code

        Returns:
            [B, N, N_z] per-point features (code broadcast to each point)
        """
        if code is None:
            return None
        B, N, _ = x.shape
        # Broadcast: code [B, N_z] -> [B, N, N_z]
        return code.unsqueeze(1).expand(B, N, -1)


class FiLMConditioning(Conditioning):
    """Feature-wise Linear Modulation: code → affine transform of input features.

    Learns two networks: code → γ (gain) and code → β (bias).
    Per-point features: γ ⊙ position + β.
    """

    def __init__(self, n_z: int = 256, output_dim: int = 3):
        """Initialize FiLM conditioning.

        Args:
            n_z: code dimension
            output_dim: dimension to modulate (typically 3 for positions)
        """
        super().__init__()
        self.dim = output_dim
        self.gamma_net = nn.Linear(n_z, output_dim)
        self.beta_net = nn.Linear(n_z, output_dim)

    def forward(self, x: Tensor, code: Optional[Tensor]) -> Optional[Tensor]:
        """Modulate input via learned affine transform.

        Args:
            x: [B, N, 3] point positions
            code: [B, N_z] per-shape code

        Returns:
            [B, N, output_dim] modulated features: γ ⊙ x + β
        """
        if code is None:
            return None

        B, N, D = x.shape
        code_exp = code.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)

        # Compute γ and β from code
        gamma = self.gamma_net(code_exp).view(B, N, self.dim)
        beta = self.beta_net(code_exp).view(B, N, self.dim)

        # Modulate: γ ⊙ x + β
        if self.dim == D:
            return gamma * x + beta
        else:
            x_mod = x[..., :self.dim]
            return gamma * x_mod + beta
