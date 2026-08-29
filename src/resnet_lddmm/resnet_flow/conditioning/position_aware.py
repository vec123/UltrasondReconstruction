"""Position-aware conditioning: grid-based features (STEPS T22)."""

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from src.resnet_lddmm.conditioning.base import Conditioning


class PositionAware(Conditioning):
    """Grid-based conditioning with trilinear interpolation.

    Decodes a global code z into a g³ grid of C-dim feature vectors,
    then uses trilinear interpolation (with smooth extrapolation outside [0,1]³)
    to produce per-point features based on point position.
    """

    def __init__(self, n_z: int = 256, g: int = 2, channels: int = 32):
        """Initialize grid-based conditioning.

        Args:
            n_z: code dimension
            g: grid resolution (g³ grid points)
            channels: feature dimension per grid point
        """
        super().__init__()
        self.g = g
        self.dim = channels
        self.to_grid = nn.Linear(n_z, g**3 * channels)

    def forward(self, x: Tensor, code: Optional[Tensor]) -> Optional[Tensor]:
        """Interpolate grid features at point locations.

        Args:
            x: [B, N, 3] point positions in unit cube [0,1]³
            code: [B, N_z] per-shape code

        Returns:
            [B, N, channels] per-point features via trilinear interpolation
        """
        if code is None:
            return None

        B, N, _ = x.shape
        g = self.g

        # Decode code to grid: [B, N_z] -> [B, g, g, g, channels]
        Z = self.to_grid(code).view(B, g, g, g, self.dim)

        # Extract coordinates [B, N, 3]
        u, v, w = x.unbind(-1)

        # Trilinear weights: [B, N, 2] each
        # Negative outside [0,1] for smooth extrapolation
        wu = torch.stack([1 - u, u], -1)  # [B, N, 2]
        wv = torch.stack([1 - v, v], -1)  # [B, N, 2]
        ww = torch.stack([1 - w, w], -1)  # [B, N, 2]

        # Trilinear interpolation via einsum
        # wu: [B, N, 2], Z: [B, g, g, g, C]
        # Result: [B, N, C]
        return torch.einsum("bni,bnj,bnk,bijkc->bnc", wu, wv, ww, Z)
