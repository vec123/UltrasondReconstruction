"""Position-aware conditioning: grid-based features (STEPS T22)."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from src.resnet_lddmm.resnet_flow.conditioning.base import Conditioning
from src.resnet_lddmm.resnet_flow.conditioning.grid_regularization import trilinear_corners


class PositionAware(Conditioning):
    """Grid-based conditioning with trilinear interpolation.

    Decodes a global code z into a g³ grid of C-dim feature vectors, spread evenly
    over the normalisation domain (nodes at its corners and g-2 more per axis
    between them), then trilinearly interpolates them at each point's position.
    Outside the domain the nearest edge cell's interpolant is extended linearly,
    so the features stay smooth there. With g=2 the grid is the domain's 8
    corners: one cell, the original formulation.

    With direct=True the code IS the grid: it is reshaped into the g³ node
    features instead of decoded by a linear layer, so each code entry belongs to
    one node and acts only on the points around it.
    """

    def __init__(self, n_z: int = 256, g: int = 2, channels: int = 32,
                 domain: Tuple[float, float] = (0.0, 1.0), direct: bool = False):
        """Initialize grid-based conditioning.

        Args:
            n_z: code dimension; must equal g³ * channels when direct
            g: grid resolution (g³ grid points), at least 2
            channels: feature dimension per grid point
            domain: (min, max) of the normalised frame the points live in, per
                axis (runner.CENTERING_DOMAINS); the grid spans exactly this box
            direct: use the code as the grid itself, without the decoding layer
        """
        super().__init__()
        if g < 2:
            raise ValueError(f"position-aware grid needs g >= 2 (one cell), got {g}")
        if direct and n_z != g**3 * channels:
            raise ValueError(f"a direct grid needs n_z = grid³ * grid_channels = {g**3 * channels}, got {n_z}")
        self.g = g
        self.dim = channels
        self.domain = (float(domain[0]), float(domain[1]))
        self.direct = direct
        self.to_grid = None if direct else nn.Linear(n_z, g**3 * channels)

    def forward(self, x: Tensor, code: Optional[Tensor]) -> Optional[Tensor]:
        """Interpolate grid features at point locations.

        Args:
            x: [B, N, 3] point positions in the normalisation domain
            code: [B, N_z] per-shape code

        Returns:
            [B, N, channels] per-point features via trilinear interpolation
        """
        if code is None:
            return None

        B, N, _ = x.shape
        g = self.g

        # Code to grid (decoded, or reshaped when direct): [B, N_z] -> [B, g³, channels], node (i, j, k) at row
        # i*g*g + j*g + k, i/j/k indexing the x/y/z axes.
        Z = (code if self.direct else self.to_grid(code)).view(B, g * g * g, self.dim)

        # The 8 nodes around each point and their trilinear weights; a point outside
        # the domain extrapolates linearly from the nearest edge cell.
        nodes, weights = trilinear_corners(x, g, self.domain)              # [B, N, 8] each
        out = 0
        for c in range(8):
            feats = torch.gather(Z, 1, nodes[..., c].unsqueeze(-1).expand(-1, -1, self.dim))
            out = out + weights[..., c].unsqueeze(-1) * feats
        return out
