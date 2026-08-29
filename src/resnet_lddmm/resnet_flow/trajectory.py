"""Trajectory: integrated path with points, velocities, and derived quantities."""

from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(frozen=True)
class Trajectory:
    """Result of neural ODE integration: points and velocities along a path."""

    points: Tensor
    """[K+1, B, N, 3] — shape at steps 0 through K (K+1 snapshots)."""

    velocities: Tensor
    """[K, B, N, 3] — velocity at each step during integration (K snapshots)."""

    dt: float
    """Integrator step size (typically 1/K). Single source of truth for all scaling."""

    @property
    def end(self) -> Tensor:
        """Terminal shape: points[-1], shape [B, N, 3]."""
        return self.points[-1]

    def kinetic_energy(self) -> Tensor:
        """Total kinetic energy: ½ ∫ ||v(t)||² dt, summed over spatial dimensions.

        Computes: 0.5 × sum over [batch, points] of ||v||² × dt.
        Returns scalar.
        """
        return 0.5 * self.velocities.pow(2).sum(-1).mean(dim=(1, 2)).sum() * self.dt
