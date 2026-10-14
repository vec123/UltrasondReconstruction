"""Trajectory: integrated path with points, velocities, and derived quantities."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


def _energy(velocities: Tensor, dt: float) -> Tensor:
    """½ ∫ mean ||v||² dt: mean over batch and points, sum over steps."""
    return 0.5 * velocities.pow(2).sum(-1).mean(dim=(1, 2)).sum() * dt


@dataclass(frozen=True)
class Trajectory:
    """Result of neural ODE integration: points and velocities along a path."""

    points: Tensor
    """[K+1, B, N, 3] — shape at steps 0 through K (K+1 snapshots)."""

    velocities: Tensor
    """[K, B, N, 3] — velocity at each step during integration (K snapshots)."""

    dt: float
    """Integrator step size (typically 1/K). Single source of truth for all scaling."""

    mode_velocities: Optional[Tensor] = None
    """[K, B, N, 3] — the part of ``velocities`` due to designed deformation modes
    (fields/modes.py), or None when the field has none. Charged by its own,
    discounted cost (losses.kinetic.ModeKineticLoss), not by ``kinetic_energy``."""

    @property
    def end(self) -> Tensor:
        """Terminal shape: points[-1], shape [B, N, 3]."""
        return self.points[-1]

    @property
    def free_velocities(self) -> Tensor:
        """[K, B, N, 3] — the velocities minus the designed modes' part: what the network moved."""
        if self.mode_velocities is None:
            return self.velocities
        return self.velocities - self.mode_velocities

    def kinetic_energy(self) -> Tensor:
        """Total kinetic energy: ½ ∫ ||v(t)||² dt, summed over spatial dimensions.

        Computes: 0.5 × sum over [batch, points] of ||v||² × dt, on
        ``free_velocities`` -- the designed modes' part is priced separately
        (``mode_kinetic_energy``). Returns scalar.
        """
        return _energy(self.free_velocities, self.dt)

    def mode_kinetic_energy(self) -> Tensor:
        """Kinetic energy of the designed modes' part alone; 0 without modes."""
        if self.mode_velocities is None:
            return self.velocities.new_zeros(())
        return _energy(self.mode_velocities, self.dt)
