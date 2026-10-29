"""Trajectory: integrated path with points, velocities, and derived quantities."""

from dataclasses import dataclass, field
from typing import Dict

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

    parts: Dict[str, Tensor] = field(default_factory=dict)
    """name -> [K, B, N, 3], the priced parts of ``velocities`` (fields/base.py
    split_velocity): designed modes, a frozen prior, a symmetric part. Each is
    charged at its own price (losses.kinetic.PartKineticLoss), not by
    ``kinetic_energy``. Empty for a plain field."""

    @property
    def end(self) -> Tensor:
        """Terminal shape: points[-1], shape [B, N, 3]."""
        return self.points[-1]

    @property
    def free_velocities(self) -> Tensor:
        """[K, B, N, 3] — the velocities minus the priced parts: what is charged at kinetic_weight."""
        if not self.parts:
            return self.velocities
        return self.velocities - sum(self.parts.values())

    def kinetic_energy(self) -> Tensor:
        """Total kinetic energy: ½ ∫ ||v(t)||² dt, summed over spatial dimensions.

        Computes: 0.5 × sum over [batch, points] of ||v||² × dt, on
        ``free_velocities`` -- the priced parts are charged separately
        (``part_kinetic_energy``). Returns scalar.
        """
        return _energy(self.free_velocities, self.dt)

    def part_kinetic_energy(self, name: str) -> Tensor:
        """Kinetic energy of the priced part ``name`` alone; 0 when the field has no such part."""
        if name not in self.parts:
            return self.velocities.new_zeros(())
        return _energy(self.parts[name], self.dt)
