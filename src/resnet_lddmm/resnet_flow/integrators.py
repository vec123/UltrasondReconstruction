"""ODE integrators for neural flows."""

import abc
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from src.resnet_lddmm.resnet_flow.fields.base import VelocityField
from src.resnet_lddmm.resnet_flow.trajectory import Trajectory


def _velocity(field: VelocityField, q: Tensor, step: int, code: Optional[Tensor]):
    """(velocity, its designed-mode part or None) -- see fields/modes.py ModalField.split."""
    split = getattr(field, "split", None)
    if split is None:
        return field(q, step=step, code=code), None
    return split(q, step=step, code=code)


def _trajectory(points, velocities, modal, dt) -> Trajectory:
    return Trajectory(torch.stack(points), torch.stack(velocities), dt,
                      torch.stack(modal) if modal and modal[0] is not None else None)


class Integrator(abc.ABC):
    """Abstract integrator: steps a velocity field to produce a trajectory."""

    @abc.abstractmethod
    def integrate(
        self,
        q0: Tensor,
        field: VelocityField,
        code: Optional[Tensor],
        num_steps: int,
    ) -> Trajectory:
        """Integrate the velocity field from initial position.

        Args:
            q0: [B, N, 3] initial positions
            field: velocity field
            code: [B, N_z] per-shape code (or None)
            num_steps: number of Euler steps

        Returns:
            Trajectory with points [K+1, B, N, 3] and velocities [K, B, N, 3]
        """


class ForwardEuler(Integrator):
    """Forward Euler integration: q_{k+1} = q_k + dt * v_k.

    Time step dt = 1/num_steps is computed here — the sole owner of this invariant.
    """

    def integrate(
        self,
        q0: Tensor,
        field: VelocityField,
        code: Optional[Tensor],
        num_steps: int,
    ) -> Trajectory:
        """Integrate field from q0 through num_steps forward Euler steps.

        Args:
            q0: [B, N, 3] initial positions
            field: velocity field
            code: [B, N_z] per-shape code (or None)
            num_steps: number of steps; dt = 1/num_steps

        Returns:
            Trajectory containing integrated points and velocities
        """
        dt = 1.0 / num_steps
        q = q0
        points = [q0]
        velocities = []
        modal = []

        for k in range(num_steps):
            v, v_mode = _velocity(field, q, k, code)
            q = q + dt * v
            points.append(q)
            velocities.append(v)
            modal.append(v_mode)

        return _trajectory(points, velocities, modal, dt)


class ModifiedEuler(Integrator):
    """Modified Euler (implicit backward step with Picard iteration).

    Explicit approximation of the implicit backward Euler that would exactly
    invert ForwardEuler. Approximation quality is WATCHED, not assumed:
    diagnostics.inverse_residual is a logged metric (Plan §11).

    Scheme (from AD-SVFD Eq. `inverse mapping modified euler`):
        x^(k-1) = x^(k) - dt * F(x^(k) - dt * F(x^(k), z), z)

    For time-varying fields, steps are processed in REVERSED order (K-1..0)
    to correctly invert the forward pass which used steps in forward order.
    """

    def integrate(
        self,
        qT: Tensor,
        field: VelocityField,
        code: Optional[Tensor],
        num_steps: int,
    ) -> Trajectory:
        """Integrate backward from qT through num_steps modified Euler steps.

        Args:
            qT: [B, N, 3] terminal (deformed) positions
            field: velocity field
            code: [B, N_z] per-shape code (or None)
            num_steps: number of steps; dt = 1/num_steps

        Returns:
            Trajectory of backward integration (reversed step order)
        """
        dt = 1.0 / num_steps
        q = qT
        points = [qT]
        velocities = []
        modal = []

        # Reversed step order for time-varying fields: K-1, K-2, ..., 0
        for k in reversed(range(num_steps)):
            # Predictor step: tentative q at previous time
            v_hat = field(q, step=k, code=code)
            q_pred = q - dt * v_hat

            # Corrector step: velocity at predicted location (Picard iteration)
            v, v_mode = _velocity(field, q_pred, k, code)
            q = q - dt * v

            points.append(q)
            velocities.append(v)
            modal.append(v_mode)

        return _trajectory(points, velocities, modal, dt)
