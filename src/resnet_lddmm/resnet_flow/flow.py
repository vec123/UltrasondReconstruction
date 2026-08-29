"""NeuralODEFlow: facade combining a velocity field with forward and inverse integrators."""

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from src.resnet_lddmm.fields.base import VelocityField
from src.resnet_lddmm.integrators import Integrator
from src.resnet_lddmm.trajectory import Trajectory


class NeuralODEFlow(nn.Module):
    """Deformation flow: field + two integrators (forward and inverse).

    Integrates a velocity field to produce a trajectory. Supports both forward
    deformation (source → target) and inverse (target → source).
    """

    def __init__(
        self,
        field: VelocityField,
        direct: Integrator,
        inverse: Integrator,
        num_steps: int = 10,
    ):
        """Initialize flow with field and integrators.

        Args:
            field: velocity field that maps positions to velocities
            direct: integrator for forward pass (source → target)
            inverse: integrator for inverse pass (target → source)
            num_steps: number of Euler steps (default 10)
        """
        super().__init__()
        self.field = field
        self.direct = direct
        self.inv = inverse
        self.num_steps = num_steps

    def forward(self, q0: Tensor, code: Optional[Tensor] = None) -> Trajectory:
        """Integrate forward: initial → deformed.

        Args:
            q0: [B, N, 3] initial positions
            code: [B, N_z] per-shape code (optional)

        Returns:
            Trajectory of forward deformation
        """
        return self.direct.integrate(q0, self.field, code, self.num_steps)

    def inverse(self, qT: Tensor, code: Optional[Tensor] = None) -> Trajectory:
        """Integrate backward: deformed → initial.

        Args:
            qT: [B, N, 3] deformed positions
            code: [B, N_z] per-shape code (optional)

        Returns:
            Trajectory of inverse deformation
        """
        return self.inv.integrate(qT, self.field, code, self.num_steps)


class IdentityFlowWrapper(nn.Module):
    """Wraps a flow but returns identity transformation (no deformation).

    Interface matches NeuralODEFlow exactly:
      - forward(points, code) -> Trajectory (with .end = points unchanged)
      - inverse(points, code) -> Trajectory (identity is self-inverse)
      - field, parameters, state_dict(), load_state_dict()

    Kinetic energy is always 0 (identity has no cost).

    Usage:
        flow = IdentityFlowWrapper(real_flow) if freeze else real_flow

    This wrapper enables "pose-only training" mode: the encoder learns rotation/translation
    without the neural ODE flow deforming the template. Loss is purely Chamfer distance
    between template@pose and augmented_sample.
    """

    def __init__(self, wrapped_flow: NeuralODEFlow):
        """Initialize wrapper around a NeuralODEFlow.

        Args:
            wrapped_flow: NeuralODEFlow instance to wrap
        """
        super().__init__()
        self.wrapped_flow = wrapped_flow
        self.field = wrapped_flow.field  # Expose field for callbacks/logging
        self.num_steps = wrapped_flow.num_steps

    def forward(self, q0: Tensor, code: Optional[Tensor] = None) -> Trajectory:
        """Return identity trajectory (q0 unchanged).

        Args:
            q0: [B, N, 3] initial positions
            code: [B, N_z] per-shape code (optional, ignored)

        Returns:
            Trajectory with:
              - .end = q0 (unchanged)
              - .kinetic_energy() = 0
              - .points[0] = .points[-1] = q0
        """
        B, N, D = q0.shape
        K = self.num_steps

        # Build trajectory: all steps return the same points (identity)
        # points: [K+1, B, N, 3] — same points at every step
        points = q0.unsqueeze(0).expand(K + 1, -1, -1, -1)  # [K+1, B, N, 3]

        # velocities: [K, B, N, 3] — all zeros (no motion)
        velocities = torch.zeros(K, B, N, D, device=q0.device, dtype=q0.dtype)

        # dt = 1/K to match wrapped flow's convention
        dt = 1.0 / K if K > 0 else 1.0

        return Trajectory(points=points, velocities=velocities, dt=dt)

    def inverse(self, qT: Tensor, code: Optional[Tensor] = None) -> Trajectory:
        """Return identity trajectory (qT unchanged).

        Identity is self-inverse: forward and backward are the same.

        Args:
            qT: [B, N, 3] terminal positions
            code: [B, N_z] per-shape code (optional, ignored)

        Returns:
            Trajectory with .end = qT (unchanged)
        """
        B, N, D = qT.shape
        K = self.num_steps

        # Same as forward: identity trajectory
        points = qT.unsqueeze(0).expand(K + 1, -1, -1, -1)  # [K+1, B, N, 3]
        velocities = torch.zeros(K, B, N, D, device=qT.device, dtype=qT.dtype)
        dt = 1.0 / K if K > 0 else 1.0

        return Trajectory(points=points, velocities=velocities, dt=dt)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save wrapped flow state (identity has no learnable params).

        Args:
            destination: dict to populate
            prefix: prefix for keys
            keep_vars: if True, keep Variables (deprecated)

        Returns:
            state_dict with wrapped flow's parameters under "wrapped_flow." prefix
        """
        state = {}
        # Save wrapped flow's state under a prefix
        wrapped_state = self.wrapped_flow.state_dict(destination=None, prefix="", keep_vars=keep_vars)
        for k, v in wrapped_state.items():
            state[f"wrapped_flow.{k}"] = v
        return state

    def load_state_dict(self, state_dict, strict=True):
        """Load wrapped flow state.

        Args:
            state_dict: state dict with "wrapped_flow." prefixed keys
            strict: if True, require exact match

        Returns:
            Incompatible keys info
        """
        # Extract wrapped flow's state (remove prefix)
        wrapped_state = {}
        for k, v in state_dict.items():
            if k.startswith("wrapped_flow."):
                wrapped_state[k[len("wrapped_flow."):]] = v

        if wrapped_state:
            return self.wrapped_flow.load_state_dict(wrapped_state, strict=strict)
        else:
            # No wrapped_flow keys found, load nothing
            return type("", (), {"missing_keys": [], "unexpected_keys": []})()
