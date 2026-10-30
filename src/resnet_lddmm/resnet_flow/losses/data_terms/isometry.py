"""Flow regularizer that penalizes non-isometric deformation along the trajectory."""

from typing import Optional
import torch
from torch import Tensor
from src.resnet_lddmm.resnet_flow.dynamics.trajectory import Trajectory
from src.resnet_lddmm.resnet_flow.fields.base import VelocityField

from .base import FlowTerm


class IsometryLoss(FlowTerm):
    """Penalizes non-isometric deformation, per Euler step, along the trajectory.

    The quantity being judged is the jacobian of the STEP MAP the integrator
    actually applies, ``F = I + dt J``, where ``J = dv/dx``. Not J itself: the
    deformation is ``x + dt v(x)``, so an isometric flow is one where F is a
    rotation, which for small dt means the SYMMETRIC part of J vanishes (the
    Killing-field condition). Penalizing J directly would have the objective
    backwards -- it is minimized by a large velocity gradient, and maximal at the
    identity deformation.

    Every penalty is divided by dt^2, which makes it a RATE and therefore
    independent of ``field.num_steps``: the same weight means the same thing when
    K changes.

    Supported loss types (rate form, up to O(dt) corrections):
    - "strain":     ||sym(J)||_F^2 -- the strain-rate tensor; the cheapest, and
                    the linearization of the one below (which it equals times 4,
                    as dt -> 0). EXACTLY zero for any rigid flow, which makes it
                    the default choice.
    - "orthogonal": ||F^T F - I||_F^2 / dt^2 -- the exact condition on the
                    discrete step map, but see the floor below.
    - "det":        ((det F - 1)/dt)^2 -- volume only; permits shear. Same floor.

    The floor: an explicit Euler step of a genuine ROTATION is not exactly
    isometric. For a rotation generator W, F^T F = I - dt^2 W^2 and det F =
    1 + dt^2 |w|^2, so "orthogonal" and "det" both charge a rigid flow for the
    integrator's error -- dt^2 ||W^T W||_F^2 and (dt |w|^2)^2 respectively. That
    is not always negligible: at dt = 0.1 and |w| = 3.7 it measures 3.92 and 1.96,
    which is MORE than those same terms report for a 50%-per-unit-time uniform
    expansion (3.15 and 2.48). Under those two settings a fast rigid rotation
    looks worse than a real distortion. "strain" has no such term.

    NO singular-value decomposition anywhere, deliberately. The natural way to
    write "distance to the closest rotation" is ||F - UV^T||_F, but the SVD
    backward carries 1/(s_i^2 - s_j^2) terms that are infinite when singular
    values repeat -- which is not a rare edge case here but the TARGET state: a
    rigid F has s = (1,1,1), and a zero-initialized field gives s = (0,0,0). That
    formulation returns a NaN gradient exactly where the loss is minimal, which
    then propagates into the field weights and surfaces one step later as
    "linalg.svd: input contained non-finite values". The forms above are
    polynomial in F, so they are smooth everywhere including at the optimum.
    """

    def __init__(self, loss_type: str = "strain", sample_points: int = 64,
                 detach_trajectory: bool = True):
        super().__init__()
        if loss_type not in ("strain", "det", "orthogonal"):
            raise ValueError(f"loss_type must be 'strain', 'det', or 'orthogonal', got {loss_type}")
        self.loss_type = loss_type
        self.sample_points = sample_points
        self.detach_trajectory = detach_trajectory

    def forward(self, trajectory: Trajectory, field: VelocityField,
                code: Optional[Tensor] = None) -> Tensor:
        """Compute isometry penalty using finite differences for jacobian.

        Samples a subset of points to reduce computational cost.

        ``code`` must be the SAME per-shape code the trajectory was integrated
        with. A conditioned field's first layer takes [3 + n_z] inputs, so
        omitting it is a shape error rather than a milder unconditioned
        approximation.

        ``detach_trajectory`` decides how far the gradient reaches. True (default)
        detaches both the trajectory and the code, leaving this a pure FIELD
        regularizer. False lets it reach whatever produced the trajectory -- which
        under ``pose_application="pre"`` includes the POSE, and is the only way the
        "least non-rigid flow" criterion can actually train it. Detaching here is
        not a free choice under that wiring: it silently severs the coupling while
        the loss VALUE still appears to respond to the pose.
        """
        points = trajectory.points.detach() if self.detach_trajectory else trajectory.points
        K = points.shape[0] - 1
        N = points.shape[2]
        if code is not None and self.detach_trajectory:
            code = code.detach()

        # Sample points for efficiency
        n_sample = min(self.sample_points, N)
        indices = torch.randperm(N, device=points.device)[:n_sample]

        all_losses = []

        # For each step
        for k in range(K):
            # [B, n_sample, 3]. The batch dim is KEPT: conditioning broadcasts a
            # per-shape code along it and cannot recover it once flattened away.
            x_sampled = points[k][:, indices, :]

            # Create a wrapper for this step
            def field_fn(x_):
                return field(x_, code=code, step=k)

            # Compute loss for all sampled points at once (vectorized)
            all_losses.append(self._step_penalty(field_fn, x_sampled, trajectory.dt).reshape(-1))

        # Compute final loss as mean of all point losses
        return torch.cat(all_losses).mean()

    def _step_penalty(self, field_fn, x: Tensor, dt: float) -> Tensor:
        """Per-point isometry penalty for ONE Euler step.

        Args:
            x: [B, M, 3] points
            dt: integrator step size, so ``F = I + dt J`` is the step map's jacobian
        Returns:
            [B, M] loss values
        """
        jac = self._jacobians(field_fn, x)                     # [B, M, 3, 3]
        eye = torch.eye(3, device=x.device, dtype=x.dtype)

        # Every statistic below is invariant to transposing F, which is what lets
        # ``_jacobians`` return the transposed layout it finds convenient:
        # ||F^T F - I||_F = ||F F^T - I||_F (same eigenvalues), det F = det F^T,
        # and sym(J) is unchanged by construction.
        if self.loss_type == "strain":
            # ||sym(J)||_F^2 -- no matmul, no dt: already a rate.
            sym = 0.5 * (jac + jac.transpose(-1, -2))
            return (sym ** 2).sum(dim=(-2, -1))

        F = eye + dt * jac
        if self.loss_type == "det":
            return ((torch.det(F) - 1.0) / dt) ** 2

        # "orthogonal": the right Cauchy-Green deviation C - I, which vanishes iff
        # F is orthogonal. One 3x3 matmul per point -- cheaper than an SVD, and
        # differentiable at the optimum, which the SVD form was not.
        C = F.transpose(-1, -2) @ F
        return ((C - eye) ** 2).sum(dim=(-2, -1)) / (dt ** 2)

    @staticmethod
    def _jacobians(field_fn, x: Tensor) -> Tensor:
        """Finite-difference Jacobians of the velocity field at every point.

        Forward differences: 4 field evaluations per point (one base, three
        perturbed), issued as a SINGLE batched call. At these point counts the
        field is launch-bound rather than FLOP-bound -- the [B, M, 3] base call
        costs about as much as the [B, 3M, 3] perturbed one -- so fusing the two
        into [B, 4M, 3] is worth close to a factor of 2 on the whole term.

        The step is sqrt(machine eps), the textbook optimum for a forward
        difference -- ~1.5e-8 in float64, ~3.4e-4 in float32 -- which assumes
        points of order 1, as the normalized frame gives.

        Args:
            x: [B, M, 3] points, the batch dim being the SHAPE index
        Returns:
            [B, M, 3, 3] jacobians. Rows hold the jacobian COLUMNS (dv/dx_j), i.e.
            the block is J^T; every statistic taken from it is transpose-invariant
            (see _step_penalty), so the layout does not matter.
        """
        eps = torch.finfo(x.dtype).eps ** 0.5
        B, M, _ = x.shape

        eye = torch.eye(3, device=x.device, dtype=x.dtype)     # [3, 3]
        # [B, M, 4, 3]: the base point followed by its three perturbations.
        stacked = torch.cat([x.unsqueeze(-2), x.unsqueeze(-2) + eps * eye], dim=-2)

        # Flattened to [B, 4M, 3], NOT [B*4M, 3]: the leading dim has to stay the
        # shape index or a conditioned field broadcasts the wrong code.
        v = field_fn(stacked.reshape(B, 4 * M, 3)).reshape(B, M, 4, 3)
        return (v[..., 1:, :] - v[..., :1, :]) / eps
