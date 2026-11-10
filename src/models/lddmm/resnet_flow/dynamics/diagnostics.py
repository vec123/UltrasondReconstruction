"""Diffeomorphism diagnostics (STEPS T19)."""

import torch


def triangle_flips(trajectory, faces):
    """Detect triangle flips by comparing face normals over the trajectory.

    A triangle flip occurs when the normal direction (via cross product) reverses,
    indicating a folding that violates the diffeomorphism guarantee.

    Args:
        trajectory: Trajectory with points [K+1, B, N, 3]
        faces: [F, 3] face indices

    Returns:
        [K+1, B, F] bool tensor: True if face is flipped at that step
    """
    if faces is None or len(faces) == 0:
        return torch.tensor([], dtype=torch.bool)

    points = trajectory.points  # [K+1, B, N, 3]
    K_plus_1, B, N, _ = points.shape
    F = len(faces)

    # Initial normals at step 0
    p0, p1, p2 = points[0, 0, faces[:, 0]], points[0, 0, faces[:, 1]], points[0, 0, faces[:, 2]]
    e1 = p1 - p0
    e2 = p2 - p0
    normals_0 = torch.cross(e1, e2, dim=-1)  # [F, 3]

    flips = []

    for k in range(K_plus_1):
        # Normals at step k
        pk0, pk1, pk2 = points[k, 0, faces[:, 0]], points[k, 0, faces[:, 1]], points[k, 0, faces[:, 2]]
        ek1 = pk1 - pk0
        ek2 = pk2 - pk0
        normals_k = torch.cross(ek1, ek2, dim=-1)  # [F, 3]

        # Dot product: if negative, normal has flipped
        dot_prod = torch.sum(normals_0 * normals_k, dim=-1)  # [F]
        flipped_k = dot_prod < 0

        flips.append(flipped_k)

    return torch.stack(flips, dim=0).unsqueeze(1)  # [K+1, B, F] (broadcast B=1)


def lipschitz_bound(field):
    """Compute a Lipschitz bound via spectral norm of layer weights.

    A ReLU MLP is Lipschitz if all weight matrices have spectral norm ≤ 1.
    We compute the product of spectral norms of all layers as a bound on the
    Lipschitz constant of the composition.

    Args:
        field: VelocityField (TimeVaryingField or StationaryField)

    Returns:
        float: conservative upper bound on Lipschitz constant
    """
    product = 1.0

    # Collect all Linear layers in the field
    for module in field.modules():
        if isinstance(module, torch.nn.Linear):
            # Compute spectral norm (largest singular value)
            u, s, vh = torch.svd(module.weight)
            spectral_norm = s[0].item()
            product *= spectral_norm

    return product


def inverse_residual(trajectory_forward, trajectory_inverse):
    """Measure ‖φ⁻¹(φ(q)) - q‖ per point — the numerical invertibility error.

    Comparing forward trajectory end with inverse trajectory end gives a sense
    of how well the integrator pairs: φ⁻¹ ∘ φ ≈ identity.

    Args:
        trajectory_forward: Trajectory from forward integration
        trajectory_inverse: Trajectory from inverse integration, started at
            trajectory_forward.end

    Returns:
        torch.Tensor: [B, N] per-point distances, in the trajectory's own
        (normalised) length units -- comparable to a loss_distance_handle
        breakpoint, unlike a norm over the whole cloud, which grows with N.
    """
    # Compare endpoints: inverse(forward(q0)) should equal q0
    q_start = trajectory_forward.points[0]  # [B, N, 3]
    q_end_forward_then_inverse = trajectory_inverse.points[-1]  # [B, N, 3]

    return (q_end_forward_then_inverse - q_start).norm(dim=-1)
