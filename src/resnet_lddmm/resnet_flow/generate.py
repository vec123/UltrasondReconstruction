"""Generative sampling from learned code distribution (STEPS T30)."""

import torch
import numpy as np


def sample_codes(Z, n):
    """Sample codes from empirical Gaussian: z ~ N(μ_Z, Σ_Z).

    Computes the mean and covariance of the learned code matrix Z
    (e.g., from AutoDecoderCodes.Z) and draws n samples from the fit Gaussian.

    Args:
        Z: [num_shapes, n_z] learned code matrix (typically from code_source.Z)
        n: number of samples to draw

    Returns:
        [n, n_z] tensor of sampled codes
    """
    # Empirical mean and covariance
    mu = Z.mean(dim=0)  # [n_z]
    cov = torch.cov(Z.T)  # [n_z, n_z]

    # Sample from N(mu, cov)
    # Use Cholesky decomposition for stability
    L = torch.linalg.cholesky(cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype))
    samples = torch.randn(n, Z.shape[1], device=Z.device, dtype=Z.dtype) @ L.T + mu
    return samples


def interpolate_codes(z_a, z_b, ts):
    """Linear interpolation between two codes.

    Args:
        z_a: [n_z] first code
        z_b: [n_z] second code
        ts: [M] or scalar interpolation parameter in [0, 1]

    Returns:
        [M, n_z] or [n_z] interpolated codes
    """
    if isinstance(ts, (int, float)):
        return (1 - ts) * z_a + ts * z_b
    else:
        # ts is a tensor [M]
        ts = ts.view(-1, 1)  # [M, 1]
        return (1 - ts) * z_a + ts * z_b  # [M, n_z]


def synthesize(flow, code, template_points):
    """Generate a new shape by applying inverse flow to template.

    The generative model: start with the template (canonical shape in grid domain),
    apply the learned inverse mapping to deform it according to the code.
    This gives φ_z⁻¹(template) in world coordinates.

    Args:
        flow: NeuralODEFlow instance (frozen parameters)
        code: [1, n_z] or [n_z] code for the shape
        template_points: [1, N, 3] template points in grid domain

    Returns:
        Trajectory from inverse integration (points at each step)
    """
    # Ensure code is [1, n_z]
    if code.dim() == 1:
        code = code.unsqueeze(0)

    # Compute inverse trajectory: φ⁻¹(template)
    trajectory = flow.inverse(template_points, code)
    return trajectory
