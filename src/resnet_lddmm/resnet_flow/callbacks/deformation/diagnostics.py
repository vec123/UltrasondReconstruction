"""DiagnosticsCallback: how well-behaved the deformation is - Jacobians, triangle flips,
Lipschitz bound, inverse residual - logged as diag/* metrics."""

import torch

from src.resnet_lddmm.learning.callbacks.base import Callback
from src.resnet_lddmm.resnet_flow.diagnostics import jacobian_determinants, triangle_flips, lipschitz_bound, inverse_residual


class DiagnosticsCallback(Callback):
    """Log diagnostic metrics into the training metrics dict.

    Computes and records:
    - Jacobian determinants (invertibility check)
    - Triangle flips (topology check)
    - Lipschitz bound (Lipschitz continuity check)
    - Inverse residual ‖φ⁻¹(φ(q)) - q‖ (inverse-integrator accuracy)
    """

    def __init__(self, every_n_steps=100):
        super().__init__(every_n_steps)

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Compute and log diagnostics if due."""
        if not self._due(step):
            return

        if pred is None:
            return

        traj = pred
        template, _ = batch
        faces = template.faces if hasattr(template, "faces") else None

        # Log subsampling info
        try:
            stepper = ctx.stepper
            mapping_error = stepper.mapping_error if hasattr(stepper, "mapping_error") else None
            if mapping_error is not None:
                if hasattr(mapping_error, "last_full_vertices") and mapping_error.last_full_vertices is not None:
                    metrics["diag/vertices_full"] = float(mapping_error.last_full_vertices)
                if hasattr(mapping_error, "last_subsample_vertices") and mapping_error.last_subsample_vertices is not None:
                    metrics["diag/vertices_subsampled"] = float(mapping_error.last_subsample_vertices)
        except Exception:
            pass

        # Jacobian determinants: should be ≥ 0 (positive measure for diffeomorphism)
        try:
            dets = jacobian_determinants(traj, faces)
            min_det = dets.min().item()
            metrics["diag/det_min"] = min_det
            metrics["diag/det_mean"] = dets.mean().item()
        except Exception:
            pass

        # Triangle flips: should be zero (no folds)
        if faces is not None and len(faces) > 0:
            try:
                flips = triangle_flips(traj, faces)
                num_flips = flips.sum().item()
                metrics["diag/flips_count"] = float(num_flips)
            except Exception:
                pass

        # Lipschitz bound: inferred from network weights
        try:
            flow = ctx.stepper.flow if hasattr(ctx.stepper, "flow") else None
            if flow is not None:
                bound = lipschitz_bound(flow)
                metrics["diag/lipschitz_bound"] = bound
        except Exception:
            pass

        # Inverse residual: ModifiedEuler only approximates φ⁻¹, and under
        # direction: bidirectional the backward data term trains THROUGH it, so its
        # accuracy is watched rather than assumed. Same units as a breakpoint.
        try:
            code = getattr(ctx.stepper.mapping_error, "last_code", None)
            with torch.no_grad():
                back = ctx.stepper.flow.inverse(traj.end.detach(), code)
            dist = inverse_residual(traj, back)
            metrics["diag/inverse_residual_mean"] = dist.mean().item()
            metrics["diag/inverse_residual_max"] = dist.max().item()
        except Exception:
            pass
