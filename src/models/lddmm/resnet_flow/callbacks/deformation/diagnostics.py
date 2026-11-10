"""DiagnosticsCallback: how well-behaved the deformation is - triangle flips,
Lipschitz bound, inverse residual - logged as diag/* metrics."""

import torch

from src.models.lddmm.learning.callbacks.base import Callback
from src.models.lddmm.resnet_flow.dynamics.diagnostics import triangle_flips, lipschitz_bound, inverse_residual


class DiagnosticsCallback(Callback):
    """Log diagnostic metrics into the training metrics dict.

    Computes and records:
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
        if isinstance(faces, list):  # cohort batches carry one entry per template
            faces = faces[0] if len(faces) == 1 else None
        # A subsampled trajectory doesn't carry every vertex the faces index. On
        # CPU that indexing raises (and is swallowed below); on CUDA it is a
        # device-side assert that kills the context, so skip the face terms.
        if faces is not None and len(faces) > 0 and int(faces.max()) >= traj.points.shape[-2]:
            faces = None

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
