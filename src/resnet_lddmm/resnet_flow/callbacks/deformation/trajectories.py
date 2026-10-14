"""TrajectoryExporter: the deformation's path, exported as .vtp snapshots during training."""

import torch

from src.resnet_lddmm.learning.callbacks.base import Callback
from src.resnet_lddmm.resnet_flow.io import export_trajectory, export_reference_shapes, Shape, FrameTransform


class TrajectoryExporter(Callback):
    """Export predicted trajectories to VTP files for visualization.

    Calls io.export_trajectory() at its cadence to save deformed shapes
    at each integration step, with velocity fields attached.
    Supports selective export in cohort mode via export_shapes and export_strategy.
    """

    def __init__(self, every_n_steps=100, export_shapes=0, export_strategy="sequential", rng=None):
        super().__init__(every_n_steps)
        self.transform = None  # Set by runner after build()
        self.export_shapes = export_shapes  # 0 = all shapes
        self.export_strategy = export_strategy  # sequential | random
        self.rng = rng  # torch.Generator for random selection

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Export trajectory if due, in BOTH coordinate spaces.

        world/       denormalised, comparable with the input meshes
        normalized/  the frame the model actually operates in -- where the pose is
                     defined, so this is the one to read when judging alignment

        The same geometry twice, differing only by the affine frame transform, so a
        discrepancy between the two directories means the transform is wrong rather
        than the model.
        """
        if not self._due(step):
            return

        identity = FrameTransform.identity()
        world = self.transform if self.transform is not None else identity
        for space, transform in (("world", world), ("normalized", identity)):
            self._export_space(ctx, step, batch, pred, transform,
                               f"{ctx.log_dir}/trajectories/step_{step}/{space}")

    def _export_space(self, ctx, step, batch, pred, transform, base_out_dir):
        """Write one coordinate space's worth of trajectory files.

        Args:
            transform: applied via invert(); pass an identity transform to stay in
                the normalised frame
            base_out_dir: destination for this space
        """
        # Trajectory is in the prediction (first return value from stepper.train_step)
        if pred is None:
            print(f"[TrajectoryExporter] Step {step}: pred is None, skipping")
            return

        # pred is a Trajectory object (from stepper.train_step return)
        traj = pred

        # Get batch geometry for faces (optional—point clouds work without faces)
        # template is canonical reference (where flow starts), sample is the augmented input
        template, sample = batch
        template_faces = template.faces if hasattr(template, "faces") and template.faces is not None else None
        sample_faces = sample.faces if hasattr(sample, "faces") and sample.faces is not None else None

        try:
            # Extract number of shapes in batch (B dimension)
            num_shapes = traj.points.shape[1]

            # Determine which shapes to export
            if self.export_shapes == 0:
                # Export all shapes
                shape_indices = list(range(num_shapes))
            else:
                # Export subset
                num_to_export = min(self.export_shapes, num_shapes)
                if self.export_strategy == "random":
                    shape_indices = torch.randperm(num_shapes, generator=self.rng)[:num_to_export].tolist()
                else:  # sequential
                    shape_indices = list(range(num_to_export))

            # Export forward trajectory for selected shapes
            for shape_idx in shape_indices:
                fwd_dir = f"{base_out_dir}/forward/shape_{shape_idx}"

                # Export template and shape-specific sample to trajectory directory
                try:
                    # Extract template (canonical reference): template.points is [1, N, 3]
                    if isinstance(template.faces, list) and len(template.faces) > 0:
                        template_faces = template.faces[0]
                    elif isinstance(template.faces, list):
                        template_faces = None  # Empty face list
                    else:
                        template_faces = template.faces
                    template_slice = Shape(
                        points=template.points[0:1, :, :],
                        faces=template_faces,
                        weights=template.weights[0:1, :] if hasattr(template, 'weights') and template.weights is not None else None,
                    )
                    # Extract sample for this shape (augmented input): sample.points is [B, M, 3]
                    if isinstance(sample.faces, list) and shape_idx < len(sample.faces):
                        sample_faces = sample.faces[shape_idx]
                    elif isinstance(sample.faces, list):
                        sample_faces = None  # Empty or out-of-range
                    else:
                        sample_faces = sample.faces
                    sample_slice = Shape(
                        points=sample.points[shape_idx:shape_idx+1, :, :],
                        faces=sample_faces,
                        weights=sample.weights[shape_idx:shape_idx+1, :] if hasattr(sample, 'weights') and sample.weights is not None else None,
                    )
                    export_reference_shapes(template_slice, sample_slice, transform, fwd_dir)
                except Exception as e:
                    print(f"[TrajectoryExporter] Reference shape export failed for shape {shape_idx}: {e}")

                # Slice trajectory to single shape: [K+1, 1, N, 3]
                traj_slice = type(traj)(
                    points=traj.points[:, shape_idx:shape_idx+1, :, :],
                    velocities=traj.velocities[:, shape_idx:shape_idx+1, :, :],
                    dt=traj.dt
                )
                # Get template faces (single template for all shapes)
                shape_faces = template_faces[0] if (isinstance(template_faces, list) and len(template_faces) > 0) else template_faces
                print(f"[TrajectoryExporter] Exporting forward trajectory (shape {shape_idx}) to {fwd_dir}")
                export_trajectory(traj_slice, shape_faces, transform, fwd_dir)
            print(f"[TrajectoryExporter] Successfully exported {len(shape_indices)} forward trajectories")

            # Export full forward trajectory if save_full is enabled
            stepper = ctx.stepper
            mapping_error = stepper.mapping_error if hasattr(stepper, "mapping_error") else None
            if mapping_error is not None and hasattr(mapping_error, "last_fwd_traj_full") and mapping_error.last_fwd_traj_full is not None:
                traj_full = mapping_error.last_fwd_traj_full
                for shape_idx in shape_indices:
                    # Slice full trajectory to single shape: [K+1, 1, N, 3]
                    traj_full_slice = type(traj_full)(
                        points=traj_full.points[:, shape_idx:shape_idx+1, :, :],
                        velocities=traj_full.velocities[:, shape_idx:shape_idx+1, :, :],
                        dt=traj_full.dt
                    )
                    shape_faces = template_faces[0] if (isinstance(template_faces, list) and len(template_faces) > 0) else template_faces
                    fwd_full_dir = f"{base_out_dir}/forward_full/shape_{shape_idx}"
                    print(f"[TrajectoryExporter] Exporting full forward trajectory (shape {shape_idx}) to {fwd_full_dir}")
                    export_trajectory(traj_full_slice, shape_faces, transform, fwd_full_dir)
                print(f"[TrajectoryExporter] Successfully exported {len(shape_indices)} full forward trajectories")

            # Export backward trajectory if bidirectional mode
            if hasattr(stepper, "backward_traj") and stepper.backward_traj is not None:
                bwd_traj = stepper.backward_traj
                for shape_idx in shape_indices:
                    bwd_dir = f"{base_out_dir}/backward/shape_{shape_idx}"

                    # For backward, flow goes from sample back to template
                    try:
                        # Extract sample for this shape: sample.points is [B, M, 3]
                        sample_faces = sample.faces[shape_idx] if (isinstance(sample.faces, list) and shape_idx < len(sample.faces)) else sample.faces
                        sample_slice = Shape(
                            points=sample.points[shape_idx:shape_idx+1, :, :],
                            faces=sample_faces,
                            weights=sample.weights[shape_idx:shape_idx+1, :] if hasattr(sample, 'weights') and sample.weights is not None else None,
                        )
                        # Extract template (always a single reference): template.points is [1, N, 3]
                        template_faces = template.faces[0] if (isinstance(template.faces, list) and len(template.faces) > 0) else template.faces
                        template_slice = Shape(
                            points=template.points[0:1, :, :],
                            faces=template_faces,
                            weights=template.weights[0:1, :] if hasattr(template, 'weights') and template.weights is not None else None,
                        )
                        # Labelled by IDENTITY, not by role: template_points.vtp holds
                        # the template in both directories, so a file can be loaded
                        # without checking which one it came from. The flow direction is
                        # already carried by the forward/ vs backward/ directory name.
                        # (This writes reference clouds only; it does not touch the flow,
                        # which still runs template -> sample forward and inverts from the
                        # sample for the backward trajectory.)
                        export_reference_shapes(template_slice, sample_slice, transform, bwd_dir)
                    except Exception as e:
                        print(f"[TrajectoryExporter] Reference shape export failed for backward shape {shape_idx}: {e}")

                    # Slice backward trajectory: [K+1, 1, N, 3]
                    bwd_traj_slice = type(bwd_traj)(
                        points=bwd_traj.points[:, shape_idx:shape_idx+1, :, :],
                        velocities=bwd_traj.velocities[:, shape_idx:shape_idx+1, :, :],
                        dt=bwd_traj.dt
                    )
                    # Get sample faces (sample_faces is a list)
                    smp_faces = sample_faces[shape_idx] if (isinstance(sample_faces, list) and shape_idx < len(sample_faces)) else sample_faces
                    print(f"[TrajectoryExporter] Exporting backward trajectory (shape {shape_idx}) to {bwd_dir}")
                    export_trajectory(bwd_traj_slice, smp_faces, transform, bwd_dir)
                print(f"[TrajectoryExporter] Successfully exported {len(shape_indices)} backward trajectories")

                # Export full backward trajectory if save_full is enabled
                if mapping_error is not None and hasattr(mapping_error, "last_bwd_traj_full") and mapping_error.last_bwd_traj_full is not None:
                    bwd_traj_full = mapping_error.last_bwd_traj_full
                    for shape_idx in shape_indices:
                        # Slice full backward trajectory: [K+1, 1, N, 3]
                        bwd_traj_full_slice = type(bwd_traj_full)(
                            points=bwd_traj_full.points[:, shape_idx:shape_idx+1, :, :],
                            velocities=bwd_traj_full.velocities[:, shape_idx:shape_idx+1, :, :],
                            dt=bwd_traj_full.dt
                        )
                        smp_faces = sample_faces[shape_idx] if (isinstance(sample_faces, list) and shape_idx < len(sample_faces)) else sample_faces
                        bwd_full_dir = f"{base_out_dir}/backward_full/shape_{shape_idx}"
                        print(f"[TrajectoryExporter] Exporting full backward trajectory (shape {shape_idx}) to {bwd_full_dir}")
                        export_trajectory(bwd_traj_full_slice, smp_faces, transform, bwd_full_dir)
                    print(f"[TrajectoryExporter] Successfully exported {len(shape_indices)} full backward trajectories")
        except Exception as e:
            print(f"[TrajectoryExporter] Export failed: {e}")
            import traceback
            traceback.print_exc()
