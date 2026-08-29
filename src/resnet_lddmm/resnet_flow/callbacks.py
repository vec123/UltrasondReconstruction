"""Diagnostics and export callbacks for ResNetLDDMM (STEP T20)."""

import os
import json
import torch
import numpy as np

from src.resnet_lddmm.learning.callbacks.base import Callback
from src.resnet_lddmm.resnet_flow.diagnostics import jacobian_determinants, triangle_flips, lipschitz_bound
from src.resnet_lddmm.resnet_flow.flow import IdentityFlowWrapper
from src.resnet_lddmm.resnet_flow.io import export_trajectory, export_reference_shapes, export_shape, Shape, FrameTransform


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


class DiagnosticsCallback(Callback):
    """Log diagnostic metrics into the training metrics dict.

    Computes and records:
    - Jacobian determinants (invertibility check)
    - Triangle flips (topology check)
    - Lipschitz bound (Lipschitz continuity check)
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


class EncoderGraphLogger(Callback):
    """Log encoder input graphs with node-type marking (full nodes vs supernodes).

    When using EncoderCodes, saves the point cloud and graph structure that the
    encoder receives at each logging step. Includes a point field "node_type" to
    distinguish regular nodes (0) from supernodes (1).

    Useful for debugging graph construction: visualize node density, dropout effects,
    supernode placement, and graph connectivity.
    """

    def __init__(self, every_n_steps=100):
        super().__init__(every_n_steps)
        self._last_graph_cache = None
        self._last_supergraph_cache = None

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Cache graph and log if due."""
        if pred is None:
            return

        # Store references for potential use in on_log or explicit export
        stepper = ctx.stepper
        if hasattr(stepper, "code_source"):
            code_source = stepper.code_source
            # Check if this is an encoder (has graph_builder and encoder)
            if hasattr(code_source, "graph_builder") and hasattr(code_source, "_last"):
                # Cache for potential use later
                self._last_graph_cache = getattr(code_source, "_last_graph", None)
                self._last_supergraph_cache = getattr(code_source, "_last_supergraph", None)

        if not self._due(step):
            return

        # Export graph and reference shapes if available
        self._current_stepper = stepper  # Store for use in _export_reference_shapes_for_batch
        self._export_encoder_graph(ctx, step, batch, stepper.transform if hasattr(stepper, "transform") else None)

    def _export_encoder_graph(self, ctx, step, batch, transform):
        """Export the last encoder graph with node-type marking and reference shapes."""
        stepper = ctx.stepper
        if not hasattr(stepper, "code_source"):
            return

        code_source = stepper.code_source

        # Check if this is an encoder (has graph_builder)
        if not hasattr(code_source, "graph_builder"):
            return

        # Try to get the cached graph (stored during forward pass)
        graph = getattr(code_source, "_last_graph", None)
        supergraph = getattr(code_source, "_last_supergraph", None)

        if graph is None:
            print(f"[EncoderGraphLogger] Step {step}: No graph cached (encoder not called yet)")
            return

        try:
            # Debug: print batch size and graph shape
            if hasattr(graph, "batch"):
                import torch
                batch_ids = torch.unique(graph.batch).cpu().numpy()
                print(f"[EncoderGraphLogger] Step {step}: batch_ids={sorted(batch_ids)}, graph has {graph.pos.shape[0]} nodes")

            # Create output directory
            out_dir = f"{ctx.log_dir}/encoder_graphs/step_{step}"
            os.makedirs(out_dir, exist_ok=True)

            # Export full graph with per-sample split
            self._export_graph_with_node_types(graph, out_dir, "graph")

            # Export supergraph with bipartite visualization if it exists
            if supergraph is not None:
                self._export_bipartite_supergraph(supergraph, out_dir)

            # Export reference shapes (source and target) for visualization
            if batch is not None:
                self._export_reference_shapes_for_batch(batch, out_dir, transform)

            print(f"[EncoderGraphLogger] Exported encoder graph to {out_dir}")

        except Exception as e:
            print(f"[EncoderGraphLogger] Export failed: {e}")
            import traceback
            traceback.print_exc()

    def _export_graph_with_node_types(self, graph, out_dir, name):
        """Export per-batch graphs as separate VTP files using tested helpers.

        Args:
            graph: PyG Data object with pos, batch, edge_index
            out_dir: output directory
            name: base filename (e.g. "graph")
        """
        try:
            from src.vtk.create import create_polydata_w_lines
            from src.vtk.fields import add_point_field
            from src.vtk.io import save_vtp
            from src.graphs.graphs import get_individual_graph
            import torch
        except ImportError:
            print(f"[EncoderGraphLogger] VTK utilities not available, skipping export")
            return

        if not hasattr(graph, "batch"):
            print(f"[EncoderGraphLogger] Graph has no batch info, skipping export")
            return

        # Get unique batch indices
        batch_ids = torch.unique(graph.batch).cpu().numpy()
        print(f"[EncoderGraphLogger] Graph batch ids: {sorted(batch_ids)}, total nodes: {graph.pos.shape[0]}")

        # Export each sample separately using tested helper function
        for batch_id in batch_ids:
            sample_idx = int(batch_id)

            try:
                # Use tested helper to extract graph with proper edge remapping
                V, E = get_individual_graph(graph, sample_idx)

                # Create VTK polydata with proper line segments
                polydata = create_polydata_w_lines(V, E)

                # Add area if available
                if hasattr(graph, "area") and graph.area is not None:
                    node_mask = (graph.batch == sample_idx)
                    area = graph.area[node_mask].detach().cpu().numpy().flatten()
                    polydata = add_point_field(polydata, area.astype(np.float32), "area")

                # Save per-sample graph
                filepath = os.path.join(out_dir, f"{name}_sample{sample_idx}.vtp")
                save_vtp(polydata, filepath)
                print(f"[EncoderGraphLogger] Saved {name} sample {sample_idx}: {V.shape[0]} nodes, {E.shape[0]} edges")
            except Exception as e:
                print(f"[EncoderGraphLogger] Failed to export graph for sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()

    def _export_reference_shapes_for_batch(self, batch, out_dir, transform):
        """Export template and augmented sample reference shapes from batch for all samples.

        Exports the template (once) and all augmented samples for visualization
        alongside the graph. Saves as template_points.vtp and sample_points_{idx}.vtp.
        """
        try:
            from src.vtk.create import create_polydata
            from src.vtk.io import save_vtp

            template, sample = batch
            stepper = getattr(self, '_current_stepper', None)

            # Use default transform if none provided
            if transform is None:
                transform = FrameTransform(center=torch.zeros(3), scale=torch.tensor(1.0))

            os.makedirs(out_dir, exist_ok=True)

            # Debug: check batch structure
            print(f"[EncoderGraphLogger] Batch types: template={type(template).__name__}, sample={type(sample).__name__}")
            print(f"[EncoderGraphLogger] Batch shapes: template.points={template.points.shape}, sample.points={sample.points.shape}")

            # Export template (canonical reference) - export once
            template_points_norm = template.points[0, :, :]  # [N, 3]
            template_points_world = transform.invert(template_points_norm)
            template_polydata = create_polydata(template_points_world, faces=None)
            template_path = os.path.join(out_dir, "template_points.vtp")
            save_vtp(template_polydata, template_path, binary=True)
            print(f"[EncoderGraphLogger] Exported template_points: {template_points_world.shape[0]} points")

            # Get augmented sample batch if available
            augmented_sample_batch = None
            augmented_sample = None
            if stepper is not None:
                if hasattr(stepper, 'augmented_sample_batch') and stepper.augmented_sample_batch is not None:
                    augmented_sample_batch = stepper.augmented_sample_batch
                    print(f"[EncoderGraphLogger] Found augmented_sample_batch: shape={augmented_sample_batch.points.shape}")
                elif hasattr(stepper, 'augmented_sample') and stepper.augmented_sample is not None:
                    augmented_sample = stepper.augmented_sample
                    print(f"[EncoderGraphLogger] Found augmented_sample: shape={augmented_sample.points.shape}")

            # Export all samples (augmented)
            num_samples = sample.points.shape[0]
            print(f"[EncoderGraphLogger] Batch has {num_samples} samples, augmented_sample_batch: {augmented_sample_batch is not None}, augmented_sample: {augmented_sample is not None}")
            for sample_idx in range(num_samples):
                try:
                    # Get augmented sample points for this sample
                    if augmented_sample_batch is not None:
                        sample_points_norm = augmented_sample_batch.points[sample_idx, :, :]  # [M, 3]
                    elif augmented_sample is not None:
                        # PairRegistration case: only one sample (batch size 1)
                        sample_points_norm = augmented_sample.points[sample_idx, :, :]  # [M, 3]
                    else:
                        # Fallback: use original sample (not augmented)
                        sample_points_norm = sample.points[sample_idx, :, :]

                    # Denormalize and save
                    sample_points_world = transform.invert(sample_points_norm)
                    sample_polydata = create_polydata(sample_points_world, faces=None)
                    sample_path = os.path.join(out_dir, f"sample_points_{sample_idx}.vtp")
                    save_vtp(sample_polydata, sample_path, binary=True)
                    print(f"[EncoderGraphLogger] Exported sample_points_{sample_idx}: {sample_points_world.shape[0]} points")
                except Exception as e:
                    print(f"[EncoderGraphLogger] Failed to export sample_points_{sample_idx}: {e}")
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            print(f"[EncoderGraphLogger] Reference shape export failed: {e}")
            import traceback
            traceback.print_exc()

    def _export_bipartite_supergraph(self, supergraph, out_dir):
        """Export supergraph with node-type marking (regular nodes vs supernodes).

        Uses get_bipartite_graph to show the aggregation structure.
        """
        try:
            from src.vtk.create import create_polydata_w_lines
            from src.vtk.fields import add_point_field
            from src.vtk.io import save_vtp
            from src.graphs.graphs import get_bipartite_graph
            import torch
        except ImportError:
            print(f"[EncoderGraphLogger] VTK utilities not available, skipping supergraph export")
            return

        if not hasattr(supergraph, "batch"):
            print(f"[EncoderGraphLogger] Supergraph has no batch info, skipping export")
            return

        # Get unique batch indices
        batch_ids = torch.unique(supergraph.batch).cpu().numpy()
        print(f"[EncoderGraphLogger] Supergraph batch ids: {sorted(batch_ids)}, total nodes: {supergraph.pos.shape[0]}")

        # Export each sample's bipartite structure
        for batch_id in batch_ids:
            sample_idx = int(batch_id)

            try:
                # Use bipartite helper: merges full nodes + supernodes with aggregation edges
                points, lines, node_type = get_bipartite_graph(supergraph, sample_idx)

                # Create VTK polydata with proper line segments
                polydata = create_polydata_w_lines(points, lines)

                # Add node_type field (0 = regular, 1 = supernode)
                polydata = add_point_field(polydata, node_type, "node_type")

                # Save per-sample supergraph
                filepath = os.path.join(out_dir, f"supergraph_sample{sample_idx}.vtp")
                save_vtp(polydata, filepath)
                print(f"[EncoderGraphLogger] Saved supergraph sample {sample_idx}: {points.shape[0]} nodes, {lines.shape[0]} edges")
            except Exception as e:
                print(f"[EncoderGraphLogger] Failed to export supergraph for sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()


class GradientMonitor(Callback):
    """Monitor and log gradient statistics during training.

    Tracks gradient norms, means, and detects NaN/Inf for each parameter group:
    encoder, flow, and code source. Useful for debugging gradient flow issues.
    """

    def __init__(self, every_n_steps=50):
        super().__init__(every_n_steps)

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Log gradient statistics if due."""
        if not self._due(step):
            return

        stepper = ctx.stepper
        if not hasattr(stepper, '_log_gradient_stats'):
            return

        # Call the stepper's gradient logging method
        stepper._log_gradient_stats()

        # Also store metrics for later analysis
        self._record_gradient_metrics(stepper, metrics, step)

    def _record_gradient_metrics(self, stepper, metrics, step):
        """Store gradient metrics for monitoring."""
        # Encoder stats
        if hasattr(stepper.code_source, 'encoder'):
            encoder_grad_norm = self._compute_grad_norm(stepper.code_source.encoder.parameters())
            metrics[f'grad_norm/encoder'] = encoder_grad_norm

        # Flow stats
        flow_grad_norm = self._compute_grad_norm(stepper.flow.parameters())
        metrics[f'grad_norm/flow'] = flow_grad_norm

        # Code source stats
        code_grad_norm = self._compute_grad_norm(stepper.code_source.parameters())
        metrics[f'grad_norm/code_source'] = code_grad_norm

    @staticmethod
    def _compute_grad_norm(parameters):
        """Compute total gradient norm for parameter group."""
        total_norm = 0.0
        for param in parameters:
            if param.grad is not None:
                total_norm += param.grad.data.norm().item() ** 2
        return total_norm ** 0.5


class NetworkStructureInspector(Callback):
    """Inspect and log the actual network structure once at the start."""

    def __init__(self):
        super().__init__(every_n_steps=0)  # Only run once

    def on_train_start(self, ctx):
        """Print and save detailed network structure."""
        stepper = ctx.stepper
        log_dir = os.path.join(ctx.log_dir, "grad_logs")
        os.makedirs(log_dir, exist_ok=True)

        output = []
        output.append("="*80)
        output.append("NETWORK STRUCTURE INSPECTION")
        output.append("="*80)

        # Flow parameters
        output.append("\n[FLOW PARAMETERS]")
        flow_total = 0
        for name, param in stepper.flow.named_parameters():
            size = param.numel()
            flow_total += size
            output.append(f"  {name}: {param.shape} = {size:,} params")
        output.append(f"  TOTAL FLOW: {flow_total:,} params\n")

        # Code source parameters
        output.append("[CODE SOURCE PARAMETERS]")
        code_total = 0
        for name, param in stepper.code_source.named_parameters():
            size = param.numel()
            code_total += size
            output.append(f"  {name}: {param.shape} = {size:,} params")
        output.append(f"  TOTAL CODE SOURCE: {code_total:,} params\n")

        # Encoder parameters (if exists)
        if hasattr(stepper.code_source, 'encoder'):
            output.append("[ENCODER PARAMETERS]")
            encoder_total = 0
            for name, param in stepper.code_source.encoder.named_parameters():
                size = param.numel()
                encoder_total += size
                output.append(f"  {name}: {param.shape} = {size:,} params")
            output.append(f"  TOTAL ENCODER: {encoder_total:,} params\n")

        output.append("="*80 + "\n")

        # Save to file
        output_text = "\n".join(output)
        save_path = os.path.join(log_dir, "network_structure.txt")
        with open(save_path, 'w') as f:
            f.write(output_text)


class GradientLogger(Callback):
    """Log detailed gradient information to grad_logs folder for analysis.

    Tracks gradient flow, parameter updates, and per-layer statistics.
    Saves to CSV files for post-training analysis.
    """

    def __init__(self, every_n_steps=50):
        super().__init__(every_n_steps)
        self.log_dir = None

    def on_train_start(self, ctx):
        """Create grad_logs directory at start of training."""
        self.log_dir = os.path.join(ctx.log_dir, "grad_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[GradientLogger] Logging to {self.log_dir}")

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Log gradient stats if due."""
        if not self._due(step) or self.log_dir is None:
            return

        stepper = ctx.stepper
        grad_info = self._compute_gradient_info(stepper)
        self._write_log(step, grad_info)

    def _compute_gradient_info(self, stepper):
        """Compute detailed gradient information."""
        info = {
            'flow': self._analyze_param_group(stepper.flow.parameters()),
            'code_source': self._analyze_param_group(stepper.code_source.parameters()),
        }

        # For EncoderCodes, encoder params are identical to code_source params, so skip duplication
        # Only log encoder separately if it has non-encoder params (like embeddings in AutoDecoderCodes)
        if hasattr(stepper.code_source, 'encoder'):
            encoder_params = list(stepper.code_source.encoder.parameters())
            code_source_params = list(stepper.code_source.parameters())

            # Only add encoder if it's not identical to code_source (i.e., has additional non-encoder params)
            if len(encoder_params) < len(code_source_params):
                info['encoder'] = self._analyze_param_group(stepper.code_source.encoder.parameters())

        return info

    @staticmethod
    def _analyze_param_group(parameters):
        """Analyze gradients for a parameter group."""
        params = list(parameters)

        grad_norms = []
        param_norms = []
        has_grad_count = 0
        zero_grad_count = 0
        nan_count = 0
        inf_count = 0
        total_param_count = 0
        num_tensors = len(params)

        for param in params:
            total_param_count += param.numel()
            param_norms.append(param.data.norm().item())

            if param.grad is None:
                continue

            has_grad_count += 1
            grad = param.grad.data
            grad_norm = grad.norm().item()
            grad_norms.append(grad_norm)

            if (grad == 0).all():
                zero_grad_count += 1
            if torch.isnan(grad).any():
                nan_count += 1
            if torch.isinf(grad).any():
                inf_count += 1

        total_param_norm = sum(p**2 for p in param_norms) ** 0.5
        total_grad_norm = sum(g**2 for g in grad_norms) ** 0.5

        update_ratio = total_grad_norm / total_param_norm if total_param_norm > 0 else 0

        return {
            'total_params': total_param_count,
            'num_tensors': num_tensors,
            'has_grad': has_grad_count,
            'zero_grad': zero_grad_count,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'grad_norm': total_grad_norm,
            'param_norm': total_param_norm,
            'update_ratio': update_ratio,
            'avg_grad_norm': sum(grad_norms) / len(grad_norms) if grad_norms else 0,
        }

    def _write_log(self, step, grad_info):
        """Write gradient info to CSV file."""
        import csv

        csv_path = os.path.join(self.log_dir, "gradient_history.csv")

        # Check if file exists to write header
        file_exists = os.path.exists(csv_path)

        with open(csv_path, 'a', newline='') as f:
            fieldnames = ['step']

            # Add fields for each component
            for component in grad_info.keys():
                for key in grad_info[component].keys():
                    fieldnames.append(f'{component}_{key}')

            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            # Build row
            row = {'step': step}
            for component, stats in grad_info.items():
                for key, value in stats.items():
                    row[f'{component}_{key}'] = value

            writer.writerow(row)

        # Also write a text summary for this step
        self._write_text_summary(step, grad_info)

    def _write_text_summary(self, step, grad_info):
        """Write human-readable summary for this step."""
        summary_path = os.path.join(self.log_dir, f"step_{step:06d}.txt")

        with open(summary_path, 'w') as f:
            f.write(f"Gradient Analysis - Step {step}\n")
            f.write("=" * 60 + "\n\n")

            for component, stats in grad_info.items():
                f.write(f"[{component.upper()}]\n")
                f.write(f"  Total parameters: {stats['total_params']:,}\n")
                f.write(f"  Parameter tensors: {stats['num_tensors']}\n")
                f.write(f"  Tensors with gradients: {stats['has_grad']}/{stats['num_tensors']}\n")
                f.write(f"  Tensors with zero gradients: {stats['zero_grad']}\n")
                f.write(f"  NaN detected: {stats['nan_count']}\n")
                f.write(f"  Inf detected: {stats['inf_count']}\n")
                f.write(f"  Gradient norm: {stats['grad_norm']:.4e}\n")
                f.write(f"  Parameter norm: {stats['param_norm']:.4e}\n")
                f.write(f"  Update ratio (grad/param): {stats['update_ratio']:.4e}\n")
                f.write(f"  Avg gradient: {stats['avg_grad_norm']:.4e}\n")
                f.write("\n")


class RequiresGradMonitor(Callback):
    """Monitor requires_grad status of encoder pose tensors.

    Ensures rotation matrix has requires_grad=True so gradients flow back to encoder.
    Saves a detailed report of all tensor requires_grad status to log directory.
    """

    def __init__(self, every_n_steps=100):
        super().__init__(every_n_steps)
        self.log_dir = None

    def on_train_start(self, ctx):
        """Create log directory."""
        self.log_dir = os.path.join(ctx.log_dir, "requires_grad_logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Check and log requires_grad status if due."""
        if not self._due(step):
            return

        stepper = ctx.stepper
        code_source = stepper.code_source if hasattr(stepper, 'code_source') else None
        flow = stepper.flow if hasattr(stepper, 'flow') else None

        report = {}

        # Check encoder pose requires_grad
        if code_source is not None and hasattr(code_source, 'get_pose'):
            rotation, translation = code_source.get_pose()
            report['encoder_pose'] = {
                'rotation_requires_grad': rotation.requires_grad if rotation is not None else 'N/A',
                'rotation_grad_fn': str(rotation.grad_fn) if rotation is not None else 'N/A',
                'translation_requires_grad': translation.requires_grad if translation is not None else 'N/A',
            }

            # ERROR: rotation must require grad!
            if rotation is not None and not rotation.requires_grad:
                print(f"[REQUIRES_GRAD_ERROR] Step {step}: rotation.requires_grad=False!")
                print(f"  This breaks gradient flow to encoder!")
                report['encoder_pose']['ERROR'] = 'rotation.requires_grad=False'

        # Check all encoder parameters
        if code_source is not None and hasattr(code_source, 'encoder'):
            encoder_requires_grad = []
            for name, param in code_source.encoder.named_parameters():
                encoder_requires_grad.append({
                    'name': name,
                    'requires_grad': param.requires_grad,
                    'shape': str(param.shape),
                })
            report['encoder_params'] = encoder_requires_grad

        # Check flow parameters
        if flow is not None:
            flow_requires_grad = []
            for name, param in flow.named_parameters():
                flow_requires_grad.append({
                    'name': name,
                    'requires_grad': param.requires_grad,
                    'shape': str(param.shape),
                })
            report['flow_params'] = flow_requires_grad

        # Save report
        self._save_report(step, report)

    def _save_report(self, step, report):
        """Save requires_grad status report to file."""
        report_path = os.path.join(self.log_dir, f"step_{step:06d}.txt")

        with open(report_path, 'w') as f:
            f.write(f"Requires Grad Status - Step {step}\n")
            f.write("=" * 80 + "\n\n")

            # Encoder pose
            if 'encoder_pose' in report:
                f.write("[ENCODER POSE]\n")
                pose_info = report['encoder_pose']
                for key, value in pose_info.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            # Encoder parameters
            if 'encoder_params' in report:
                f.write("[ENCODER PARAMETERS]\n")
                encoder_params = report['encoder_params']
                requires_grad_count = sum(1 for p in encoder_params if p['requires_grad'])
                f.write(f"  Total: {len(encoder_params)}, with requires_grad: {requires_grad_count}\n\n")
                for p in encoder_params:
                    f.write(f"  {p['name']:<50} requires_grad={p['requires_grad']:<5} shape={p['shape']}\n")
                f.write("\n")

            # Flow parameters
            if 'flow_params' in report:
                f.write("[FLOW PARAMETERS]\n")
                flow_params = report['flow_params']
                requires_grad_count = sum(1 for p in flow_params if p['requires_grad'])
                f.write(f"  Total: {len(flow_params)}, with requires_grad: {requires_grad_count}\n\n")
                for p in flow_params:
                    f.write(f"  {p['name']:<50} requires_grad={p['requires_grad']:<5} shape={p['shape']}\n")
                f.write("\n")


class PoseLogger(Callback):
    """Log encoder-predicted poses (rotation + translation) to JSON.

    Records the estimated rotation matrix and translation vector from the
    encoder at specified cadence. Useful for monitoring pose learning in
    encoder-based registration with use_encoder_pose=true.

    For SO(3) augmentation, translation is not logged since target has zero
    translation. For SE(3) augmentation, both rotation and translation logged.

    Output format:
    {
        "step": int,
        "augmentation": "so3" or "se3",
        "rotation": [[3x3 matrix as list of lists]],
        "translation": [3D vector as list] (only for SE(3)),
        "rotation_det": float (should be ~1 for proper rotation),
        "orthogonality_error": float (should be ~0)
    }
    """

    def __init__(self, every_n_steps=50, augmentation_kind="se3"):
        super().__init__(every_n_steps)
        self.log_dir = None
        self.augmentation_kind = augmentation_kind  # "so3" or "se3"

    def on_train_start(self, ctx):
        """Create pose_logs directory at start of training."""
        self.log_dir = os.path.join(ctx.log_dir, "pose_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[PoseLogger] Logging poses to {self.log_dir}")

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Log pose if due."""
        if not self._due(step) or self.log_dir is None:
            return

        stepper = ctx.stepper
        code_source = stepper.code_source if hasattr(stepper, 'code_source') else None

        # Check if encoder has pose
        if code_source is None or not hasattr(code_source, 'get_pose'):
            return

        rotation, translation = code_source.get_pose()

        # Skip if pose is None
        if rotation is None and translation is None:
            return

        try:
            pose_data = {
                'step': step,
                'augmentation': self.augmentation_kind,
                'rotation': None,
                'translation': None,
                'rotation_det': None,
                'rotation_frobenius_norm': None,
                'v1': None,
                'v2': None,
                'v1_norm': None,
                'v2_norm': None,
            }

            # Log rotation matrix if present
            if rotation is not None:
                R = rotation.detach().cpu().numpy()
                # Handle batch dimension: if [B, 3, 3], take first sample [0, :, :]
                if R.ndim == 3:
                    R = R[0]  # Take first batch element

                pose_data['rotation'] = R.tolist()
                det = np.linalg.det(R)
                pose_data['rotation_det'] = float(det) if np.isscalar(det) else float(det.item())

                # Frobenius norm of (R^T R - I) to check orthonormality
                orthogonality_error = np.linalg.norm(R.T @ R - np.eye(3))
                pose_data['orthogonality_error'] = float(orthogonality_error)

            # Log translation vector ONLY for SE(3) augmentation (SO(3) has zero translation)
            if translation is not None and self.augmentation_kind == "se3":
                t = translation.detach().cpu().numpy()
                # Handle batch dimension: if [B, 3], take first sample [0, :]
                if t.ndim == 2:
                    t = t[0]
                pose_data['translation'] = t.tolist()

            # Log the raw pose vectors (v1, v2) that build the rotation
            if code_source is not None and hasattr(code_source, '_last') and code_source._last is not None:
                enc_output = code_source._last
                if hasattr(enc_output, 'aux') and enc_output.aux and 'v1' in enc_output.aux:
                    v1 = enc_output.aux['v1'].detach().cpu().numpy()
                    v2 = enc_output.aux['v2'].detach().cpu().numpy()
                    # Handle batch dimension
                    if v1.ndim == 2:
                        v1 = v1[0]
                    if v2.ndim == 2:
                        v2 = v2[0]
                    pose_data['v1'] = v1.tolist()
                    pose_data['v2'] = v2.tolist()
                    pose_data['v1_norm'] = float(np.linalg.norm(v1))
                    pose_data['v2_norm'] = float(np.linalg.norm(v2))

            # Save to JSON
            pose_path = os.path.join(self.log_dir, f"step_{step:06d}.json")
            with open(pose_path, 'w') as f:
                json.dump(pose_data, f, indent=2)

            # Append to history CSV
            self._append_pose_history(step, pose_data)

        except Exception as e:
            print(f"[PoseLogger] Failed to log pose at step {step}: {e}")
            import traceback
            traceback.print_exc()

    def _append_pose_history(self, step, pose_data):
        """Append pose data to CSV history for easy plotting."""
        import csv

        csv_path = os.path.join(self.log_dir, "pose_history.csv")
        file_exists = os.path.exists(csv_path)

        # Build fieldnames based on augmentation type
        fieldnames = ['step']

        # Only include translation columns for SE(3) augmentation
        if self.augmentation_kind == "se3":
            fieldnames.extend(['translation_x', 'translation_y', 'translation_z'])

        fieldnames.extend(['rotation_det', 'orthogonality_error', 'v1_norm', 'v2_norm'])
        for i in range(3):
            for j in range(3):
                fieldnames.append(f'rotation_{i}{j}')

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            # Build row
            row = {
                'step': step,
                'rotation_det': pose_data['rotation_det'],
                'orthogonality_error': pose_data['orthogonality_error'],
                'v1_norm': pose_data['v1_norm'],
                'v2_norm': pose_data['v2_norm'],
            }

            # Add translation entries ONLY for SE(3)
            if self.augmentation_kind == "se3":
                row['translation_x'] = None
                row['translation_y'] = None
                row['translation_z'] = None
                if pose_data['translation'] and len(pose_data['translation']) >= 3:
                    row['translation_x'] = pose_data['translation'][0]
                    row['translation_y'] = pose_data['translation'][1]
                    row['translation_z'] = pose_data['translation'][2]

            # Add rotation matrix entries
            if pose_data['rotation']:
                R = pose_data['rotation']
                for i in range(3):
                    for j in range(3):
                        row[f'rotation_{i}{j}'] = R[i][j]

            writer.writerow(row)


def _as_float(value):
    """Best-effort JSON-serialisable scalar, or None when the value is not one.

    The metrics dict is open: callbacks earlier in the list drop their own entries
    into it, and nothing guarantees they are Python floats. Coercing here keeps a
    stray tensor or array from taking down the whole log write.
    """
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class LossLogger(Callback):
    """Log the composed loss and its per-term breakdown to JSON.

    The console shows the total and whatever the verbose callback selects; this
    keeps the whole breakdown, per step, in a form that can be replayed and plotted
    after the run.

    Output per due step, ``loss_logs/step_%06d.json``::

        {
          "step": 500,
          "total": 6.7412,
          "terms": {
            "data":    {"value": 0.0134, "weight": 50.0, "contribution": 0.6700},
            "kinetic": {"value": 6.0712, "weight":  1.0, "contribution": 6.0712}
          },
          "skipped": ["code_reg", "isometry"],
          "sum_of_contributions": 6.7412,
          "residual": 0.0,
          "diagnostics": {"diag/det_min": 0.83, "grad_norm/flow": 1.7e-2}
        }

    ``value`` is the term as the composer received it, UNWEIGHTED -- which is what
    ``breakdown`` carries, and the usual misreading of these logs, since the values
    do not sum to the total. ``contribution = weight * value`` is what actually
    entered the objective, and ``residual`` (total minus their sum) is the check
    that the weights logged here are the weights that were optimised. It should be
    0 up to float error; anything else means the composer and this log disagree.

    ``skipped`` names terms the composer is configured with that produced no value
    this step. That is the normal way a term switches itself off -- an unavailable
    pose, a mode that does not apply -- and it is worth seeing explicitly, because a
    term silently returning None looks exactly like a term that is satisfied.

    History goes to ``loss_logs/loss_history.jsonl``, one record per line. Not a CSV
    (as ``PoseLogger`` uses) because the key set is NOT fixed across steps: terms
    that return None vanish from the breakdown, so a header written at step 0 would
    be wrong the first time a term appears or drops out. A line-delimited record is
    schema-free and still streams into pandas via ``read_json(..., lines=True)``.

    Place this LAST in the callback list. ``metrics`` is one mutable dict shared by
    the whole list, so running last is what lets DiagnosticsCallback's ``diag/*``
    and GradientMonitor's ``grad_norm/*`` land in ``diagnostics`` rather than being
    written a step late.
    """

    def __init__(self, every_n_steps=50, subdir="loss_logs"):
        """Initialize the logger.

        Args:
            every_n_steps: cadence
            subdir: directory under ctx.log_dir to write into
        """
        super().__init__(every_n_steps)
        self.subdir = subdir
        self.log_dir = None

    def on_train_start(self, ctx):
        """Create the loss_logs directory at the start of training."""
        self.log_dir = os.path.join(ctx.log_dir, self.subdir)
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[LossLogger] Logging losses to {self.log_dir}")

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Write this step's record, if due."""
        if not self._due(step) or self.log_dir is None:
            return

        try:
            record = self._record(ctx, step, metrics)

            with open(os.path.join(self.log_dir, f"step_{step:06d}.json"), "w") as f:
                json.dump(record, f, indent=2)

            with open(os.path.join(self.log_dir, "loss_history.jsonl"), "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[LossLogger] Failed to log losses at step {step}: {e}")
            import traceback
            traceback.print_exc()

    def _record(self, ctx, step, metrics):
        """Split the metrics dict into weighted terms and everything else.

        Args:
            ctx: TrainingContext, read for ctx.stepper.composer
            step: current step
            metrics: {"loss": float, <term>: float, <diagnostic>: float}

        Returns:
            JSON-serialisable dict, as documented on the class
        """
        weights = self._weights(ctx)
        total = _as_float(metrics.get("loss"))

        terms, diagnostics = {}, {}
        for key, value in metrics.items():
            if key == "loss":
                continue
            scalar = _as_float(value)
            if key in weights:
                weight = float(weights[key])
                terms[key] = {
                    "value": scalar,
                    "weight": weight,
                    "contribution": None if scalar is None else weight * scalar,
                }
            else:
                diagnostics[key] = scalar

        contributions = [t["contribution"] for t in terms.values()
                         if t["contribution"] is not None]
        summed = float(sum(contributions))

        return {
            "step": step,
            "total": total,
            "terms": terms,
            "skipped": sorted(set(weights) - set(terms)),
            "sum_of_contributions": summed,
            "residual": None if total is None else total - summed,
            "diagnostics": diagnostics,
        }

    @staticmethod
    def _weights(ctx):
        """{term name: weight} from the stepper's composer ({} if unavailable).

        The composer is the only authority on the weights actually applied. Reading
        them back off the config instead would log what was requested rather than
        what was used, and would miss the derived ones -- ``data`` is 1/(2 sigma^2),
        never a config field.
        """
        composer = getattr(getattr(ctx, "stepper", None), "composer", None)
        if composer is None:
            return {}
        return {name: weight for name, weight, _ in composer.terms}


class PoseShapeExporter(Callback):
    """Export the shape before and after the predicted pose, as .vtp.

    PoseLogger records the pose as numbers; this writes the geometry, so the
    alignment can be judged by eye in ParaView rather than inferred from a matrix.

    Per step, per exported shape:

        before.vtp    phi(T)          the deformed template, in the CANONICAL frame,
                                      i.e. before the predicted pose is applied
        after.vtp     phi(T) @ R_hat  the same points with the pose applied -- exactly
                                      what the data term compares against the sample
        target.vtp    the augmented sample, i.e. what `after` should converge onto

    Load all three together and colour by ``point_id`` to follow individual vertices:
    `after` and `target` should coincide when the pose is right, while `before` shows
    how much of the discrepancy was pose rather than deformation.

    Those three are the SUBSAMPLED clouds the loss actually saw -- a fresh random
    ``loss.subsample_M`` points every step, so they carry no connectivity (the
    template's face indices address vertices the draw dropped). With
    ``export_full`` the same three are written again at FULL resolution WITH faces,
    as ``*_full.vtp``, which is what you want to look at rather than a point cloud
    that changes membership every export. See _full_endpoint for when that is
    available without recomputing anything.

    Coordinates are the NORMALIZED frame, not world -- the pose is defined there, and
    mapping back through FrameTransform would fold in a rotation the pose head never
    saw. That is deliberate and differs from TrajectoryExporter, which does export to
    world coordinates.

    Reads what mapping_error recorded for the step it just finished, so it re-does no
    computation and cannot perturb training: `last_fwd_traj` for phi(T) and
    `last_effective_pose` for the pose that was actually applied (post the
    requires_grad filter, so this shows the real transform, not the predicted one).
    """

    def __init__(self, every_n_steps=100, export_shapes=2, subdir="pose_shapes",
                 export_full=True):
        """Initialize the exporter.

        Args:
            every_n_steps: cadence
            export_shapes: how many shapes of the batch to write (0 = all)
            subdir: directory under ctx.log_dir to write into
            export_full: also write the full-resolution input mesh, with faces,
                before and after the pose. Skipped silently when it would require
                recomputing the flow -- see _full_endpoint.
        """
        super().__init__(every_n_steps)
        self.export_shapes = export_shapes
        self.subdir = subdir
        self.export_full = export_full
        self.transform = None  # Set by runner after build(), as for TrajectoryExporter
        self._warned_no_full = False

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Write before/after/target for this step, if due."""
        if not self._due(step):
            return

        stepper = ctx.stepper
        mapping_error = getattr(stepper, "mapping_error", None)
        if mapping_error is None or getattr(mapping_error, "last_fwd_traj", None) is None:
            return

        rotation, translation = getattr(mapping_error, "last_effective_pose", (None, None))
        if rotation is None and translation is None:
            return  # no pose was applied; before and after would be identical

        from src.resnet_lddmm.resnet_flow.losses.terms import apply_pose
        from src.vtk.create import create_polydata
        from src.vtk.fields import add_point_field
        from src.vtk.io import save_vtp

        before = mapping_error.last_fwd_traj.end.detach()
        after = apply_pose(before, rotation, translation).detach()
        target = self._target_points(stepper, mapping_error)

        count = before.shape[0] if self.export_shapes == 0 else min(self.export_shapes,
                                                                   before.shape[0])
        identity = FrameTransform.identity()
        world = self.transform if self.transform is not None else identity

        # Both spaces: `normalized` is where the pose is defined and is the one to
        # judge alignment in; `world` is comparable with the input meshes.
        for space, transform in (("normalized", identity), ("world", world)):
            out_dir = os.path.join(ctx.log_dir, self.subdir, f"step_{step}", space)
            os.makedirs(out_dir, exist_ok=True)
            for b in range(count):
                clouds = {"before": before[b], "after": after[b]}
                if target is not None and b < target.shape[0]:
                    clouds["target"] = target[b]
                for name, points in clouds.items():
                    polydata = create_polydata(transform.invert(points), faces=None)
                    ids = np.arange(points.shape[0], dtype=np.float32)
                    polydata = add_point_field(polydata, ids, field_name="point_id")
                    save_vtp(polydata, os.path.join(out_dir, f"shape_{b}_{name}.vtp"),
                             binary=True)

            if self.export_full:
                self._export_full_meshes(stepper, mapping_error, batch, out_dir,
                                         transform, rotation, translation)

        print(f"[PoseShapeExporter] step {step}: wrote before/after/target for "
              f"{count} shape(s) in normalized/ and world/")

    def _export_full_meshes(self, stepper, mapping_error, batch, out_dir, transform,
                            rotation, translation):
        """Write the full-resolution template and sample, with connectivity.

        The subsampled clouds above are a different random subset every export, so
        they cannot be compared across steps and cannot carry faces. These can be
        both: the vertex set is the whole input mesh, so its face indices are valid
        and the same vertex is the same vertex at every step.

        Only shape 0 -- pair mode has one template, and a cohort's full meshes would
        multiply the written volume by the batch for no extra information about the
        pose, which is shared geometry-wide.

        Args:
            out_dir: this space's directory
            transform: the space's FrameTransform, applied via invert()
            rotation, translation: the pose that was actually applied
        """
        from src.resnet_lddmm.resnet_flow.losses.terms import apply_pose

        template, sample = batch
        before = self._full_endpoint(stepper, mapping_error, template)
        if before is None:
            return

        after = apply_pose(before, rotation, translation).detach()
        meshes = {
            "before_full": (before[0], getattr(template, "faces", None)),
            "after_full": (after[0], getattr(template, "faces", None)),
            "target_full": (sample.points[0], getattr(sample, "faces", None)),
        }
        for name, (points, faces) in meshes.items():
            # export_shape drops connectivity that does not address these exact
            # vertices, so a template whose faces went missing degrades to a cloud
            # rather than writing a mesh out of the wrong indices.
            export_shape(points.detach(), faces, transform,
                         os.path.join(out_dir, f"shape_0_{name}.vtp"), label=name)

    def _full_endpoint(self, stepper, mapping_error, template):
        """phi(T) at FULL resolution, or None when it cannot be had for free.

        Three cases, in order:

        1. ``loss.save_full: true`` already computed it -- use it.
        2. The flow is frozen at the identity, so phi(T_full) IS T_full. Exact and
           costs nothing, which is the rigid-registration case: the pose is then
           the entire model and the full posed mesh is the actual deliverable.
        3. A live flow without save_full. Producing it would mean an extra
           integration of the whole mesh from inside a callback -- real compute,
           silently added to every export. Skipped, with one message saying which
           flag turns it on.

        Also skipped under ``pose_application='pre'`` with a live flow: there the
        pose is applied BEFORE the flow, so the caller's ``apply_pose(phi(T), R)``
        would be applying it a second time. Under a frozen flow the two orders
        coincide, so that case is still exported.
        """
        traj_full = getattr(mapping_error, "last_fwd_traj_full", None)
        frozen = isinstance(getattr(stepper, "flow", None), IdentityFlowWrapper)

        if getattr(mapping_error, "pose_application", "post") == "pre" and not frozen:
            return None

        if traj_full is not None:
            return traj_full.end.detach()
        if frozen:
            return template.points.detach()

        if not self._warned_no_full:
            self._warned_no_full = True
            print("[PoseShapeExporter] full-resolution meshes need the full "
                  "trajectory; set loss.save_full: true (or freeze the flow) to "
                  "get *_full.vtp alongside the subsampled clouds.")
        return None

    @staticmethod
    def _target_points(stepper, mapping_error):
        """The augmented sample the prediction is compared against, or None.

        Prefers what mapping_error actually used (subsampled, so it lines up with
        `after` point-for-point); falls back to the stepper's stored augmented batch,
        whose attribute name differs between the pair and cohort steppers.
        """
        points = getattr(mapping_error, "last_sample_points", None)
        if points is not None:
            return points.detach()
        for attribute in ("augmented_sample_batch", "augmented_sample"):
            sample = getattr(stepper, attribute, None)
            if sample is not None and getattr(sample, "points", None) is not None:
                return sample.points.detach()
        return None
