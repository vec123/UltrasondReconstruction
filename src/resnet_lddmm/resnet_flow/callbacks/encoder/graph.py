"""EncoderGraphLogger: the encoder's input graph, exported for inspection."""

import os
import torch
import numpy as np

from src.resnet_lddmm.learning.callbacks.base import Callback
from src.resnet_lddmm.resnet_flow.io import FrameTransform


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
