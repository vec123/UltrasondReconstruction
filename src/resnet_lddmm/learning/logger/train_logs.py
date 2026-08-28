import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless backend: save PNGs without a display / GUI event loop
import matplotlib.pyplot as plt

from src.vtk.io import save_vtp
from src.vtk.create import create_polydata, create_polydata_w_lines
from src.vtk.fields import add_point_field
from src.graphs.graphs import get_individual_graph, get_bipartite_graph



class Logger1:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(os.path.join(log_dir, "vtk"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    def log_metrics(self, metrics, step):
        print(f"[step {step}] " + ", ".join(f"{k}={v:.6f}" for k, v in metrics.items()))

    def save_checkpoint(self, stepper, step):
        path = os.path.join(self.log_dir, "checkpoints", f"step_{step}.pt")
        torch.save({
            "encoder": stepper.encoder.state_dict(),
            "decoder": stepper.decoder.state_dict(),
            "optimizer": stepper.optimizer.state_dict(),
        }, path)
        print(f"  checkpoint -> {path}")

    def visualize_results(self, pred, step, max_num = 4):
        pred = pred.detach().cpu().numpy()
        for i in range(pred.shape[0]):
            if i < max_num:
                path = os.path.join(self.log_dir, "vtk", f"pred_shape{i}_step{step}.vtp")
                save_vtp(create_polydata(pred[i]), path)
        print(f"  saved some of {pred.shape[0]} prediction VTP(s) at step {step}")
        
        
        
    

class TrainingLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        # metric name -> list of (step, value), so train and validation series can be
        # plotted together and persisted for later inspection.
        self.history = {}

    def log_metrics(self, metrics, step):
        print(f"[step {step}] " + ", ".join(f"{k}={v:.6f}" for k, v in metrics.items()))
        for name, value in metrics.items():
            self.history.setdefault(name, []).append((int(step), float(value)))
        self._save_metrics()

    def _save_metrics(self):
        """Persist the metric history to JSON so a run's curves survive even if the
        process dies before the final plot (and can be re-plotted without re-training)."""
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "metrics.json"), "w") as f:
            json.dump(self.history, f, indent=2)

    def plot_metrics(self, filename="metrics.png"):
        """Plot every logged metric series (``train/loss``, ``val/recon``, ...) on one
        step axis and save a PNG next to the logs. No-ops on an empty history.

        Series names are only ever labels and dict keys here, never path segments,
        so the ``train/`` and ``val/`` prefixes are safe on every platform."""
        series = {name: pts for name, pts in self.history.items() if pts}
        if not series:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, pts in series.items():
            steps, values = zip(*pts)
            ax.plot(steps, values, marker=".", label=name)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        # Losses span orders of magnitude early in training; log scale is only valid
        # when every point is positive (chamfer + laplacian are, but guard anyway).
        if all(v > 0 for pts in series.values() for _, v in pts):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, filename)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved metrics plot -> {path}")

    def save_checkpoint(self, stepper, step,  directory="checkpoints"):
        checkpoint_dir = os.path.join(self.log_dir, directory)
        path = os.path.join(checkpoint_dir, f"step_{step}.pt")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            "encoder": stepper.encoder.state_dict(),
            "decoder": stepper.decoder.state_dict(),
            "optimizer": stepper.optimizer.state_dict(),
        }, path)
        print(f"  checkpoint -> {path}")

    def visualize_results(self, pred, step, max_num = 4, subdir = "vtk", random_sample=False):
        pred = pred.detach().cpu().numpy()
        out_dir = os.path.join(self.log_dir, subdir)
        os.makedirs(out_dir, exist_ok=True)

        if random_sample and pred.shape[0] > 0:
            indices = torch.randperm(pred.shape[0])[:min(max_num, pred.shape[0])].tolist()
        else:
            indices = list(range(min(max_num, pred.shape[0])))

        for i in indices:
            path = os.path.join(out_dir, f"pred_shape{i}_step{step}.vtp")
            save_vtp(create_polydata(pred[i]), path)
        print(f" saved {len(indices)} of {pred.shape[0]} prediction VTP(s) at step {step}")

    def visualize_batch(self, batch, pred, step, subdir = "vtk", max_num= 4, random_sample=False):
        """Save, for this step, the input graph, the supergraph (if any), the true
        target verts, and the predictions.

        ``batch`` is the ``(graph, super_graph, true_verts, mask)`` tuple the trainer
        steps on. Graph tensors are expected on CPU (the trainer moves its own copies to
        the device), so they render directly. ``subdir`` selects the output folder under
        ``log_dir`` so train and validation VTPs never overwrite each other at the same
        step number. ``random_sample`` determines whether to use random indices or first N.
        """
        graph, super_graph, true_verts, mask = batch[0], batch[1], batch[2], batch[3]
        if graph is not None:
            self._save_graph_vtp(graph, step, name="input_graph",
                                is_supernodes=False,
                                subdir=subdir,
                                max_num = max_num, random_sample=random_sample)
        if super_graph is not None:
            self._save_graph_vtp(super_graph, step, name="supergraph",
                                 is_supernodes=True,
                                 subdir=subdir,
                                  max_num = max_num, random_sample=random_sample)
        if true_verts is not None:
            self._save_true_verts(true_verts, mask, step, subdir=subdir,  max_num = max_num, random_sample=random_sample)

        if pred is not None:
            self.visualize_results(pred, step, subdir=subdir,  max_num = max_num, random_sample=random_sample)

    def visualize_val_batch(self, batch, pred, step, max_num=4, random_sample=False):
        """Same as ``visualize_batch`` but writes to ``vtk/validation`` so the validation
        reconstructions live alongside — not on top of — the training ones.

        ``max_num`` is forwarded so validation honours the same per-shape cap as
        training; it used to be dropped here, silently pinning validation to 4.
        ``random_sample`` determines whether to use random indices or first N shapes."""
        self.visualize_batch(batch, pred, step, subdir=os.path.join("vtk", "validation"),
                             max_num=max_num, random_sample=random_sample)

    def _save_true_verts(self, true_verts, mask, step, max_num = 4, subdir = "vtk", random_sample=False):
        """Save the reconstruction target as a point cloud (no edges), one VTP per shape.
        Padded entries are dropped via ``mask`` so only real vertices are written."""
        out_dir = os.path.join(self.log_dir, subdir)
        os.makedirs(out_dir, exist_ok=True)
        tv = true_verts.detach().cpu()
        m = mask.detach().cpu().bool() if mask is not None else None

        if random_sample and tv.shape[0] > 0:
            indices = torch.randperm(tv.shape[0])[:min(max_num, tv.shape[0])].tolist()
        else:
            indices = list(range(min(max_num, tv.shape[0])))

        for i in indices:
            pts = tv[i][m[i]] if m is not None else tv[i]        # [n_i, 3] valid verts
            save_vtp(create_polydata(pts), os.path.join(out_dir, f"true_verts_shape{i}_step{step}.vtp"))
        print(f"  saved {len(indices)} of {tv.shape[0]} true-verts VTP(s) at step {step}")

    def _save_graph_vtp(self, graph, step, name, is_supernodes=False, max_num = 4, subdir = "vtk", random_sample=False):
        """Render one VTP per shape (with edges) for a homogeneous graph, or the merged
        full+super point set with aggregation lines for a bipartite supergraph."""
        out_dir = os.path.join(self.log_dir, subdir)
        os.makedirs(out_dir, exist_ok=True)
        num_graphs = int(graph.batch.max().item()) + 1

        if random_sample and num_graphs > 0:
            indices = torch.randperm(num_graphs)[:min(max_num, num_graphs)].tolist()
        else:
            indices = list(range(min(max_num, num_graphs)))

        for i in indices:
                if is_supernodes:
                    pos, edges, node_field = get_bipartite_graph(graph, i)
                    vtp = create_polydata_w_lines(pos, edges)
                    vtp = add_point_field(vtp, field_data=node_field, field_name="super_node")

                    if hasattr(graph, 'source_area') and graph.source_area is not None:
                        src_mask = (graph.source_batch == i)
                        tgt_mask = (graph.batch == i)
                        src_area = graph.source_area[src_mask]
                        tgt_area = graph.area[tgt_mask] if hasattr(graph, 'area') and graph.area is not None else torch.zeros((tgt_mask.sum().item(),), device=graph.source_area.device)
                        area_field = torch.cat([src_area, tgt_area], dim=0).detach().cpu().numpy()
                        vtp = add_point_field(vtp, area_field.astype(np.float32), field_name="area")

                    if hasattr(graph, 'source_normal') and graph.source_normal is not None:
                        src_mask = (graph.source_batch == i)
                        tgt_mask = (graph.batch == i)
                        src_norm = graph.source_normal[src_mask]
                        tgt_norm = torch.zeros((tgt_mask.sum().item(), 3), dtype=graph.source_normal.dtype, device=graph.source_normal.device)
                        normal_field = torch.cat([src_norm, tgt_norm], dim=0).detach().cpu().numpy()
                        vtp = add_point_field(vtp, normal_field.astype(np.float32), field_name="normal")

                else:
                    pos, edges = get_individual_graph(graph, i)
                    vtp = create_polydata_w_lines(pos, edges)
                    if hasattr(graph, 'area') and graph.area is not None:
                        node_mask = (graph.batch == i)
                        area_field = graph.area[node_mask].detach().cpu().numpy()
                        vtp = add_point_field(vtp, area_field.astype(np.float32), field_name="area")
                    if hasattr(graph, 'normal') and graph.normal is not None:
                        node_mask = (graph.batch == i)
                        normal_field = graph.normal[node_mask].detach().cpu().numpy()
                        vtp = add_point_field(vtp, normal_field.astype(np.float32), field_name="normal")

                save_vtp(vtp, os.path.join(out_dir, f"{name}_shape{i}_step{step}.vtp"))
        print(f"  saved {len(indices)} of {num_graphs} {name} VTP(s) at step {step}")

    def log_visualizations(self, data_dict, step, sample_idx=0, max_num = 4):
        """
        Expects data_dict: {'original': np.array, 'canonical': np.array, ...}
        """
        for i, (name, data) in enumerate(data_dict.items()):
            if i < max_num:
                print(f"Logging visualization for {name} at step {step}")
                poly = create_polydata(data)
                path = os.path.join(self.log_dir, "vtk", f"{sample_idx}_{name}_{step}.vtp")
                save_vtp(poly, path)
        