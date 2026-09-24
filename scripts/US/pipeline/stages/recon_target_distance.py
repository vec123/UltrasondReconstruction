"""Stage: nearest-neighbor (L2) point-to-point distance between a
reconstruction and its registration target, in both directions.

- recon -> target: for every recon vertex, distance to the nearest target
  vertex. Written onto the recon mesh as the point field 'recon_dist_2_target'.
- target -> recon: for every target vertex, distance to the nearest recon
  vertex. Written onto the target mesh as the point field 'target_dist_2_recon'.

This is a point-to-point (vertex-to-nearest-vertex) approximation of the true
point-to-surface distance, computed with scipy.spatial.cKDTree - same
approach as scan_consistency.py's consistency scoring.

The reconstruction carries heatmap landmark fields. A similarity transform is
fit from those landmark vertices to `landmarks_path`, then applied to both the
reconstruction and target before distances are computed. The saved meshes and
distance fields therefore share the same landmark frame and can be loaded
together in ParaView. Distances are reported in that frame's units.

Each direction is also written out as a distance-over-percentage curve, to
'distance_coverage.json' and 'distance_coverage.png': the share of points
already within a given distance, read left to right. That curve is what
separates fits the mean cannot - target->recon means sit within a few tenths
of each other across very different models, because they are dominated by the
stretch of scan no reconstruction covers, while the share of the scan
explained to within a millimetre is not.

Those same landmark pairs are also scored directly: for every label shared by
the reconstruction's heatmap fields and the landmark CSV, the Mahalanobis
distance sqrt(d^T M d) between the aligned recon landmark and the CSV landmark
is written to 'landmark_distances.json'. M is `landmark_metric`, the identity
by default - in which case this is the plain Euclidean landmark residual left
over by the similarity fit.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from vtk.util.numpy_support import vtk_to_numpy

from src.config.US.schema import ReconTargetDistanceConfig as Config
from src.landmarks.vtk_landmarks import (
    fit_landmark_alignment,
    heatmap_landmark_vertex_indices,
    landmarks_csv_to_polydata,
)
from src.vtk.fields import add_point_field
from src.vtk.io import load_vtp, save_vtp



def nearest_neighbor_distances(query_points, reference_points):
    """For every point in `query_points`, the L2 distance to its nearest
    neighbor in `reference_points`."""
    tree = cKDTree(reference_points)
    distances, _ = tree.query(query_points, k=1, workers=-1)
    return distances


def summarize(distances, coverage_thresholds=()):
    """The per-direction stats block written into distance_summary.json.

    `rms` sits between the mean and the max: it answers the tail without
    resting on the single worst point the way max does. It is always >= mean,
    and the gap between the two is itself the read on how heavy that tail is.

    `fraction_within` is the share of distances below each threshold. It is
    what separates two fits that neither the mean nor the rms can: target->recon
    means sit within a few tenths of each other across very different models,
    because both are dominated by the stretch of scan no reconstruction covers,
    while the share of the scan explained to within a millimetre is not.
    """
    summary = {
        "min": float(distances.min()),
        "mean": float(distances.mean()),
        "median": float(np.median(distances)),
        "rms": float(np.sqrt(np.mean(distances ** 2))),
        "max": float(distances.max()),
    }
    if len(coverage_thresholds):
        summary["fraction_within"] = {
            f"{float(threshold):g}": float((distances < threshold).mean())
            for threshold in coverage_thresholds
        }
    return summary


COVERAGE_PERCENT_STEP = 0.5


def coverage_curve(distances):
    """The distance-over-percentage curve, sampled on a fixed percentage grid.

    Returns (percent, distance): the distance below which that share of points
    lies. Sampling in percent rather than in distance keeps the curve exact
    (these are real quantiles, not binned counts) and the same length for
    every mesh, however many points it has.
    """
    percent = np.arange(0.0, 100.0 + COVERAGE_PERCENT_STEP, COVERAGE_PERCENT_STEP)
    return percent, np.percentile(distances, percent)


def draw_coverage(curves, cfg, output_path):
    """One figure: distance on x, share of points already within it on y."""
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for label, color, (percent, distance) in curves:
        axis.plot(distance, percent, color=color, linewidth=2.0, label=label)

    # The x range is set by the 99th percentile, not the max: the last percent
    # of either direction runs several times further than the rest of the
    # curve and would squeeze everything interesting into the left edge.
    axis.set_xlim(0, max(np.percentile(d, 99) for _, _, (_, d) in curves))
    axis.set_ylim(0, 100)
    for threshold in cfg.coverage_thresholds:
        axis.axvline(threshold, color="#9a9a94", linewidth=0.8, linestyle=":")
    axis.set_title("Share of points within a given distance")
    axis.set_xlabel(f"Distance in {cfg.units or 'landmark frame units'}")
    axis.set_ylabel("Points within that distance (%)")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved distance coverage plot to: {output_path}")


def resolve_metric(landmark_metric):
    """The 3x3 metric matrix M of the landmark Mahalanobis distance. Empty /
    unset gives the identity, i.e. plain Euclidean distance."""
    if landmark_metric is None or len(landmark_metric) == 0:
        return np.eye(3)

    metric = np.asarray(landmark_metric, dtype=float)
    if metric.shape != (3, 3):
        raise ValueError(f"landmark_metric must be 3x3, got shape {metric.shape}")
    if not np.allclose(metric, metric.T):
        raise ValueError("landmark_metric must be symmetric")
    if np.min(np.linalg.eigvalsh(metric)) < -1e-9:
        raise ValueError("landmark_metric must be positive semi-definite")
    return metric


def mahalanobis_distances(deltas, metric):
    """sqrt(d^T M d) for every row d of `deltas`. The quadratic form is clipped
    at 0 before the root, so a PSD metric's round-off can't produce a NaN."""
    quadratic = np.einsum("ij,jk,ik->i", deltas, metric, deltas)
    return np.sqrt(np.maximum(quadratic, 0.0))


def landmark_distances(recon_points, cfg, metric):
    """Per-label Mahalanobis distance between the reconstruction's own
    heatmap landmarks - `recon_points` must already be in the landmark frame -
    and the labelled positions in `cfg.landmarks_path`."""
    template_indices = heatmap_landmark_vertex_indices(cfg.recon_path)
    _, subject_df = landmarks_csv_to_polydata(cfg.landmarks_path, lps_to_ras=False)
    subject_df = subject_df.set_index("label")

    labels = sorted(label for label in template_indices if label in subject_df.index)
    if not labels:
        raise ValueError(
            f"No label in common between '{cfg.recon_path}' heatmap fields and "
            f"'{cfg.landmarks_path}'."
        )

    recon_landmarks = recon_points[[template_indices[label] for label in labels]].astype(float)
    csv_landmarks = subject_df.loc[labels, ["x", "y", "z"]].values.astype(float)
    deltas = recon_landmarks - csv_landmarks
    distances = mahalanobis_distances(deltas, metric)

    per_landmark = {
        label: {
            "distance": float(distance),
            "recon_point": recon_point.tolist(),
            "landmark_point": csv_point.tolist(),
        }
        for label, distance, recon_point, csv_point in zip(
            labels, distances, recon_landmarks, csv_landmarks
        )
    }
    return per_landmark, distances


def run(cfg: Config):
    output_target_path = os.path.join(cfg.output_dir or os.path.dirname(cfg.target_path), "target_w_distance.vtp")
    output_recon_path = os.path.join(cfg.output_dir or os.path.dirname(cfg.recon_path), "recon_w_distance.vtp")
    output_summary_path = os.path.join(cfg.output_dir or os.path.dirname(cfg.recon_path), "distance_summary.json")
    output_plot_path = os.path.join(cfg.output_dir or os.path.dirname(cfg.recon_path), "distance_distribution.png")
    output_landmark_path = os.path.join(cfg.output_dir or os.path.dirname(cfg.recon_path), "landmark_distances.json")
    output_coverage_path = os.path.join(cfg.output_dir or os.path.dirname(cfg.recon_path), "distance_coverage.json")
    output_coverage_plot_path = os.path.join(cfg.output_dir or os.path.dirname(cfg.recon_path), "distance_coverage.png")

    target_mesh = load_vtp(cfg.target_path)
    recon_mesh = load_vtp(cfg.recon_path)

    target_points = vtk_to_numpy(target_mesh.GetPoints().GetData())
    recon_points = vtk_to_numpy(recon_mesh.GetPoints().GetData())

    if not cfg.landmarks_path:
        raise ValueError("recon_target_distance requires landmarks_path")

    alignment = fit_landmark_alignment(
        recon_points,
        template_vtp=cfg.recon_path,
        landmark_path=cfg.landmarks_path,
        lps_to_ras=False,
        with_scale=True,
    )
    recon_mesh = alignment.apply_polydata(recon_mesh)
    target_mesh = alignment.apply_polydata(target_mesh)
    recon_points = vtk_to_numpy(recon_mesh.GetPoints().GetData())
    target_points = vtk_to_numpy(target_mesh.GetPoints().GetData())

    scale = 1.0
    units = cfg.units or "landmark frame units"

    def place_geometry(mesh):
        return mesh

    recon_dist_2_target = nearest_neighbor_distances(recon_points, target_points) / scale
    add_point_field(recon_mesh, recon_dist_2_target, field_name="recon_dist_2_target")
    recon_mesh = place_geometry(recon_mesh)
    save_vtp(recon_mesh, output_recon_path)
    print(
        f"recon_dist_2_target ({units}): mean={recon_dist_2_target.mean():.4f} "
        f"max={recon_dist_2_target.max():.4f} -> {output_recon_path}"
    )

    target_dist_2_recon = nearest_neighbor_distances(target_points, recon_points) / scale
    add_point_field(target_mesh, target_dist_2_recon, field_name="target_dist_2_recon")
    target_mesh = place_geometry(target_mesh)
    save_vtp(target_mesh, output_target_path)
    print(
        f"target_dist_2_recon ({units}): mean={target_dist_2_recon.mean():.4f} "
        f"max={target_dist_2_recon.max():.4f} -> {output_target_path}"
    )

    recon_curve = coverage_curve(recon_dist_2_target)
    target_curve = coverage_curve(target_dist_2_recon)
    draw_coverage(
        [
            ("Reconstruction to target", "#2f6f9f", recon_curve),
            ("Target to reconstruction", "#b5653b", target_curve),
        ],
        cfg,
        output_coverage_plot_path,
    )
    coverage = {
        "units": units,
        "percent_step": COVERAGE_PERCENT_STEP,
        "target_path": cfg.target_path,
        "recon_path": cfg.recon_path,
    }
    for name, distances, (percent, distance) in (
        ("recon_dist_2_target", recon_dist_2_target, recon_curve),
        ("target_dist_2_recon", target_dist_2_recon, target_curve),
    ):
        coverage[name] = {
            "n_points": int(distances.size),
            "percent": [round(float(value), 4) for value in percent],
            "distance": [round(float(value), 6) for value in distance],
            "fraction_within": summarize(distances, cfg.coverage_thresholds).get(
                "fraction_within", {}
            ),
        }
    with open(output_coverage_path, "w") as f:
        json.dump(coverage, f, indent=2)
    print(f"Saved distance coverage curves to: {output_coverage_path}")

    metric = resolve_metric(cfg.landmark_metric)
    per_landmark, landmark_dists = landmark_distances(recon_points, cfg, metric)
    landmark_summary = {
        "units": units,
        "recon_path": cfg.recon_path,
        "landmarks_path": cfg.landmarks_path,
        "metric": metric.tolist(),
        "metric_is_identity": bool(np.allclose(metric, np.eye(3))),
        "landmarks": per_landmark,
        "summary": summarize(landmark_dists, cfg.coverage_thresholds),
    }
    with open(output_landmark_path, "w") as f:
        json.dump(landmark_summary, f, indent=2)
    print(
        f"landmark Mahalanobis distance ({units}, {len(per_landmark)} landmarks): "
        f"mean={landmark_dists.mean():.4f} max={landmark_dists.max():.4f} "
        f"-> {output_landmark_path}"
    )

    bins = np.histogram_bin_edges(
        np.concatenate((recon_dist_2_target, target_dist_2_recon)), bins=100
    )
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    axis.hist(
        recon_dist_2_target,
        bins=bins,
        color="#2f6f9f",
        alpha=0.55,
        label="Reconstruction to target",
    )
    axis.hist(
        target_dist_2_recon,
        bins=bins,
        color="#b5653b",
        alpha=0.55,
        label="Target to reconstruction",
    )
    axis.set_title("Point-distance distribution")
    axis.set_xlabel(f"Distance in {units}")
    axis.set_ylabel("Number of points")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved distance distribution plot to: {output_plot_path}")

    summary = {
        "units": units,
        "target_path": cfg.target_path,
        "recon_path": cfg.recon_path,
        "distance_distribution_plot": output_plot_path,
        "distance_coverage": output_coverage_path,
        "distance_coverage_plot": output_coverage_plot_path,
        "landmark_distances": output_landmark_path,
        "recon_dist_2_target": summarize(recon_dist_2_target, cfg.coverage_thresholds),
        "target_dist_2_recon": summarize(target_dist_2_recon, cfg.coverage_thresholds),
    }
    with open(output_summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved distance summary to: {output_summary_path}")
