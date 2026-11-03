"""Distance metrics between a reconstruction and its target.

The functions take plain arrays, so any model's output can be scored the same
way: the stages scripts/US/pipeline/stages/recon_target_distance.py (which
writes the distance fields) and distance_distribution_overlay.py (which
compares them) both read them from here.
"""

import numpy as np
from scipy.spatial import cKDTree


def nearest_neighbor_distances(query_points, reference_points):
    """For every point in `query_points`, the L2 distance to its nearest
    neighbor in `reference_points`."""
    tree = cKDTree(reference_points)
    distances, _ = tree.query(query_points, k=1, workers=-1)
    return distances


def fraction_within(distances, thresholds):
    """The share of `distances` below each threshold, keyed by the threshold."""
    return {
        f"{float(threshold):g}": float((distances < threshold).mean())
        for threshold in thresholds
    }


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
        summary["fraction_within"] = fraction_within(distances, coverage_thresholds)
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
