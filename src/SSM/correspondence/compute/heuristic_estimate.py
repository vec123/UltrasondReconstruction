"""Heuristic correspondence search: k-d tree nearest neighbours, then filters.

Pure functions on PointClouds. Each returns the pairs as numpy arrays
``(rows, cols, dists)``: ``rows`` index the source (query) cloud and ``cols``
the target cloud. The strategy classes in src.SSM.correspondence.methods hold
the settings, check them, and wrap these arrays into a Correspondences.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.SSM.correspondence.cloud import PointCloud


def trim_pairs(rows, cols, dists, ratio: float):
    """Drops the worst ``ratio`` fraction of pairs by distance.

    The relative counterpart to ``max_dist``: that is an absolute cut that never
    moves, this one always removes the tail whatever the current scale. Early in
    a fit, when everything is far, ``max_dist`` does the work; near convergence
    it is this. Trimmed ICP, in one line.
    """
    if ratio <= 0.0 or rows.size == 0:
        return rows, cols, dists
    keep = max(1, int(round(rows.size * (1.0 - ratio))))
    order = np.argsort(dists)[:keep]
    return rows[order], cols[order], dists[order]


def deduplicate_pairs(rows, cols, dists):
    """Keeps only the closest pair per source row and per target column.

    Two passes, each a distance-sorted ``np.unique``: first collapse every
    target claimed by more than one source down to its closest source, then do
    the same the other way round on what is left. The second pass only removes
    rows, so it cannot reintroduce a target duplicate - what comes out has each
    source and each target at most once, i.e. one-to-one in both directions.

    This is a greedy heuristic, not the optimal one-to-one assignment (that is
    the Hungarian algorithm): a source that loses its closest target in the
    first pass is not given a second-best target, it just goes unmatched. Good
    enough for NICP, where a strategy that re-estimates every few iterations
    (``refresh_every``) self-corrects whatever the greedy choice got wrong.
    """
    if rows.size == 0:
        return rows, cols, dists

    def _best_per_key(keys):
        order = np.argsort(dists)
        _, first = np.unique(keys[order], return_index=True)
        return order[first]

    keep = _best_per_key(cols)
    rows, cols, dists = rows[keep], cols[keep], dists[keep]
    keep = _best_per_key(rows)
    rows, cols, dists = rows[keep], cols[keep], dists[keep]
    return rows, cols, dists


def normal_agreement(src_normals, tgt_reference, rows, cols, threshold_deg):
    """Mask of pairs whose normals agree to within ``threshold_deg``."""
    dots = np.einsum("ij,ij->i", src_normals[rows], tgt_reference[cols])
    return dots >= np.cos(np.deg2rad(threshold_deg))


def nearest_pairs(src: PointCloud, tgt: PointCloud, *, max_dist: Optional[float],
                  trim_ratio: float, one_to_one: bool):
    """1-NN in ``tgt`` for every point of ``src``; only ``max_dist`` and ``trim_ratio`` reject."""
    dists, indices = tgt.query(src.points, k=1)
    dists = dists[:, 0]
    indices = indices[:, 0]

    keep = np.isfinite(dists)
    if max_dist is not None:
        keep &= dists <= max_dist

    rows = np.flatnonzero(keep)
    cols, dists = indices[rows], dists[rows]
    if one_to_one:
        rows, cols, dists = deduplicate_pairs(rows, cols, dists)
    return trim_pairs(rows, cols, dists, trim_ratio)


def constrained_pairs(src: PointCloud, tgt: PointCloud, *, max_dist: float,
                      normal_thresh_deg: float, k: int, use_target_normals: bool,
                      trim_ratio: float, one_to_one: bool):
    """For each ``src`` point, the closest of its ``k`` nearest that passes the distance and normal checks."""
    dists, indices = tgt.query(src.points, k=k)

    # candidates within range; scipy pads with inf when k exceeds the cloud
    valid = np.isfinite(dists) & (dists <= max_dist)

    src_normals = src.normals
    reference = tgt.normals if use_target_normals else tgt.points
    # dot(model normal, reference vector) per candidate -> (N, k)
    dots = np.einsum("ij,ikj->ik", src_normals, reference[indices])
    valid &= dots >= np.cos(np.deg2rad(normal_thresh_deg))

    # first accepted candidate per vertex (query returns them sorted by distance)
    has_match = valid.any(axis=1)
    first = np.argmax(valid, axis=1)

    rows = np.flatnonzero(has_match)
    cols = first[rows]
    cols, dists = indices[rows, cols], dists[rows, cols]
    if one_to_one:
        rows, cols, dists = deduplicate_pairs(rows, cols, dists)
    return trim_pairs(rows, cols, dists, trim_ratio)


def filter_pairs(src: PointCloud, tgt: PointCloud, rows, cols, dists, *,
                 max_dist: Optional[float], normal_thresh_deg: Optional[float],
                 trim_ratio: float):
    """Applies the rejection rules to one direction's pairs."""
    keep = np.isfinite(dists)
    if max_dist is not None:
        keep &= dists <= max_dist
    if normal_thresh_deg is not None:
        keep &= normal_agreement(src.normals, tgt.normals, rows, cols, normal_thresh_deg)

    rows, cols, dists = rows[keep], cols[keep], dists[keep]
    return trim_pairs(rows, cols, dists, trim_ratio)


def sample_cloud(cloud: PointCloud, ratio: float, rng):
    """A random ``ratio`` of the cloud, with the indices needed to map back."""
    total = len(cloud)
    if ratio <= 0.0 or ratio >= 1.0:
        return cloud, np.arange(total)
    n = max(1, int(round(total * ratio)))
    if n >= total:
        return cloud, np.arange(total)
    keep = np.sort(rng.choice(total, size=n, replace=False))
    return PointCloud(cloud.points[keep]), keep


def symmetric_pairs(src_full: PointCloud, tgt_full: PointCloud, *, forward_weight: float,
                    backward_weight: float, max_dist: Optional[float], trim_ratio: float,
                    normal_thresh_deg: Optional[float], source_ratio: float,
                    target_ratio: float, seed: Optional[int]):
    """1-NN pairs in both directions, each direction with its own weight.

    Returns ``(rows, cols, dists, weights)``, with the indices into the full
    clouds even when ``source_ratio``/``target_ratio`` thin them first.
    """
    rng = np.random.default_rng(seed)
    src, src_map = sample_cloud(src_full, source_ratio, rng)
    tgt, tgt_map = sample_cloud(tgt_full, target_ratio, rng)
    rules = dict(max_dist=max_dist, normal_thresh_deg=normal_thresh_deg, trim_ratio=trim_ratio)

    rows, cols, dists, weights = [], [], [], []

    if forward_weight > 0:
        d, j = tgt.query(src.points, k=1)
        r, c, dd = filter_pairs(src, tgt, np.arange(len(src)), j[:, 0], d[:, 0], **rules)
        rows.append(r); cols.append(c); dists.append(dd)
        weights.append(np.full(r.size, forward_weight))

    if backward_weight > 0:
        d, i = src.query(tgt.points, k=1)
        r, c, dd = filter_pairs(src, tgt, i[:, 0], np.arange(len(tgt)), d[:, 0], **rules)
        rows.append(r); cols.append(c); dists.append(dd)
        weights.append(np.full(r.size, backward_weight))

    # indices are into the sampled clouds; map them back to the originals
    return (src_map[np.concatenate(rows)], tgt_map[np.concatenate(cols)],
            np.concatenate(dists), np.concatenate(weights))
