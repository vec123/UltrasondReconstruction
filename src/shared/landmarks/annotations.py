"""Named landmarks of a target and of a template.

  * TARGET landmarks: one or more Slicer fiducial CSVs (x, y, z, label, ...) in
    one folder, one file per annotator/pass. Several files naming the same
    label give a mean position AND a per-axis spread.
  * TEMPLATE landmarks: a per-vertex ``heatmap_<label>`` point field for each
    label; the landmark is the ARGMAX vertex of that field.
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_annotations(folder: str) -> Dict[str, List[np.ndarray]]:
    """Read every ``*.csv`` in ``folder`` as a Slicer fiducial export.

    Only ``x``, ``y``, ``z`` and ``label`` are read; the remaining Slicer markup
    columns (``ow, ox, oy, oz, vis, sel, lock, desc, associatedNodeID``) describe
    display state, not geometry, and are ignored.

    Args:
        folder: directory of ``.csv`` files, each one annotator's/pass's set of
            named landmarks in world coordinates.

    Returns:
        {label: [xyz array, ...]} -- one [3] array per file that named that
        label. A label present in only some files simply has a shorter list.

    Raises:
        ValueError: no ``.csv`` files found in ``folder``.
    """
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not paths:
        raise ValueError(f"no landmark .csv files found in {folder}")

    annotations: Dict[str, List[np.ndarray]] = {}
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                label = row["label"]
                xyz = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
                annotations.setdefault(label, []).append(xyz)
    return annotations


def aggregate(annotations: Dict[str, List[np.ndarray]]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Per-label mean position and per-axis std across annotation files.

    Args:
        annotations: {label: [xyz, ...]}, as returned by ``load_annotations``.

    Returns:
        (labels, mean, std): ``labels`` is a sorted list of the L labels found;
        ``mean``/``std`` are [L, 3] arrays, in the same (world) frame the input
        arrays were in. ``std`` uses ddof=1 (sample std) when a label has 2+
        observations, and is exactly 0 for a label with only 1 -- callers must
        floor it before using it as a precision, since a single observation
        carries no information about spread.
    """
    labels = sorted(annotations)
    mean = np.zeros((len(labels), 3))
    std = np.zeros((len(labels), 3))
    for i, label in enumerate(labels):
        points = np.stack(annotations[label], axis=0)  # [n_obs, 3]
        mean[i] = points.mean(axis=0)
        std[i] = points.std(axis=0, ddof=1) if len(points) > 1 else 0.0
    return labels, mean, std


def select_annotations(annotations: Dict[str, List[np.ndarray]],
                       label_weights: Optional[Dict[str, float]] = None,
                       hard_precision: Optional[Dict[str, float]] = None) -> Dict[str, List[np.ndarray]]:
    """The labels of ``annotations`` the landmark term actually guides on.

    See ``build_landmark_term_kwargs`` for what ``label_weights`` and
    ``hard_precision`` mean; only their inclusion role is applied here.

    Args:
        annotations: {label: [xyz, ...]}, as returned by ``load_annotations``.
        label_weights: optional ``{label: weight}``; None keeps every label.
        hard_precision: optional ``{label: precision}``.

    Returns:
        ``annotations`` restricted to the included labels.

    Raises:
        ValueError: ``label_weights`` (plus ``hard_precision``) excludes every
            annotated label.
    """
    if label_weights is None:
        return annotations

    # A label named in hard_precision is included even if label_weights
    # omits it (or gives it 0) -- hard_precision is its own inclusion
    # signal, since it goes on to REPLACE this label's precision outright
    # (see build_landmark_term_kwargs), independent of label_weights' soft
    # multiplier.
    hard_labels = set(hard_precision) if hard_precision else set()
    included = {
        k: v for k, v in annotations.items()
        if label_weights.get(k, 0.0) > 0 or k in hard_labels
    }
    if not included:
        raise ValueError(
            f"loss.landmark_kwargs.label_weights {label_weights} (plus "
            f"hard_precision {hard_precision}) leaves no landmark labels for "
            f"this subject (annotated: {sorted(annotations)})"
        )
    return included


def resolve_template_points(template_shape, labels: List[str],
                             heatmap_prefix: str = "heatmap_") -> np.ndarray:
    """The template's own landmark position for each label: its heatmap's argmax vertex.

    Args:
        template_shape: an ``src.shared.geometry.shape.Shape`` with ``points`` [1, N, 3] and
            ``point_fields`` carrying one ``f"{heatmap_prefix}{label}"`` array
            per label (see module docstring).
        labels: label names to resolve, in the order the result should have.
        heatmap_prefix: field-name prefix; the field for ``label`` is looked up
            as ``point_fields[f"{heatmap_prefix}{label}"]``.

    Returns:
        [L, 3] array, in whatever frame ``template_shape.points`` is in.

    Raises:
        ValueError: naming the label whose heatmap field is missing.
    """
    point_fields = template_shape.point_fields or {}
    points = template_shape.points[0].detach().cpu().numpy()  # [N, 3]

    resolved = np.zeros((len(labels), 3))
    for i, label in enumerate(labels):
        field_name = f"{heatmap_prefix}{label}"
        heatmap = point_fields.get(field_name)
        if heatmap is None:
            raise ValueError(
                f"template has no {field_name!r} point field for landmark "
                f"{label!r}; available heatmap fields: "
                f"{sorted(n for n in point_fields if n.startswith(heatmap_prefix))}"
            )
        resolved[i] = points[np.argmax(heatmap)]
    return resolved
