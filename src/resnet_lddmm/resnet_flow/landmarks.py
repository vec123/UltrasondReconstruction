"""Landmark guidance: sparse, named point correspondences between template and target.

Two independent sources of landmarks meet here, in two different frames:

  * TARGET landmarks live in the ``landmarks:`` folder as one or more Slicer
    fiducial CSVs (x, y, z, label, ...), one file per annotator/pass. Several
    files naming the same label give a mean position AND a per-axis spread --
    the spread is the annotators' disagreement, and sets how tightly this
    label should be enforced.

  * TEMPLATE landmarks are not stored as an explicit point list at all. The
    template mesh instead carries a per-vertex ``heatmap_<label>`` point field
    for each label (see ``io.Shape.point_fields``) -- a soft indicator of where
    that landmark sits on the template's own surface. The template's landmark
    position for a label is taken as the ARGMAX vertex of that field: exactly
    on the mesh, trivial to sanity-check by exporting as a point cloud.

Both landmark sets are resolved once, at build time (``build_landmark_term_kwargs``),
into tensors in the SAME normalised frame training already runs in -- the
``FrameTransform`` produced by ``io.joint_normalize`` over source+target. The
per-step loss term (``losses.terms.LandmarkGuidanceLoss``) then only has to flow
the template points and compare; it never touches CSVs or point fields.
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


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


def resolve_template_points(template_shape, labels: List[str],
                             heatmap_prefix: str = "heatmap_") -> np.ndarray:
    """The template's own landmark position for each label: its heatmap's argmax vertex.

    Args:
        template_shape: an ``io.Shape`` with ``points`` [1, N, 3] and
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


def build_landmark_term_kwargs(cfg, template_shape, transform,
                                sigma_floor: float = 1e-3,
                                heatmap_prefix: str = "heatmap_",
                                label_weights: Optional[Dict[str, float]] = None,
                                hard_precision: Optional[Dict[str, float]] = None) -> Optional[dict]:
    """cfg.landmarks + the template's heatmaps -> LandmarkGuidanceLoss constructor kwargs.

    Resolves both sides of the correspondence (see module docstring) and maps
    them into the SAME normalised frame ``transform`` was built for: the mean
    position through ``transform.apply`` (a position: center and scale both
    apply), the std through division by ``transform.scale`` alone (a spread --
    shifting it by the center would be meaningless). ``transform.scale`` is a
    single isotropic float (``joint_normalize`` fits one bounding box with one
    scale for all axes), so this division is exact per axis.

    Args:
        cfg: ExperimentCfg; only ``cfg.landmarks`` and ``cfg.dtype`` are read.
        template_shape: the NORMALISED template Shape (e.g. ``source_norm`` from
            ``load_pair``) -- must still carry ``point_fields``, so pass the
            Shape itself, not a stripped batch wrapper.
        transform: the FrameTransform training runs in (from ``load_pair`` /
            ``joint_normalize``), so the resolved targets line up with
            ``ctx.template_points``/``ctx.flow`` at loss time.
        sigma_floor: minimum per-axis std, in the NORMALISED frame, applied
            after the division above. Without a floor, a label every annotator
            placed identically (or the common single-file case, std == 0)
            would give an infinite precision and a dominating, noise-fitting
            gradient.
        heatmap_prefix: forwarded to ``resolve_template_points``.
        label_weights: optional ``{label: weight}`` -- doubles as BOTH the
            inclusion filter and a multiplier on the annotation-derived
            precision. A label is only used at all if it is present here
            with weight > 0; a label the target annotated but that is
            missing from this dict, or given weight <= 0, is dropped BEFORE
            template resolution -- so a template that cannot resolve that
            label (no ``heatmap_<label>`` field) never raises for it. A
            surviving label's weight multiplies its annotator-derived
            precision as before, e.g. to lean harder on a landmark the
            annotators happened to agree on by luck, or ease off one that
            matters less for this run than its spread implies. All three
            axes of a label share its one weight -- this scales how much the
            label counts, not its per-axis anisotropy. None (default) is the
            old behaviour: every annotated label is used, uniformly weighted
            1.0.
        hard_precision: optional ``{label: precision}`` OVERRIDE, applied AFTER
            ``label_weights`` and replacing that label's precision outright --
            the annotator-derived std (and any label_weights entry for it) is
            discarded, not multiplied. This is the closest a gradient-trained
            quadratic penalty gets to a hard equality constraint: pick a value
            large enough (e.g. 1e6-1e8, so deviating costs far more than
            deforming the rest of the shape) without being LITERALLY infinite
            -- an infinite precision turns zero deviation into ``0 * inf =
            nan`` and a nonzero one into an infinite loss, either of which
            breaks training on the very first step. A label not named here
            keeps its normal ``std x label_weights`` precision. None (default,
            same as ``{}``): no overrides.

    Returns:
        None if ``cfg.landmarks`` is not set (landmark guidance disabled).
        Otherwise a dict with keys ``template_points``, ``target_points``,
        ``target_precision`` (each a ``[1, L, 3]`` tensor in ``cfg.dtype``) and
        ``labels`` (the L label names, same order as the tensors' L axis).

    Raises:
        ValueError: ``label_weights`` (plus ``hard_precision``, which also
            counts as inclusion) excludes every label the target annotated
            (nothing left to guide on); the same label appears in both
            ``label_weights`` and ``hard_precision`` (ambiguous -- scaling a
            precision that is about to be thrown away and replaced makes no
            sense); or a ``hard_precision`` value is not > 0. A
            ``hard_precision`` label this subject didn't annotate is silently
            skipped, not an error -- see the inclusion filter above.
    """
    if not cfg.landmarks:
        return None

    dtype = getattr(torch, cfg.dtype)

    annotations = load_annotations(cfg.landmarks)
    if label_weights is not None:
        # A label named in hard_precision is included even if label_weights
        # omits it (or gives it 0) -- hard_precision is its own inclusion
        # signal, since it goes on to REPLACE this label's precision outright
        # (see below), independent of label_weights' soft multiplier.
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
        annotations = included
    labels, target_mean_world, target_std_world = aggregate(annotations)
    template_points_world = resolve_template_points(template_shape, labels, heatmap_prefix)

    template_points = torch.tensor(template_points_world, dtype=dtype).unsqueeze(0)
    target_points = transform.apply(torch.tensor(target_mean_world, dtype=dtype)).unsqueeze(0)
    target_std = (torch.tensor(target_std_world, dtype=dtype) / transform.scale).clamp_min(sigma_floor)
    target_precision = (1.0 / (2.0 * target_std ** 2)).unsqueeze(0)

    if label_weights and hard_precision:
        both = sorted(set(label_weights) & set(hard_precision))
        if both:
            raise ValueError(
                f"label(s) {both} are in both loss.landmark_kwargs.label_weights "
                f"and hard_precision -- ambiguous, since hard_precision replaces "
                f"the precision label_weights would have scaled. Put each label "
                f"in only one of the two."
            )

    if label_weights:
        # A label included only via hard_precision has no label_weights entry
        # (that's allowed -- see the inclusion filter above); its 1.0 default
        # here is moot regardless, since hard_precision overwrites its
        # precision outright right below.
        weights = torch.tensor([label_weights.get(label, 1.0) for label in labels], dtype=dtype)
        target_precision = target_precision * weights.view(1, -1, 1)

    if hard_precision:
        for label, value in hard_precision.items():
            if value <= 0:
                raise ValueError(f"hard_precision[{label!r}] must be > 0, got {value}")
        # A hard_precision label this subject didn't annotate at all is simply
        # skipped, not an error -- same reasoning as label_weights above:
        # hard_precision is typically one shared master list (see
        # configs/master/end_to_end.yaml), and which labels a given subject's
        # own CSVs actually carry legitimately varies subject to subject.
        for label, value in hard_precision.items():
            if label in labels:
                target_precision[0, labels.index(label), :] = value

    return {
        "template_points": template_points,
        "target_points": target_points,
        "target_precision": target_precision,
        "labels": labels,
    }
