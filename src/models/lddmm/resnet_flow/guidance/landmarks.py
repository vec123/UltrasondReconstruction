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

from src.shared.landmarks.annotations import (  # noqa: F401  (part of this module's API)
    aggregate,
    load_annotations,
    resolve_template_points,
    select_annotations,
)


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

    annotations = select_annotations(load_annotations(cfg.landmarks), label_weights, hard_precision)
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
        # configs/pipelines/end_to_end.yaml), and which labels a given subject's
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
