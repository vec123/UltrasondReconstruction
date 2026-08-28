"""Shared "named landmarks, each with its own decay + importance" machinery,
used by both the source-side (``landmark_weight_field.py``, heatmap/Euclidean)
and target-side (``target_weight_field.py``, CSV/geodesic) precompute scripts.

Deliberately geometry-agnostic: this module knows nothing about heatmaps,
meshes, or distance metrics -- callers supply a signed 3D displacement (a
single point-to-point diff, or an edge, one call at a time) and this module
turns it into a normalised (Mahalanobis-style) distance, then combines those
across landmarks. Both the "parse the ``include`` list" and "normalise +
combine" parts are identical for the Euclidean and geodesic cases, so they
live here once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

DEFAULT_DECAY_SCALE = 0.001
DEFAULT_MAX_WEIGHT = 1.0

AXES = ("x", "y", "z")

_ALLOWED_ENTRY_KEYS = {"label", "decay_scale", "max_weight", "decay_axes"}
_ALLOWED_AXIS_DIR_KEYS = {"neg", "pos"}


@dataclass(frozen=True)
class LandmarkEntry:
    """One included landmark: its decay scale and the peak it decays from.

    ``max_weight`` is the value AT the landmark itself (distance 0); the
    gaussian decays away from it as usual. Two landmarks can therefore have
    the same halo size (``decay_scale``) but different importance -- or the
    same importance but different halo sizes -- independently.

    ``decay_axes`` makes the halo ANISOTROPIC and, per axis, DIRECTIONAL:
    ``None`` (the default) means fully isotropic -- every axis and direction
    uses ``decay_scale``. Otherwise it maps a subset of ``{"x","y","z"}`` to
    an ``(neg, pos)`` pair -- the scale used for a displacement with a
    negative vs. positive component on that axis, respectively. An axis not
    present falls back to ``(decay_scale, decay_scale)``. See ``axis_bounds``.
    """

    label: str
    decay_scale: float
    max_weight: float = DEFAULT_MAX_WEIGHT
    decay_axes: Optional[Dict[str, Tuple[float, float]]] = None


def axis_bounds(entry: LandmarkEntry) -> Dict[str, Tuple[float, float]]:
    """``entry`` -> ``{"x": (neg, pos), "y": (neg, pos), "z": (neg, pos)}``.

    Every axis defaults to ``(entry.decay_scale, entry.decay_scale)``
    (isotropic, symmetric); ``entry.decay_axes`` overrides whichever axes it
    names. Always returns all three axes, so callers never need to branch on
    whether ``decay_axes`` was set.
    """
    bounds = {axis: (entry.decay_scale, entry.decay_scale) for axis in AXES}
    if entry.decay_axes:
        bounds.update(entry.decay_axes)
    return bounds


def mahalanobis_norm(diff: torch.Tensor, bounds: Dict[str, Tuple[float, float]]) -> torch.Tensor:
    """``[..., 3]`` signed displacement -> ``[...]`` normalised distance.

    Each of the 3 components is divided by ITS OWN direction's scale --
    ``bounds[axis][1]`` (``pos``) where that component is >= 0, ``bounds[axis][0]``
    (``neg``) where it is negative -- before taking the L2 norm. With every
    axis's ``neg == pos == s``, this reduces exactly to ``|diff| / s``: plain
    isotropic normalisation is the special case, not a different formula.

    Args:
        diff: signed displacement(s), last dim is ``(dx, dy, dz)``.
        bounds: see ``axis_bounds``.

    Returns:
        Tensor with ``diff``'s shape minus the last dimension.
    """
    scaled = torch.empty_like(diff)
    for i, axis in enumerate(AXES):
        neg, pos = bounds[axis]
        component = diff[..., i]
        scale = torch.where(component >= 0,
                            torch.full_like(component, pos),
                            torch.full_like(component, neg))
        scaled[..., i] = component / scale
    return scaled.norm(dim=-1)


def _parse_decay_axes(raw: dict, default_scale: float, label: str) -> Dict[str, Tuple[float, float]]:
    """The per-entry ``decay_axes`` mapping -> ``{"x": (neg,pos), ...}`` (only
    the axes actually named in ``raw``; ``axis_bounds`` fills in the rest).

    Each named axis's value is either a number (symmetric override for that
    axis) or a ``{neg, pos}`` mapping (either key optional, defaulting to
    ``default_scale`` -- e.g. ``{pos: 0.02}`` alone reaches further only in
    the positive direction, leaving the negative direction untouched).
    """
    unknown_axes = set(raw) - set(AXES)
    if unknown_axes:
        raise ValueError(
            f"decay_axes for {label!r} has unknown axes: {sorted(unknown_axes)}; "
            f"expected a subset of {list(AXES)}"
        )
    resolved: Dict[str, Tuple[float, float]] = {}
    for axis, value in raw.items():
        if isinstance(value, (int, float)):
            if value <= 0:
                raise ValueError(
                    f"decay_axes.{axis} for {label!r} must be > 0, got {value}"
                )
            resolved[axis] = (float(value), float(value))
        elif isinstance(value, dict):
            unknown_dir = set(value) - _ALLOWED_AXIS_DIR_KEYS
            if unknown_dir:
                raise ValueError(
                    f"decay_axes.{axis} for {label!r} has unknown keys: {sorted(unknown_dir)}"
                )
            neg = float(value.get("neg", default_scale))
            pos = float(value.get("pos", default_scale))
            if neg <= 0 or pos <= 0:
                raise ValueError(
                    f"decay_axes.{axis} for {label!r} must be > 0, got neg={neg}, pos={pos}"
                )
            resolved[axis] = (neg, pos)
        else:
            raise ValueError(
                f"decay_axes.{axis} for {label!r} must be a number or a "
                f"{{neg, pos}} mapping, got {value!r}"
            )
    return resolved


def parse_include_entries(raw_include, default_decay_scale: float,
                          default_max_weight: float) -> List[LandmarkEntry]:
    """The YAML ``include`` list -> resolved ``LandmarkEntry`` list.

    Each item is either a plain label string (uses the two defaults, fully
    isotropic) or a ``{label, decay_scale, max_weight, decay_axes}`` mapping
    (overrides any subset for just that one landmark -- see ``LandmarkEntry``
    for what ``decay_axes`` does). Shared by every landmark-weight-field
    config -- source and target side alike -- so the syntax and validation
    stay identical regardless of which mesh/distance-metric ends up using it.

    Raises:
        ValueError: an entry has no 'label', an unknown key, a non-positive
            decay_scale/max_weight/decay_axes value, a malformed decay_axes,
            a duplicate label, or is neither a string nor a mapping.
    """
    entries: List[LandmarkEntry] = []
    seen = set()
    for item in raw_include:
        if isinstance(item, str):
            label = item
            decay_scale, max_weight, decay_axes = default_decay_scale, default_max_weight, None
        elif isinstance(item, dict):
            item = dict(item)
            if "label" not in item:
                raise ValueError(f"include entry {item!r} needs a 'label'")
            label = item.pop("label")
            unknown = set(item) - _ALLOWED_ENTRY_KEYS
            if unknown:
                raise ValueError(
                    f"unknown keys in include entry for {label!r}: {sorted(unknown)}"
                )
            decay_scale = float(item.get("decay_scale", default_decay_scale))
            if decay_scale <= 0:
                raise ValueError(f"decay_scale for {label!r} must be > 0, got {decay_scale}")
            max_weight = float(item.get("max_weight", default_max_weight))
            if max_weight <= 0:
                raise ValueError(f"max_weight for {label!r} must be > 0, got {max_weight}")
            raw_axes = item.get("decay_axes")
            decay_axes = (_parse_decay_axes(raw_axes, decay_scale, label)
                         if raw_axes else None)
        else:
            raise ValueError(
                f"'include' entries must be a label string or a "
                f"{{label, decay_scale, max_weight, decay_axes}} mapping, got {item!r}"
            )

        if label in seen:
            raise ValueError(f"'include' has duplicate label: {label!r}")
        seen.add(label)
        entries.append(LandmarkEntry(label=label, decay_scale=decay_scale,
                                     max_weight=max_weight, decay_axes=decay_axes))
    return entries


def combine_peaked_gaussians(distance: torch.Tensor, entries: List[LandmarkEntry]) -> torch.Tensor:
    """``[N, L]`` per-landmark NORMALISED distances -> ``[N]`` max-combined weight.

    ``max_i(entry_i.max_weight * exp(-0.5 * distance[:, i] ** 2))``. ``distance``
    must already be in "how many decay_scales away" units -- i.e. already
    normalised via ``mahalanobis_norm``/``axis_bounds`` -- NOT a raw
    world-space distance; this function no longer divides by ``decay_scale``
    itself, since anisotropic/directional entries need that division done
    per-axis-and-direction, upstream, before landmarks with different bounds
    can be compared on the same footing.

    Unfloored and uncombined with anything else: callers apply their own
    ``min_weight`` (and, for the target side, their own extra multiplicative
    fields) afterward.

    Args:
        distance: ``[N, L]``, one column per entry, same order as ``entries``,
            already normalised.
        entries: L ``LandmarkEntry``, matching ``distance``'s column order.

    Returns:
        ``[N]`` tensor.
    """
    peaks = torch.tensor([e.max_weight for e in entries],
                         dtype=distance.dtype, device=distance.device)
    return (peaks * torch.exp(-0.5 * distance ** 2)).max(dim=1).values
