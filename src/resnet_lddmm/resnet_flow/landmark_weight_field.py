"""Build a ``landmark_weight`` point field from an EXPLICIT subset of the
template's ``heatmap_<label>`` landmarks, with a per-landmark decay, a
per-landmark importance (peak weight), and a floor.

Distinct from ``field_weights.py``: that module turns an ALREADY-BAKED point
field into a runtime weight (generic, knows nothing about landmarks). This
module is the thing that BAKES a landmark-derived field in the first place,
offline, via ``scripts/write_landmark_weight_field.py``. The two are siblings,
not layered -- neither imports the other's field-specific logic, only
``landmarks.resolve_template_points`` for "where is landmark X on the mesh".

Why not just filter ``landmark_distance``: that field is "distance to the
NEAREST of ALL labels", baked in once with every label already mixed together
-- there is no way to recover "distance to nearest of THIS subset" from it.
Recomputing from the per-label ``heatmap_<label>`` fields (the same argmax
resolution ``LandmarkGuidanceLoss`` itself uses, via
``resolve_template_points``) is the only way to make the included set
explicit, and keeps this script and the training pipeline agreeing on where
each landmark actually is.

The ``include``-parsing and per-landmark-gaussian-combination machinery is
shared with the target-side sibling (``target_weight_field.py``) via
``landmark_fields.py``; this module supplies the two things that differ here:
landmark POSITIONS (heatmap argmax) and the DISTANCE metric (Euclidean).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from src.resnet_lddmm.resnet_flow.config import reject_unknown_keys
from src.resnet_lddmm.resnet_flow.io import Shape
from src.resnet_lddmm.resnet_flow.landmarks import resolve_template_points
from src.resnet_lddmm.resnet_flow.landmark_fields import (
    DEFAULT_DECAY_SCALE,
    DEFAULT_MAX_WEIGHT,
    LandmarkEntry,
    axis_bounds,
    combine_peaked_gaussians,
    mahalanobis_norm,
    parse_include_entries,
)

DEFAULT_HEATMAP_PREFIX = "heatmap_"
DEFAULT_OUTPUT_FIELD = "landmark_weight"

_ALLOWED_KEYS = {"vtp", "include", "decay_scale", "max_weight", "min_weight",
                 "output_field", "heatmap_prefix", "output"}


@dataclass
class LandmarkWeightFieldCfg:
    """One run of ``scripts/write_landmark_weight_field.py``.

    Attributes:
        vtp: template mesh carrying the ``heatmap_<label>`` fields (and the one
            this run writes ``output_field`` onto).
        entries: landmarks to use, each with its own resolved ``decay_scale``
            and ``max_weight`` (see ``landmark_fields.LandmarkEntry``).
            Whatever label is NOT here is simply never consulted -- there is
            no separate ``exclude``, since a whitelist already excludes
            everything else. Built from the YAML ``include`` list, where each
            item is either a plain label string (uses the top-level
            ``decay_scale``/``max_weight`` defaults) or a
            ``{label, decay_scale, max_weight}`` mapping (overrides either
            default, or both, for that one landmark).
        min_weight: floor applied after the decay, so a vertex arbitrarily far
            from every included landmark still gets a non-zero weight instead
            of being ignored completely.
        output_field: name of the point field this run writes.
        heatmap_prefix: prefix of the per-label fields ``entries`` are looked
            up under.
    """

    vtp: str
    entries: List[LandmarkEntry]
    min_weight: float = 0.0
    output_field: str = DEFAULT_OUTPUT_FIELD
    heatmap_prefix: str = DEFAULT_HEATMAP_PREFIX
    output: Optional[str] = None  # None = overwrite `vtp` in place

    @property
    def include(self) -> List[str]:
        """Labels only, in ``entries`` order -- for display and lookup."""
        return [entry.label for entry in self.entries]

    @classmethod
    def from_dict(cls, d: dict) -> "LandmarkWeightFieldCfg":
        d = dict(d)
        reject_unknown_keys(d, _ALLOWED_KEYS, what="landmark weight field config")

        if not d.get("vtp"):
            raise ValueError("landmark weight field config needs a non-empty 'vtp'")

        raw_include = d.get("include")
        if not raw_include:
            raise ValueError(
                "landmark weight field config needs a non-empty 'include' list "
                "of landmark labels"
            )

        default_decay_scale = float(d.get("decay_scale", DEFAULT_DECAY_SCALE))
        if default_decay_scale <= 0:
            raise ValueError(f"decay_scale must be > 0, got {default_decay_scale}")

        default_max_weight = float(d.get("max_weight", DEFAULT_MAX_WEIGHT))
        if default_max_weight <= 0:
            raise ValueError(f"max_weight must be > 0, got {default_max_weight}")

        entries = parse_include_entries(raw_include, default_decay_scale, default_max_weight)

        min_weight = float(d.get("min_weight", cls.min_weight))
        if not (0.0 <= min_weight <= 1.0):
            raise ValueError(f"min_weight must be in [0, 1], got {min_weight}")

        return cls(
            vtp=d["vtp"],
            entries=entries,
            min_weight=min_weight,
            output_field=d.get("output_field", cls.output_field),
            heatmap_prefix=d.get("heatmap_prefix", cls.heatmap_prefix),
            output=d.get("output"),
        )


def compute_landmark_weight_field(shape: Shape, cfg: LandmarkWeightFieldCfg) -> torch.Tensor:
    """Per-vertex weight: max over included landmarks of that landmark's own
    peaked gaussian EUCLIDEAN decay, floored.

    Each landmark contributes
    ``max_weight * exp(-0.5 * mahalanobis_norm(vertex - landmark, axis_bounds(entry)) ** 2)``
    everywhere on the mesh; a vertex's weight is the strongest of those. With
    every landmark isotropic (no ``decay_axes``) and sharing one
    ``decay_scale``/``max_weight``, this is exactly equivalent to "gaussian
    decay from the nearest included landmark" (the original, single-scale
    behaviour), since a shared peak means the landmark with the smallest
    distance already gives the largest term. ``max_weight`` is what lets one
    REGION outrank another beyond its own halo: a landmark with a high
    ``max_weight`` can still dominate a nearer landmark that has a lower one.
    ``decay_axes`` (see ``LandmarkEntry``) lets a landmark reach further in
    one axis/direction than another -- e.g. further UP than down.

    Args:
        shape: the template Shape ``cfg.entries``' heatmap fields live on.
        cfg: see ``LandmarkWeightFieldCfg``.

    Returns:
        [N] tensor, one weight per vertex, in
        ``[cfg.min_weight, max(entry.max_weight for entry in cfg.entries)]``.
    """
    landmark_points = resolve_template_points(shape, cfg.include, cfg.heatmap_prefix)  # [L, 3]
    landmarks = torch.tensor(landmark_points, dtype=torch.float32)

    vertices = shape.points[0].detach().to(torch.float32)  # [N, 3]
    diff = vertices[:, None, :] - landmarks[None, :, :]  # [N, L, 3]

    distance = torch.stack([
        mahalanobis_norm(diff[:, i, :], axis_bounds(entry))
        for i, entry in enumerate(cfg.entries)
    ], dim=1)  # [N, L]

    weight = combine_peaked_gaussians(distance, cfg.entries)  # [N]
    return weight.clamp_min(cfg.min_weight)
