"""Per-vertex precision-weight fields for the field-weighted chamfer term.

One small, self-contained concern: turn a NAMED point field on a .vtp file into
a per-vertex weight tensor. Deliberately generic -- it knows nothing about
chamfer, sigma, or landmarks, unlike ``landmarks.py`` -- so anything that needs
"a scalar per vertex, read from a mesh" can reuse it later, not just
``FieldWeightedCDData``.

Unlike ``landmarks.build_landmark_term_kwargs``, the RESULT here needs no unit
conversion through a ``FrameTransform``: a weight field is a dimensionless
MULTIPLIER on a direction's own ``base_sigma``-derived precision, not a length
or a position, so it means the same thing before and after normalisation. A
raw field that IS a length (e.g. a landmark-distance field) still gets
converted -- see ``field_transform`` -- but the conversion is a RATIO against
its own ``field_scale``, so no ``FrameTransform`` enters into that either.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.resnet_lddmm.resnet_flow.io import Shape, load_shape


def resolve_point_weight_field(field_name: Optional[str], field_source: Optional[str],
                                fallback_shape: Shape,
                                field_transform: Optional[str] = None,
                                field_scale: Optional[float] = None,
                                dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
    """A named point field, as a [N] per-vertex weight tensor, or None.

    Args:
        field_name: name of the point field to read. None (the default in
            ChamferSideCfg) means "no field configured" -- returns None, which
            callers treat as a CONSTANT field (every vertex weighted equally).
        field_source: path to the .vtp the field lives on. None reads it from
            ``fallback_shape`` itself -- the run's own, already-loaded
            source/target shape. Given, it is loaded FRESH and must describe
            the same vertices, in the same order, as ``fallback_shape``: this
            function has no correspondence machinery of its own, so a
            different mesh (even a very similar one) would silently pair each
            vertex with an unrelated weight.
        fallback_shape: the run's own NORMALISED shape for whichever direction
            this field belongs to (the source for source_to_target, the target
            for target_to_source) -- point_fields must still be intact, so pass
            the Shape itself, not a stripped batch wrapper.
        field_transform: None (default) uses the raw field values as the
            weight directly. "gaussian_distance" treats the raw values as a
            DISTANCE and converts via
            ``exp(-0.5 * (distance / field_scale) ** 2)`` -- see
            ChamferSideCfg's docstring for why a raw distance needs this
            before it can serve as an importance weight.
        field_scale: required when field_transform is set; same units as the
            field's own raw values.
        dtype: dtype of the returned tensor.

    Returns:
        None if field_name is None. Otherwise a [N] tensor, N =
        fallback_shape.points.shape[1], one non-negative weight per vertex, in
        vertex order.

    Raises:
        ValueError: field_source describes a different vertex count than
            fallback_shape; field_name is missing from the resolved point
            fields; the named field is not a scalar per-vertex array, does not
            address fallback_shape's vertex count, or (before any transform)
            contains negative values.
    """
    if field_name is None:
        return None

    n_expected = fallback_shape.points.shape[1]

    if field_source is None:
        point_fields = fallback_shape.point_fields or {}
        source_desc = "the run's own shape"
    else:
        loaded = load_shape(field_source)
        if loaded.points.shape[1] != n_expected:
            raise ValueError(
                f"field_source {field_source!r} has {loaded.points.shape[1]} vertices, "
                f"but the run's shape has {n_expected}; a point field can only be "
                f"shared between meshes that describe the same vertices, in the "
                f"same order."
            )
        point_fields = loaded.point_fields or {}
        source_desc = field_source

    if field_name not in point_fields:
        raise ValueError(
            f"point field {field_name!r} not found in {source_desc}; available "
            f"fields: {sorted(point_fields)}"
        )

    values = torch.tensor(point_fields[field_name], dtype=dtype)

    if values.dim() != 1:
        raise ValueError(
            f"point field {field_name!r} in {source_desc} is not a scalar "
            f"per-vertex array (shape {tuple(values.shape)}); a chamfer weight "
            f"field must carry exactly one value per vertex."
        )
    if values.shape[0] != n_expected:
        raise ValueError(
            f"point field {field_name!r} in {source_desc} has {values.shape[0]} "
            f"entries but must address {n_expected} vertices."
        )
    if (values < 0).any():
        raise ValueError(
            f"point field {field_name!r} in {source_desc} has negative values; "
            f"a precision-weight field must be >= 0."
        )

    if field_transform is None:
        return values
    if field_transform == "gaussian_distance":
        # ChamferSideCfg.from_dict already requires field_scale > 0 whenever
        # field_transform is set, so no fallback/validation needed here.
        return torch.exp(-0.5 * (values / field_scale) ** 2)

    raise ValueError(f"unknown field_transform {field_transform!r}")


def normalize_unit_mean(values: torch.Tensor, label: str = "weight field") -> torch.Tensor:
    """Rescale a non-negative weight field to mean 1 over all its vertices.

    Keeps the field's SHAPE -- its ratios between vertices -- and drops its
    absolute scale, so a term weighted by it stays on the scale of its
    unweighted counterpart. Also what makes a uniform random subset's mean an
    unbiased estimate of the full-mesh mean of 1.

    Args:
        values: [N] non-negative weights
        label: name used in the error message

    Returns:
        [N] weights with mean exactly 1.

    Raises:
        ValueError: the field's mean is not positive (all zeros), so there is
            no scale to divide out.
    """
    mean = values.mean()
    if not mean > 0:
        raise ValueError(f"{label} has mean {mean.item():.4g}; cannot rescale a field "
                         f"that is zero everywhere to mean 1.")
    return values / mean
