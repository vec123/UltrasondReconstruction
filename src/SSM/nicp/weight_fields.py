"""One direction of a two-sided data term, and the per-point field weighting it.

A Chamfer distance has two halves that mean different things, and on a partial
scan they are not interchangeable:

    source_to_target   every model vertex reaches for the scan. Model vertices
                       outside the scanned region have no counterpart, so this
                       direction invents attraction for them.
    target_to_source   every scan point must be explained by the model. No such
                       problem, and usually the one to trust.

So each direction carries its own ``weight`` and its own ``base_sigma``, and may
additionally be modulated by a per-point field - a confidence, a support count,
a distance to a landmark - so that "how much this point counts" can vary across
the surface rather than being one number for the whole side.

This mirrors ``ChamferSideCfg`` in ``src/resnet_lddmm/resnet_flow/config.py``
deliberately, down to the key names: the two subsystems solve the same problem
and a reader moving between them should not have to learn a second vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class ChamferSide:
    """``weight``, ``base_sigma`` and an optional per-point field, for one direction.

    The precision this direction contributes is ``weight / (2 * base_sigma**2)``
    - the usual Gaussian-noise reading, where ``base_sigma`` is the length scale
    at which a residual costs about one unit. Splitting a single scalar into
    those two is what lets a config say "trust this direction twice as much"
    (weight) separately from "this direction's residuals are measured in
    millimetres, that one's in tenths" (base_sigma).

    Attributes:
        weight: how much this direction counts. 0 switches it off entirely.
        base_sigma: this direction's noise scale, in the units the fit runs in -
            millimetres for an atlas model, model units for a .mat one.
        field_name: point field to read as a per-point multiplier. None (the
            default) weights every point in this direction equally, which is
            plain Chamfer.
        field_source: .vtp the field lives on. None reads it from the cloud this
            direction is indexed by - the target for ``target_to_source``. The
            model side has no fields of its own, so ``source_to_target`` needs
            this whenever it names a field, and the file must describe the
            model's vertices in the model's order.
        field_transform: how to turn raw values into a multiplier.
            None - the values already are the multiplier.
            "gaussian_distance" - the values are a DISTANCE (small near the
            thing that matters, large away from it); converted by
            ``exp(-0.5 * (value / field_scale)**2)``. Used raw, a distance field
            would emphasise the points furthest from the landmark, which is
            exactly backwards.
        field_scale: required when field_transform is set; same units as the
            field's own values.
    """

    weight: float = 1.0
    base_sigma: float = 1.0
    field_name: Optional[str] = None
    field_source: Optional[str] = None
    field_transform: Optional[str] = None
    field_scale: Optional[float] = None

    TRANSFORMS = ("gaussian_distance",)
    KEYS = ("weight", "base_sigma", "field_name", "field_source",
            "field_transform", "field_scale")

    @classmethod
    def from_dict(cls, spec: Optional[Dict[str, Any]], label: str) -> "ChamferSide":
        """Validates one direction's mapping. ``None``/``{}`` gives the defaults."""
        spec = dict(spec or {})
        unknown = sorted(set(spec) - set(cls.KEYS))
        if unknown:
            raise ValueError(
                f"unknown key(s) {unknown} under '{label}'. Accepted: {list(cls.KEYS)}"
            )

        side = cls(
            weight=float(spec.get("weight", 1.0)),
            base_sigma=float(spec.get("base_sigma", 1.0)),
            field_name=spec.get("field_name"),
            field_source=spec.get("field_source"),
            field_transform=spec.get("field_transform"),
            field_scale=(None if spec.get("field_scale") is None
                         else float(spec["field_scale"])),
        )

        if side.weight < 0:
            raise ValueError(f"{label}.weight must be >= 0, got {side.weight}")
        if side.base_sigma <= 0:
            raise ValueError(f"{label}.base_sigma must be > 0, got {side.base_sigma}")
        if side.field_transform is not None:
            if side.field_transform not in cls.TRANSFORMS:
                raise ValueError(
                    f"unknown {label}.field_transform {side.field_transform!r}; "
                    f"expected one of {list(cls.TRANSFORMS)}"
                )
            if side.field_name is None:
                raise ValueError(
                    f"{label}.field_transform is set but {label}.field_name is not; "
                    f"there is nothing to transform"
                )
            if side.field_scale is None or side.field_scale <= 0:
                raise ValueError(
                    f"{label}.field_transform={side.field_transform!r} needs a positive "
                    f"{label}.field_scale - the length the field is measured against"
                )
        return side

    @property
    def precision(self) -> float:
        """``weight / (2 * base_sigma**2)`` - what this direction multiplies by."""
        return self.weight / (2.0 * self.base_sigma ** 2)

    @property
    def active(self) -> bool:
        return self.weight > 0

    def describe(self) -> str:
        bits = [f"weight={self.weight:g}", f"base_sigma={self.base_sigma:g}"]
        if self.field_name is not None:
            bits.append(f"field={self.field_name!r}")
            if self.field_transform is not None:
                bits.append(f"{self.field_transform}(scale={self.field_scale:g})")
        return ", ".join(bits)

    def resolve_field(self, n_expected: int, cloud=None, label: str = "") -> Optional[np.ndarray]:
        """The named field as an ``(n_expected,)`` non-negative weight, or None.

        Read from ``field_source`` when given, otherwise from ``cloud``'s own
        point fields. Either way the values must address exactly the points this
        direction scores, in their order: there is no correspondence machinery
        here, so a mesh that merely resembles the right one would silently pair
        every point with an unrelated weight.
        """
        if self.field_name is None:
            return None

        if self.field_source is not None:
            values, where = _load_field(self.field_source, self.field_name)
        elif cloud is not None:
            values, where = cloud.field(self.field_name), "the target cloud"
        else:
            raise ValueError(
                f"{label}.field_name={self.field_name!r} needs a {label}.field_source: "
                f"the model side carries no point fields of its own. Point it at a .vtp "
                f"with one value per model vertex, in the model's vertex order."
            )

        values = np.asarray(values, dtype=np.float64).squeeze()
        if values.ndim != 1:
            raise ValueError(
                f"{label} field {self.field_name!r} in {where} is not a scalar per-point "
                f"array (shape {values.shape}); a weight field carries one value per point"
            )
        if values.shape[0] != n_expected:
            raise ValueError(
                f"{label} field {self.field_name!r} in {where} has {values.shape[0]} "
                f"entries but must address {n_expected} points"
            )
        if (values < 0).any():
            raise ValueError(
                f"{label} field {self.field_name!r} in {where} has negative values; "
                f"a weight field must be >= 0"
            )

        if self.field_transform == "gaussian_distance":
            return np.exp(-0.5 * (values / self.field_scale) ** 2)
        return values


def _load_field(path: str, name: str):
    """One named point array from a .vtp, plus a description of where it came from."""
    # Local import: vtk is heavy and only this branch needs it.
    from src.vtk.io import load_vtp

    polydata = load_vtp(str(path))
    point_data = polydata.GetPointData()
    available = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]
    array = point_data.GetArray(name)
    if array is None:
        raise ValueError(
            f"point field {name!r} not found in {path}; available: {sorted(a for a in available if a)}"
        )
    from vtk.util.numpy_support import vtk_to_numpy

    return vtk_to_numpy(array), str(path)
