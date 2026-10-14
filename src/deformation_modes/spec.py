"""The mode specification: axes, constraint sets and modes, as immutable values.

Everything is a frozen dataclass of tuples, so a :class:`ModeSet` is a value:
an edit builds a new one (``with_item``, ``without``, ``renamed``), and keeping
the previous one is all an undo needs. It also makes every part hashable, so
resolved selections and compiled fields can be cached by the spec that
produced them.

Selections are stored as *recipes* - an ordered list of operations - not as
vertex lists, so they say how a region was made and can be replayed:

    Brush           geodesic ball around one vertex          (this mesh only)
    NearLandmarks   geodesic balls around named landmarks    (any template with them)
    FieldThreshold  vertices where a point field is above / below a value
    Mirror          add the mirror images across the midline

Each operation adds to (or, with ``subtract``, removes from) the set built so far.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple, Union

import numpy as np

from src.deformation_modes.directions import DirectionRule
from src.deformation_modes.mesh import TemplateMesh

SCHEMA_VERSION = 1

CONSTRAINT_KINDS = ("anchor", "slide", "rigid")
MODE_KINDS = ("displacement", "strain")


# -- selection recipes ----------------------------------------------------------


@dataclass(frozen=True)
class Brush:
    """Geodesic ball of ``radius`` around ``vertex``."""

    vertex: int
    radius: float
    subtract: bool = False

    def region(self, mesh: TemplateMesh) -> np.ndarray:
        return mesh.geodesic_ball(self.vertex, self.radius)

    def describe(self) -> str:
        return f"{'-' if self.subtract else '+'}brush v{self.vertex} r={self.radius:.3g}"


@dataclass(frozen=True)
class NearLandmarks:
    """Geodesic balls of ``radius`` around each named landmark."""

    labels: Tuple[str, ...]
    radius: float
    subtract: bool = False

    def __post_init__(self):
        if isinstance(self.labels, str):
            object.__setattr__(self, "labels", (self.labels,))
        object.__setattr__(self, "labels", tuple(self.labels))
        if not self.labels:
            raise ValueError("NearLandmarks needs at least one landmark label")

    def region(self, mesh: TemplateMesh) -> np.ndarray:
        missing = [label for label in self.labels if label not in mesh.landmarks]
        if missing:
            raise KeyError(f"no landmark(s) {missing}; available: {sorted(mesh.landmarks)}")
        return np.unique(np.concatenate([mesh.geodesic_ball(mesh.landmarks[label], self.radius)
                                         for label in self.labels]))

    def describe(self) -> str:
        return f"{'-' if self.subtract else '+'}near {','.join(self.labels)} r={self.radius:.3g}"


@dataclass(frozen=True)
class FieldThreshold:
    """Vertices whose point field ``field`` is above ``above`` and/or below ``below``."""

    field: str
    above: Optional[float] = None
    below: Optional[float] = None
    subtract: bool = False

    def __post_init__(self):
        if self.above is None and self.below is None:
            raise ValueError("FieldThreshold needs 'above', 'below' or both")

    def region(self, mesh: TemplateMesh) -> np.ndarray:
        if self.field not in mesh.point_fields:
            raise KeyError(f"no point field {self.field!r}; available: {sorted(mesh.point_fields)}")
        values = np.asarray(mesh.point_fields[self.field], float)
        if values.ndim != 1:
            raise ValueError(f"point field {self.field!r} is not scalar")
        keep = np.ones(len(values), bool)
        if self.above is not None:
            keep &= values > self.above
        if self.below is not None:
            keep &= values < self.below
        return np.flatnonzero(keep)

    def describe(self) -> str:
        bounds = " ".join(part for part in (f">{self.above:g}" if self.above is not None else "",
                                            f"<{self.below:g}" if self.below is not None else "") if part)
        return f"{'-' if self.subtract else '+'}{self.field} {bounds}"


@dataclass(frozen=True)
class Mirror:
    """Add the mirror image of everything selected so far."""

    def describe(self) -> str:
        return "+mirror"


Operation = Union[Brush, NearLandmarks, FieldThreshold, Mirror]


@dataclass(frozen=True)
class Selection:
    """An ordered recipe of operations, replayed to get vertex indices."""

    ops: Tuple[Operation, ...] = ()

    def resolve(self, mesh: TemplateMesh) -> np.ndarray:
        """Sorted vertex indices on ``mesh``."""
        mask = np.zeros(mesh.n_vertices, bool)
        for op in self.ops:
            if isinstance(op, Mirror):
                mask[mesh.mirror.index[np.flatnonzero(mask)]] = True
            else:
                mask[op.region(mesh)] = not op.subtract
        return np.flatnonzero(mask)

    def plus(self, op: Operation) -> "Selection":
        return Selection(self.ops + (op,))

    def without_last(self) -> "Selection":
        return Selection(self.ops[:-1])

    def without_steps(self, start: int, stop: int) -> "Selection":
        """The recipe with steps ``start .. stop - 1`` left out (the rest replay as before)."""
        return Selection(self.ops[:start] + self.ops[stop:])

    def describe(self) -> str:
        return ", ".join(op.describe() for op in self.ops) or "(empty)"

    @property
    def mesh_specific(self) -> bool:
        """True if replaying needs this exact mesh (it contains brush strokes)."""
        return any(isinstance(op, Brush) for op in self.ops)


# -- items --------------------------------------------------------------------


@dataclass(frozen=True)
class Axis:
    """A named direction rule that modes refer to."""

    name: str
    rule: DirectionRule


@dataclass(frozen=True)
class ConstraintSet:
    """A named region and what it is held to: anchor (v = 0), slide (n.v = 0) or rigid (no strain)."""

    name: str
    kind: str = "anchor"
    selection: Selection = Selection()
    weight: float = 1.0

    def __post_init__(self):
        if self.kind not in CONSTRAINT_KINDS:
            raise ValueError(f"constraint kind must be one of {CONSTRAINT_KINDS}, got {self.kind!r}")
        if self.weight <= 0:
            raise ValueError(f"constraint weight must be > 0, got {self.weight}")


@dataclass(frozen=True)
class Mode:
    """One deformation primitive.

    displacement: each site moves along ``axis`` (a direction rule) by ``magnitude``.
    strain:       each site stretches by ``stretch`` along a frame whose tangent
                  follows ``axis`` (tangent, bitangent, normal).
    Either way, the field is the smoothest one that does so while honouring
    the named ``constraints`` sets.
    """

    name: str
    kind: str = "displacement"
    sites: Selection = Selection()
    axis: str = ""
    magnitude: float = 1.0
    stretch: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    constraints: Tuple[str, ...] = ()
    sigma: Optional[float] = None       # kernel width override; None = the ModeSet's

    def __post_init__(self):
        if self.kind not in MODE_KINDS:
            raise ValueError(f"mode kind must be one of {MODE_KINDS}, got {self.kind!r}")
        object.__setattr__(self, "stretch", tuple(float(s) for s in self.stretch))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        if len(self.stretch) != 3:
            raise ValueError(f"stretch needs 3 values (tangent, bitangent, normal), got {self.stretch}")
        if self.sigma is not None and self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")


Item = Union[Axis, ConstraintSet, Mode]
GROUPS = ("axes", "constraint_sets", "modes")


@dataclass(frozen=True)
class ModeSet:
    """Everything one template's modes are made of, plus the mesh it was authored on."""

    template: str
    n_vertices: int
    faces_sha1: str
    sigma: float                         # default kernel width, template units
    spacing: float = 0.5                 # constraint-point spacing, as a fraction of sigma
    nu: float = 1e-4                     # constraint softness (see KernelField)
    axes: Tuple[Axis, ...] = ()
    constraint_sets: Tuple[ConstraintSet, ...] = ()
    modes: Tuple[Mode, ...] = ()
    description: str = ""

    def __post_init__(self):
        for group in GROUPS:
            names = [item.name for item in getattr(self, group)]
            if len(set(names)) != len(names):
                raise ValueError(f"duplicate names in {group}: {names}")

    # -- lookup --------------------------------------------------------------

    def group_of(self, item: Item) -> str:
        return {Axis: "axes", ConstraintSet: "constraint_sets", Mode: "modes"}[type(item)]

    def get(self, group: str, name: str) -> Item:
        for item in getattr(self, group):
            if item.name == name:
                return item
        raise KeyError(f"no {group[:-1].replace('_', ' ')} named {name!r}")

    def axis(self, name: str) -> Axis:
        return self.get("axes", name)

    def constraint_set(self, name: str) -> ConstraintSet:
        return self.get("constraint_sets", name)

    def mode(self, name: str) -> Mode:
        return self.get("modes", name)

    def has(self, group: str, name: str) -> bool:
        return any(item.name == name for item in getattr(self, group))

    # -- edits (each returns a new ModeSet) -------------------------------------

    def with_item(self, item: Item) -> "ModeSet":
        """Add ``item``, or replace the one of the same group and name in place."""
        group = self.group_of(item)
        items = list(getattr(self, group))
        for i, existing in enumerate(items):
            if existing.name == item.name:
                items[i] = item
                break
        else:
            items.append(item)
        return replace(self, **{group: tuple(items)})

    def without(self, group: str, name: str) -> "ModeSet":
        """Remove an item; references to a removed constraint set are dropped from modes."""
        kept = tuple(item for item in getattr(self, group) if item.name != name)
        updated = replace(self, **{group: kept})
        if group == "constraint_sets":
            updated = replace(updated, modes=tuple(
                replace(m, constraints=tuple(c for c in m.constraints if c != name))
                for m in updated.modes))
        return updated

    def renamed(self, group: str, old: str, new: str) -> "ModeSet":
        """Rename an item and every reference to it.

        Raises:
            ValueError: ``new`` is already taken in that group.
        """
        if old == new:
            return self
        if self.has(group, new):
            raise ValueError(f"{new!r} is already used in {group}")
        items = tuple(replace(item, name=new) if item.name == old else item
                      for item in getattr(self, group))
        updated = replace(self, **{group: items})
        if group == "axes":
            updated = replace(updated, modes=tuple(
                replace(m, axis=new) if m.axis == old else m for m in updated.modes))
        elif group == "constraint_sets":
            updated = replace(updated, modes=tuple(
                replace(m, constraints=tuple(new if c == old else c for c in m.constraints))
                for m in updated.modes))
        return updated

    def unique_name(self, group: str, stem: str) -> str:
        """``stem``, or ``stem_2``, ``stem_3``, ... - the first free name in ``group``."""
        name, i = stem, 2
        while self.has(group, name):
            name, i = f"{stem}_{i}", i + 1
        return name


def faces_fingerprint(faces: np.ndarray) -> str:
    """SHA-1 of the triangle connectivity - unchanged by any pose or scale of the same mesh."""
    import hashlib
    return hashlib.sha1(np.ascontiguousarray(faces, dtype=np.int64).tobytes()).hexdigest()
