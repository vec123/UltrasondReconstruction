"""Direction rules: a unit vector at each chosen vertex, defined from the template's own landmarks.

A rule names *how* a direction is obtained, never its numbers, so it means the
same thing on any posed or scaled copy of the template - "lateral" is "from aR
to aL" wherever the face sits. Rules are small frozen dataclasses with one
method, ``evaluate(mesh, indices) -> (n, 3)``, and compose (``Tangential``,
``Reversed`` wrap another rule).

Every rule returns unit vectors, or zero vectors where the direction is
undefined (e.g. :class:`AwayFromMidline` exactly on the midline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

import numpy as np

from src.deformation_modes.mesh import TemplateMesh


def _unit(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, float)
    length = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.where(length > 1e-12, vectors / np.maximum(length, 1e-12), 0.0)


class DirectionRule(Protocol):
    def evaluate(self, mesh: TemplateMesh, indices: np.ndarray) -> np.ndarray:
        """(len(indices), 3) unit vectors at those vertices."""

    def describe(self) -> str:
        """Short human-readable form, for panels and logs."""


@dataclass(frozen=True)
class Between:
    """The same direction everywhere: from landmark ``start`` towards landmark ``end``."""

    start: str
    end: str

    def evaluate(self, mesh: TemplateMesh, indices: np.ndarray) -> np.ndarray:
        axis = _unit(mesh.landmark_position(self.end) - mesh.landmark_position(self.start))
        return np.tile(axis, (len(indices), 1))

    def describe(self) -> str:
        return f"{self.start} -> {self.end}"


@dataclass(frozen=True)
class SurfaceNormal:
    """The outward surface normal at each vertex."""

    def evaluate(self, mesh: TemplateMesh, indices: np.ndarray) -> np.ndarray:
        return _unit(mesh.normals[np.asarray(indices, dtype=np.int64)])

    def describe(self) -> str:
        return "surface normal"


@dataclass(frozen=True)
class MidlineNormal:
    """The same direction everywhere: the midline plane's normal, towards the face's left."""

    def evaluate(self, mesh: TemplateMesh, indices: np.ndarray) -> np.ndarray:
        return np.tile(mesh.midline.normal, (len(indices), 1))

    def describe(self) -> str:
        return "midline normal (to the left)"


@dataclass(frozen=True)
class AwayFromMidline:
    """Perpendicular to the midline, pointing away from it on either side - symmetric widening."""

    def evaluate(self, mesh: TemplateMesh, indices: np.ndarray) -> np.ndarray:
        side = np.sign(mesh.midline.signed_distance(mesh.vertices[np.asarray(indices, dtype=np.int64)]))
        return side[:, None] * mesh.midline.normal

    def describe(self) -> str:
        return "away from the midline"


@dataclass(frozen=True)
class AwayFrom:
    """Radially away from a centre: the mean position of one or more landmarks."""

    center: Tuple[str, ...]

    def __post_init__(self):
        if isinstance(self.center, str):
            object.__setattr__(self, "center", (self.center,))
        if not self.center:
            raise ValueError("AwayFrom needs at least one centre landmark")

    def evaluate(self, mesh: TemplateMesh, indices: np.ndarray) -> np.ndarray:
        origin = np.mean([mesh.landmark_position(name) for name in self.center], axis=0)
        return _unit(mesh.vertices[np.asarray(indices, dtype=np.int64)] - origin)

    def describe(self) -> str:
        return f"away from {'+'.join(self.center)}"


@dataclass(frozen=True)
class Fixed:
    """An explicit direction, in the template's own coordinates (not pose-invariant - prefer the others)."""

    vector: Tuple[float, float, float]

    def evaluate(self, mesh: TemplateMesh, indices: np.ndarray) -> np.ndarray:
        return np.tile(_unit(np.asarray(self.vector, float)), (len(indices), 1))

    def describe(self) -> str:
        return f"fixed {tuple(round(float(c), 3) for c in self.vector)}"


@dataclass(frozen=True)
class Tangential:
    """Another rule's direction projected onto the tangent plane - sliding along the surface."""

    rule: DirectionRule

    def evaluate(self, mesh: TemplateMesh, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        vectors = self.rule.evaluate(mesh, indices)
        normals = mesh.normals[indices]
        return _unit(vectors - (vectors * normals).sum(-1, keepdims=True) * normals)

    def describe(self) -> str:
        return f"tangential({self.rule.describe()})"


@dataclass(frozen=True)
class Reversed:
    """Another rule's direction, negated."""

    rule: DirectionRule

    def evaluate(self, mesh: TemplateMesh, indices: np.ndarray) -> np.ndarray:
        return -self.rule.evaluate(mesh, indices)

    def describe(self) -> str:
        return f"-({self.rule.describe()})"
