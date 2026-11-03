"""The facial midline: a plane fitted to landmarks, and the vertex-to-vertex mirror map it induces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
from scipy.spatial import cKDTree

#: Landmarks that sit on the midline of a face, used (with the midpoints of
#: every left/right pair) to place the plane.
MIDLINE_LANDMARKS: Tuple[str, ...] = ("n", "prn", "sn", "ls", "li", "sl", "pg")


def bilateral_pairs(names: Iterable[str], left: str = "L", right: str = "R") -> List[Tuple[str, str]]:
    """(left, right) label pairs sharing a stem, e.g. ('aL', 'aR'), in sorted order."""
    names = set(names)
    return sorted((name, name[:-len(left)] + right) for name in names
                  if name.endswith(left) and name[:-len(left)] + right in names)


@dataclass(frozen=True)
class MidlinePlane:
    """A plane ``(x - point) . normal = 0``; ``normal`` points to the face's left (``L`` labels)."""

    point: np.ndarray
    normal: np.ndarray

    @classmethod
    def fit(cls, positions: Mapping[str, np.ndarray],
            midline: Iterable[str] = MIDLINE_LANDMARKS) -> "MidlinePlane":
        """Fit the plane to named landmark positions.

        The normal is the mean direction from each right landmark to its left
        partner - robust, since every pair votes on it independently. The plane
        then passes through the mean of the pairs' midpoints and of whichever
        ``midline`` landmarks are present.

        Raises:
            ValueError: no left/right landmark pairs to orient the plane by.
        """
        pairs = bilateral_pairs(positions)
        if not pairs:
            raise ValueError(f"no left/right landmark pairs among {sorted(positions)}; "
                             f"cannot orient a midline plane")
        directions = []
        on_plane = []
        for left, right in pairs:
            p_left, p_right = np.asarray(positions[left], float), np.asarray(positions[right], float)
            span = p_left - p_right
            length = np.linalg.norm(span)
            if length > 0:
                directions.append(span / length)
            on_plane.append(0.5 * (p_left + p_right))
        on_plane += [np.asarray(positions[name], float) for name in midline if name in positions]
        normal = np.mean(directions, axis=0)
        normal /= np.linalg.norm(normal)
        return cls(point=np.mean(on_plane, axis=0), normal=normal)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """(n,) distance from the plane, positive on the left side."""
        return (np.asarray(points, float) - self.point) @ self.normal

    def reflect_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, float)
        return points - 2.0 * self.signed_distance(points)[..., None] * self.normal

    def reflect_vectors(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, float)
        return vectors - 2.0 * (vectors @ self.normal)[..., None] * self.normal

    def reflect_frames(self, frames: np.ndarray) -> np.ndarray:
        """Mirror (n, 3, 3) frames with columns (tangent, bitangent, normal).

        Reflection reverses handedness, so tangent and normal are reflected and
        the bitangent is rebuilt as ``normal x tangent`` - the result is a proper
        (right-handed) frame whose tangent and normal are the mirror images.
        """
        frames = np.asarray(frames, float)
        tangent = self.reflect_vectors(frames[..., 0])
        normal = self.reflect_vectors(frames[..., 2])
        return np.stack([tangent, np.cross(normal, tangent), normal], axis=-1)

    def asymmetry(self, positions: Mapping[str, np.ndarray]) -> Dict[str, float]:
        """Per pair, how far the reflected left landmark lands from the right one."""
        return {f"{left}/{right}": float(np.linalg.norm(
                    self.reflect_points(positions[left]) - np.asarray(positions[right], float)))
                for left, right in bilateral_pairs(positions)}


class MirrorMap:
    """Vertex i -> the vertex nearest to i's reflection through a midline plane.

    Only approximately an involution - a mesh is never exactly symmetric -
    which is why :attr:`distance` (how far each reflection landed from its
    match) is kept: large values flag where mirroring is unreliable.
    """

    def __init__(self, vertices: np.ndarray, plane: MidlinePlane):
        vertices = np.asarray(vertices, float)
        self.plane = plane
        self.distance, self.index = cKDTree(vertices).query(plane.reflect_points(vertices))
        self.index = self.index.astype(np.int64)

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        """The mirror images of ``indices`` (unique, sorted)."""
        return np.unique(self.index[np.asarray(indices, dtype=np.int64)])

    def symmetric(self, indices: np.ndarray) -> np.ndarray:
        """``indices`` together with their mirror images (unique, sorted)."""
        indices = np.asarray(indices, dtype=np.int64)
        return np.union1d(indices, self.index[indices])
