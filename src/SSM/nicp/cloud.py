"""Point cloud with lazily cached geometry.

Correspondence search needs a KD-tree over the target and normals on both
sides. Building them is the expensive part of a match, and the target does not
move while the model is being fitted, so they are computed on first use and
cached here instead of being recomputed per iteration.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.SSM.correspondence.correspondence import build_kdtree
from src.geometry import vertex_normals


class PointCloud:
    """``(N, 3)`` points plus the derived structures a matcher asks for.

    Treat instances as immutable: when the points move (the model shape after a
    fit step, say) build a new cloud rather than mutating one, otherwise the
    cached tree and normals no longer describe the points.
    """

    __slots__ = ("_points", "_faces", "_normals", "_tree")

    def __init__(self, points, faces=None, normals=None):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            # accept the flat (3N,) layout the SSM reconstructs into
            pts = pts.reshape(-1, 3)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must be (N, 3) or (3N,), got {np.shape(points)}")

        self._points = pts
        self._faces = None if faces is None else np.asarray(faces)
        self._tree = None
        self._normals = None
        if normals is not None:
            normals = np.asarray(normals, dtype=np.float64)
            if normals.shape != pts.shape:
                raise ValueError(
                    f"normals must match the points, got {normals.shape} for {pts.shape}"
                )
            self._normals = normals

    @classmethod
    def as_cloud(cls, obj, faces=None) -> "PointCloud":
        """Wraps raw points; passes an existing cloud (and its caches) through."""
        if isinstance(obj, cls):
            return obj
        return cls(obj, faces=faces)

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def faces(self) -> Optional[np.ndarray]:
        return self._faces

    @property
    def normals(self) -> np.ndarray:
        """Unit vertex normals; from the faces when known, else from a k-NN patch fit."""
        if self._normals is None:
            self._normals = vertex_normals(self._points, self._faces)
        return self._normals

    @property
    def tree(self):
        """cKDTree over the points, built once."""
        if self._tree is None:
            self._tree = build_kdtree(self._points)
        return self._tree

    def query(self, points, k: int = 1):
        """k nearest points, distances ascending. ``k`` is clipped to the cloud size."""
        k_eff = max(1, min(int(k), len(self)))
        dists, indices = self.tree.query(np.asarray(points, dtype=np.float64), k=k_eff)
        # scipy drops the k axis when k == 1; keep the shape uniform for callers
        if k_eff == 1:
            dists = np.asarray(dists).reshape(-1, 1)
            indices = np.asarray(indices).reshape(-1, 1)
        return np.asarray(dists), np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return self._points.shape[0]

    def __repr__(self) -> str:
        cached = [n for n, v in (("normals", self._normals), ("tree", self._tree)) if v is not None]
        return f"PointCloud(n={len(self)}, cached={cached or 'none'})"
