"""The mean shape, discretised - the domain the deformation field lives on.

In a GPMM a shape is ``f(p) = u_bar(p) + u(p)``: the deformation is always
*added to the mean*, so the mean shape is not one shape among many, it is the
domain of the process. This object is that domain, evaluated at ``n`` points,
plus the two facts about those points that coordinates alone cannot carry.

Why it is a type rather than a bare ``(n, 3)`` array:

======================  ==========================================  ============================
what                    needed for                                  derivable from the points?
======================  ==========================================  ============================
``points``              every distance                              yes
``mass``                the quadrature measure of the eigenproblem   **no** - needs ``faces``
``metric``              geodesic vs Euclidean distance              **no** - needs ``faces``
======================  ==========================================  ============================

Without it those get threaded through every kernel call as extra arguments,
recomputed per call, or defaulted - and the default *is* the uniform-``d mu``
assumption of §Solving the Eigenvalue Problem, which would then be accepted
without ever having been decided. See :func:`vertex_areas`.

**``points`` aliases the model's mean; it does not copy it.** That is deliberate
- there is one geometry, not two - but it means a transform applied to the mean
must be applied here too. ``StatisticalShapeModel.similarity_transformed`` is
what keeps the pair coherent; hand-rebuilding a model from ``mean``,
``eigenfunctions`` and ``eigenvalues`` alone silently leaves the domain pointing
at the untransformed shape.

**What is deliberately absent.** Landmarks. The model already carries them and
kernels never ask for them, so a second copy here would only be a second thing
to keep in sync. The contract the kernels actually depend on is smaller still:
``src.SSM.kernels.base.KernelDomain`` needs only ``points``, with
``pairwise_distances`` used if present. That is what lets the kernel package be
exercised against a stand-in object and keeps it free of any dependency on this
one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from src.SSM.kernels.base import _euclidean, _same_indices

METRICS = ("euclidean", "geodesic")

#: All-pairs geodesics need an ``n x n`` matrix: 264 MB at the fetus model's
#: 5743 vertices, 7.7 GB at the baby model's 31027. Refusing loudly past this is
#: better than a machine that stops responding for twenty minutes.
MAX_GEODESIC_VERTICES = 10_000


def vertex_areas(points: np.ndarray, faces: Optional[np.ndarray]) -> np.ndarray:
    """Barycentric vertex areas, normalised to sum to one.

    The quadrature measure ``d mu`` of the eigenvalue problem. Each triangle
    contributes a third of its area to each of its corners, which is the
    standard lumped-mass discretisation.

    §Solving the Eigenvalue Problem assumes points "sampled with uniform
    density", giving ``d mu = 1/n``. Mesh vertices are not uniform - the baby
    model's 31027 of them bunch wherever the original remeshing put detail - and
    under the uniform assumption every eigenfunction is pulled toward the dense
    regions. Carrying the real areas is a strict generalisation: it reduces to
    ``1/n`` exactly when the mesh is uniform.

    Without ``faces`` there is no area to compute and the uniform measure is all
    that is available. That is a defensible choice for a point cloud and a poor
    one for a mesh whose faces were simply not passed in, so it is worth being
    sure which case applies.
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]

    if faces is None or len(faces) == 0:
        return np.full(n, 1.0 / n, dtype=np.float64)

    faces = np.asarray(faces, dtype=np.int64)
    v0, v1, v2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    triangle_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    mass = np.bincount(
        faces.reshape(-1),
        weights=np.repeat(triangle_area / 3.0, 3),
        minlength=n,
    ).astype(np.float64)

    total = mass.sum()
    if total <= 0:
        raise ValueError("the mesh has zero total area; check the triangulation")
    return mass / total


@dataclass(frozen=True, eq=False)
class MeanDomain:
    """``n`` evaluation points on the mean shape, with a measure and a metric."""

    points: np.ndarray                       # (n, 3)
    faces: Optional[np.ndarray] = None       # (F, 3), 0-based
    mass: Optional[np.ndarray] = None        # (n,), sums to 1; derived if omitted
    metric: str = "euclidean"

    #: Lazily filled: the all-pairs geodesic matrix, when one is asked for.
    _cache: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        points = np.ascontiguousarray(np.asarray(self.points, dtype=np.float64))
        if points.ndim == 1:
            points = points.reshape(-1, 3)     # accept the flat (3n,) mean layout
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be (n, 3) or (3n,), got {np.shape(self.points)}")
        object.__setattr__(self, "points", points)

        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {list(METRICS)}, got {self.metric!r}")

        faces = self.faces
        if faces is not None:
            faces = np.ascontiguousarray(np.asarray(faces, dtype=np.int64))
            if faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError(f"faces must be (F, 3), got shape {faces.shape}")
            if faces.size and (faces.min() < 0 or faces.max() >= points.shape[0]):
                raise ValueError(
                    f"faces index vertices outside [0, {points.shape[0]}); the "
                    f"triangulation does not belong to these points"
                )
            object.__setattr__(self, "faces", faces)

        # Before computing anything: the geodesic guards are cheap and their
        # failure is about the request, not the data, so they should not be
        # reached only after a mass computation has already failed for its own
        # reasons.
        if self.metric == "geodesic":
            if faces is None:
                raise ValueError("metric='geodesic' needs a triangulation; faces is None")
            if points.shape[0] > MAX_GEODESIC_VERTICES:
                raise ValueError(
                    f"metric='geodesic' builds an all-pairs matrix: {points.shape[0]}^2 "
                    f"entries = {points.shape[0] ** 2 * 8 / 1024 ** 3:.1f} GB, past the "
                    f"{MAX_GEODESIC_VERTICES}-vertex limit. Use metric='euclidean', or "
                    f"decimate."
                )

        if self.mass is None:
            object.__setattr__(self, "mass", vertex_areas(points, faces))
        else:
            mass = np.ascontiguousarray(np.asarray(self.mass, dtype=np.float64).reshape(-1))
            if mass.size != points.shape[0]:
                raise ValueError(f"{mass.size} mass entries for {points.shape[0]} points")
            if np.any(mass < 0):
                raise ValueError("mass entries are areas and must be >= 0")
            object.__setattr__(self, "mass", mass)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_mean(cls, mean, faces=None, metric: str = "euclidean") -> "MeanDomain":
        """From a model's flat ``(3n,)`` mean and its triangulation.

        The usual entry point: ``MeanDomain.from_mean(model.mean, model.faces)``.
        ``points`` is a reshaped **view** of ``mean`` where numpy can manage it,
        so the two do not drift apart.
        """
        return cls(points=np.asarray(mean).reshape(-1, 3), faces=faces, metric=metric)

    def with_metric(self, metric: str) -> "MeanDomain":
        """The same geometry under a different metric. Drops the distance cache."""
        return MeanDomain(points=self.points, faces=self.faces, mass=self.mass, metric=metric)

    def transformed(self, scale: float, rotation, translation) -> "MeanDomain":
        """The domain under a similarity transform.

        Areas scale by ``s^2`` and the normalised mass is therefore unchanged -
        which is the point of normalising it. Distances scale by ``s``, so any
        kernel length scale must be scaled alongside; that coupling is why
        ``StatisticalShapeModel.similarity_transformed`` does both together
        rather than leaving this method to be called on its own.
        """
        R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        t = np.asarray(translation, dtype=np.float64).reshape(3)
        moved = float(scale) * (self.points @ R.T) + t
        return MeanDomain(points=moved, faces=self.faces, mass=self.mass, metric=self.metric)

    # -- size and shape ----------------------------------------------------

    @property
    def n(self) -> int:
        return self.points.shape[0]

    @property
    def n_faces(self) -> int:
        return 0 if self.faces is None else self.faces.shape[0]

    @property
    def diameter(self) -> float:
        """Bounding-box diagonal - the natural unit for a kernel length scale.

        The two models here do not share coordinates (the baby model spans about
        0.013, the fetus atlas is in millimetres), so a sigma is only meaningful
        as a fraction of this.
        """
        return float(np.linalg.norm(self.points.max(axis=0) - self.points.min(axis=0)))

    @property
    def area(self) -> float:
        """Total surface area, before the mass was normalised.

        Worth having because it predicts what a smooth kernel costs: the
        numerical rank of a stationary kernel is roughly ``area / sigma^2``, so
        this says in advance how many modes a given length scale will add.
        """
        if self.faces is None:
            return 0.0
        v0, v1, v2 = (self.points[self.faces[:, i]] for i in range(3))
        return float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())

    # -- distances ---------------------------------------------------------

    def pairwise_distances(self, rows, cols) -> np.ndarray:
        """``(|rows|, |cols|)`` distances under this domain's metric.

        Every stationary kernel routes through here, so switching the metric
        turns all of them into intrinsic kernels with no change to the kernels
        themselves.
        """
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        if self.metric == "geodesic":
            return self._geodesics()[np.ix_(rows, cols)]
        return _euclidean(self.points[rows], self.points[cols],
                          same=_same_indices(rows, cols))

    def cross_distances(self, X, cols) -> np.ndarray:
        """``(|X|, |cols|)`` distances from arbitrary coordinates to domain points.

        Always Euclidean, including under ``metric='geodesic'``: a geodesic
        distance is defined between points *on the surface*, and an arbitrary
        query point is not known to be on it. Making that substitution silently
        would give a continuous inference that disagrees with the model it
        extends. If geodesic queries are wanted, project onto the surface first
        and take the distance from the projected vertex.
        """
        cols = np.asarray(cols, dtype=np.int64)
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        return _euclidean(X, self.points[cols])

    def _geodesics(self) -> np.ndarray:
        """The all-pairs geodesic matrix, computed once and kept."""
        if "geodesics" not in self._cache:
            from src.geometry.geometry import get_all_pairs_geodesic_distance

            self._cache["geodesics"] = np.asarray(
                get_all_pairs_geodesic_distance(self.points, self.faces)
            )
        return self._cache["geodesics"]

    # -- identity ----------------------------------------------------------

    def hash(self) -> str:
        """Content hash over everything that changes a kernel evaluation.

        Keys the build cache. Hashing the bytes rather than a source path is the
        decision recorded in the architecture note: a path plus mtime goes stale
        in both directions, while 121 MB hashes in about a tenth of a second
        against a build measured in seconds.
        """
        digest = hashlib.blake2b(digest_size=16)
        digest.update(self.points.tobytes())
        digest.update(b"|")
        digest.update(b"" if self.faces is None else self.faces.tobytes())
        digest.update(b"|")
        digest.update(self.mass.tobytes())
        digest.update(b"|")
        digest.update(self.metric.encode())
        return digest.hexdigest()

    def describe(self) -> str:
        uniform = np.allclose(self.mass, 1.0 / self.n)
        spread = self.mass.max() / self.mass.min() if self.mass.min() > 0 else float("inf")
        return (
            f"MeanDomain(n={self.n}, faces={self.n_faces}, metric={self.metric}, "
            f"diameter={self.diameter:.4g}, area={self.area:.4g}, "
            f"mass={'uniform' if uniform else f'non-uniform (max/min {spread:.1f}x)'})"
        )

    def __repr__(self) -> str:
        return self.describe()
