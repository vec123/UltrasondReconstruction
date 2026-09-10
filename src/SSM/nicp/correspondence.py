"""Correspondence estimation: which model vertex is paired with which target point.

A :class:`CorrespondenceStrategy` maps (model vertices, target cloud) onto a
:class:`Correspondences` set, or onto ``None`` when the fit is meant to run
without an explicit pairing at all. Strategies are selected by name from a
registry, so the config picks the method and passes its keyword arguments:

    correspondence:
      name: constrained
      kwargs: {dist_thresh: 0.0026, normal_thresh_deg: 90.0}

Three methods ship here:

    none          no pairing is produced (``estimate`` returns ``None``). The
                  data term has to associate the surfaces itself - a Chamfer
                  style loss that re-resolves neighbours every iteration.
    nearest       plain 1-NN for every model vertex, nothing rejected.
    constrained   1-NN among the k closest, rejecting matches that are too far
                  away or whose normals disagree.

``None`` and an empty match set mean different things and are kept distinct:
``None`` is "no pairing was asked for", empty is "a pairing was asked for and
nothing survived the filters", which is a configuration error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from src.SSM.nicp.cloud import PointCloud
from src.SSM.nicp.registry import Registry, RegistryError

#: kept as the public name; the registry raises this for correspondence failures
CorrespondenceError = RegistryError


# --------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Correspondences:
    """Matched (model vertex, target point) index pairs and their distances.

    ``eq=False``: the generated __eq__ would compare the arrays element-wise and
    then call bool() on the result, which numpy refuses. Compare ``as_array()``.

    ``weight`` lets a matcher express confidence per pair; every strategy here
    emits ones, but a robust or curvature-weighted matcher can fill it in and
    the data term will pick it up without any other change.
    """

    source_idx: np.ndarray      # (P,) int64, index into the model vertices
    target_idx: np.ndarray      # (P,) int64, index into the target points
    distance: np.ndarray        # (P,) float64
    weight: np.ndarray          # (P,) float64
    n_source: int = 0           # model vertices offered to the matcher
    n_target: int = 0           # target points available to it

    def __post_init__(self):
        sizes = {
            "source_idx": self.source_idx.shape,
            "target_idx": self.target_idx.shape,
            "distance": self.distance.shape,
            "weight": self.weight.shape,
        }
        if len({s for s in sizes.values()}) != 1:
            raise CorrespondenceError(f"correspondence arrays disagree in length: {sizes}")

    @classmethod
    def from_arrays(cls, source_idx, target_idx, distance, weight=None,
                    n_source: int = 0, n_target: int = 0) -> "Correspondences":
        source_idx = np.asarray(source_idx, dtype=np.int64).reshape(-1)
        target_idx = np.asarray(target_idx, dtype=np.int64).reshape(-1)
        distance = np.asarray(distance, dtype=np.float64).reshape(-1)
        weight = (
            np.ones_like(distance) if weight is None
            else np.asarray(weight, dtype=np.float64).reshape(-1)
        )
        return cls(source_idx, target_idx, distance, weight,
                   int(n_source), int(n_target))

    @classmethod
    def from_triples(cls, triples, n_source: int = 0, n_target: int = 0) -> "Correspondences":
        """Adopts the ``[(model_idx, target_idx, distance), ...]`` list format."""
        arr = np.asarray(list(triples), dtype=np.float64).reshape(-1, 3)
        return cls.from_arrays(arr[:, 0], arr[:, 1], arr[:, 2],
                               n_source=n_source, n_target=n_target)

    def as_array(self) -> np.ndarray:
        """(P, 3) float array of (model_idx, target_idx, distance) rows."""
        return np.column_stack(
            [self.source_idx.astype(np.float64), self.target_idx.astype(np.float64), self.distance]
        )

    def as_triples(self) -> List[Tuple[int, int, float]]:
        return [(int(i), int(j), float(d))
                for i, j, d in zip(self.source_idx, self.target_idx, self.distance)]

    def __array__(self, dtype=None) -> np.ndarray:
        # lets np.asarray(corr) and anything built on it (save_matching_field,
        # ShapeInstance.l2_closed_form_step) take a Correspondences object directly
        arr = self.as_array()
        return arr if dtype is None else arr.astype(dtype)

    def __len__(self) -> int:
        return int(self.source_idx.size)

    def __iter__(self) -> Iterator[Tuple[int, int, float]]:
        return iter(self.as_triples())

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def coverage(self) -> float:
        """Fraction of the offered model vertices that found a match."""
        if not self.n_source:
            return float("nan")
        return len(np.unique(self.source_idx)) / float(self.n_source)

    def summary(self) -> str:
        if self.is_empty:
            return f"0 matches (of {self.n_source} model vertices)"
        return (
            f"{len(self)} matches, {self.coverage:.1%} of {self.n_source} model vertices, "
            f"distance mean {self.distance.mean():.3e} max {self.distance.max():.3e}"
        )


# --------------------------------------------------------------------------
# strategies
# --------------------------------------------------------------------------

class CorrespondenceStrategy(ABC):
    """Pairs model vertices with target points (or declines to)."""

    #: registry key, set by @register_correspondence
    name = "?"

    @abstractmethod
    def estimate(self, source, target) -> Optional[Correspondences]:
        """Returns the pairing, or ``None`` when this method produces none.

        ``source``/``target`` accept raw ``(N, 3)`` / ``(3N,)`` arrays or an
        existing :class:`~src.nicp.cloud.PointCloud`; passing clouds reuses
        their cached normals and KD-tree.
        """

    @property
    def produces_matches(self) -> bool:
        """False when ``estimate`` never returns a pairing (correspondence-free)."""
        return True

    def describe(self) -> str:
        return repr(self)


def _trim(rows, cols, dists, ratio: float):
    """Drops the worst ``ratio`` fraction of pairs by distance.

    The relative counterpart to ``max_dist``: that is an absolute cut that never
    moves, this one always removes the tail whatever the current scale. Early in
    a fit, when everything is far, ``max_dist`` does the work; near convergence
    it is this. Trimmed ICP, in one line.
    """
    if ratio <= 0.0 or rows.size == 0:
        return rows, cols, dists
    keep = max(1, int(round(rows.size * (1.0 - ratio))))
    order = np.argsort(dists)[:keep]
    return rows[order], cols[order], dists[order]


def _deduplicate(rows, cols, dists):
    """Keeps only the closest pair per source row and per target column.

    Two passes, each a distance-sorted ``np.unique``: first collapse every
    target claimed by more than one source down to its closest source, then do
    the same the other way round on what is left. The second pass only removes
    rows, so it cannot reintroduce a target duplicate - what comes out has each
    source and each target at most once, i.e. one-to-one in both directions.

    This is a greedy heuristic, not the optimal one-to-one assignment (that is
    the Hungarian algorithm): a source that loses its closest target in the
    first pass is not given a second-best target, it just goes unmatched. Good
    enough for NICP, where a strategy that re-estimates every few iterations
    (``refresh_every``) self-corrects whatever the greedy choice got wrong.
    """
    if rows.size == 0:
        return rows, cols, dists

    def _best_per_key(keys):
        order = np.argsort(dists)
        _, first = np.unique(keys[order], return_index=True)
        return order[first]

    keep = _best_per_key(cols)
    rows, cols, dists = rows[keep], cols[keep], dists[keep]
    keep = _best_per_key(rows)
    rows, cols, dists = rows[keep], cols[keep], dists[keep]
    return rows, cols, dists


def _normal_agreement(src_normals, tgt_reference, rows, cols, threshold_deg):
    """Mask of pairs whose normals agree to within ``threshold_deg``."""
    dots = np.einsum("ij,ij->i", src_normals[rows], tgt_reference[cols])
    return dots >= np.cos(np.deg2rad(threshold_deg))


CORRESPONDENCE = Registry("correspondence method")
register_correspondence = CORRESPONDENCE.register


def available_correspondences() -> List[str]:
    return CORRESPONDENCE.names()


def build_correspondence(name: str, kwargs: Optional[Dict[str, Any]] = None) -> CorrespondenceStrategy:
    return CORRESPONDENCE.build(name, kwargs)


@register_correspondence("none")
@dataclass(frozen=True)
class NoCorrespondence(CorrespondenceStrategy):
    """Produces no pairing; the data term has to associate the surfaces itself."""

    def estimate(self, source, target) -> Optional[Correspondences]:
        return None

    @property
    def produces_matches(self) -> bool:
        return False


@register_correspondence("nearest")
@dataclass(frozen=True)
class NearestNeighbourMatching(CorrespondenceStrategy):
    """1-NN for every model vertex.

    With ``max_dist`` and ``trim_ratio`` both unset nothing is rejected, so every
    model vertex gets a partner however far away it is - fast, and appropriate
    only when the target covers the whole model.

    ``one_to_one`` rejects many-to-one matches in either direction (see
    :func:`_deduplicate`) - off by default because a plain 1-NN pairing is
    usually many-to-one by construction (several model vertices legitimately
    share a nearest target point on a denser target), and turning it on drops
    every model vertex but the closest one to each target point.
    """

    max_dist: Optional[float] = None
    trim_ratio: float = 0.0
    one_to_one: bool = False

    def __post_init__(self):
        if self.max_dist is not None and self.max_dist <= 0:
            raise ValueError(f"max_dist must be positive or null, got {self.max_dist}")
        if not 0.0 <= self.trim_ratio < 1.0:
            raise ValueError(f"trim_ratio must be in [0, 1), got {self.trim_ratio}")

    def estimate(self, source, target) -> Correspondences:
        src = PointCloud.as_cloud(source)
        tgt = PointCloud.as_cloud(target)

        dists, indices = tgt.query(src.points, k=1)
        dists = dists[:, 0]
        indices = indices[:, 0]

        keep = np.isfinite(dists)
        if self.max_dist is not None:
            keep &= dists <= self.max_dist

        rows = np.flatnonzero(keep)
        cols, dists = indices[rows], dists[rows]
        if self.one_to_one:
            rows, cols, dists = _deduplicate(rows, cols, dists)
        rows, cols, dists = _trim(rows, cols, dists, self.trim_ratio)
        return Correspondences.from_arrays(
            rows, cols, dists, n_source=len(src), n_target=len(tgt)
        )


@register_correspondence("constrained")
@dataclass(frozen=True)
class ConstrainedMatching(CorrespondenceStrategy):
    """Nearest neighbour among the ``k`` closest, with distance and normal rejection.

    For each model vertex the candidates are walked in order of increasing
    distance and the first one that is closer than ``max_dist`` and whose normal
    agrees with the model normal to within ``normal_thresh_deg`` is taken.
    Vertices with no acceptable candidate stay unmatched, which is what keeps the
    model from being dragged onto a partially observed target.

    ``use_target_normals=False`` reproduces the original script, which compared
    the model normals against the target *positions* instead of the target
    normals. It is kept for reproducing old runs, not because it is meaningful.

    ``one_to_one`` additionally rejects many-to-one matches in either direction
    (see :func:`_deduplicate`): each target point ends up claimed by at most
    one model vertex, and vice versa, rather than several model vertices
    piling onto the same target point.
    """

    max_dist: float = 0.4
    normal_thresh_deg: float = 90.0
    k: int = 5
    use_target_normals: bool = True
    trim_ratio: float = 0.0
    one_to_one: bool = False

    def __post_init__(self):
        if self.max_dist <= 0:
            raise ValueError(f"max_dist must be positive, got {self.max_dist}")
        if not 0.0 <= self.normal_thresh_deg <= 180.0:
            raise ValueError(f"normal_thresh_deg must be in [0, 180], got {self.normal_thresh_deg}")
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if not 0.0 <= self.trim_ratio < 1.0:
            raise ValueError(f"trim_ratio must be in [0, 1), got {self.trim_ratio}")

    def estimate(self, source, target) -> Correspondences:
        src = PointCloud.as_cloud(source)
        tgt = PointCloud.as_cloud(target)

        dists, indices = tgt.query(src.points, k=self.k)

        # candidates within range; scipy pads with inf when k exceeds the cloud
        valid = np.isfinite(dists) & (dists <= self.max_dist)

        src_normals = src.normals
        reference = tgt.normals if self.use_target_normals else tgt.points
        # dot(model normal, reference vector) per candidate -> (N, k)
        dots = np.einsum("ij,ikj->ik", src_normals, reference[indices])
        valid &= dots >= np.cos(np.deg2rad(self.normal_thresh_deg))

        # first accepted candidate per vertex (query returns them sorted by distance)
        has_match = valid.any(axis=1)
        first = np.argmax(valid, axis=1)

        rows = np.flatnonzero(has_match)
        cols = first[rows]
        cols, dists = indices[rows, cols], dists[rows, cols]
        if self.one_to_one:
            rows, cols, dists = _deduplicate(rows, cols, dists)
        rows, cols, dists = _trim(rows, cols, dists, self.trim_ratio)
        return Correspondences.from_arrays(
            rows, cols, dists, n_source=len(src), n_target=len(tgt),
        )


@register_correspondence("symmetric")
@dataclass(frozen=True)
class SymmetricMatching(CorrespondenceStrategy):
    """Nearest neighbours in *both* directions, each direction weighted.

    This is the Chamfer distance expressed as a pairing rather than as a loss:

        forward  (model -> target)  every model vertex reaches for the scan
        backward (target -> model)  every scan point must be explained

    The two are not interchangeable when the target is a partial scan. Model
    vertices outside the scanned region have no counterpart, so the forward
    direction invents attraction for them - which is what ``max_dist`` and
    ``trim_ratio`` are for. The backward direction has no such problem and is
    usually the one to trust, hence a larger ``backward_weight``.

    Because it emits ordinary weighted pairs, the closed-form solver handles it:
    with the pairing held fixed, a Chamfer fit *is* a weighted least-squares
    problem, so it needs one linear solve per re-pairing rather than a few
    hundred gradient steps.
    """

    forward_weight: float = 1.0
    backward_weight: float = 1.0
    max_dist: Optional[float] = None
    trim_ratio: float = 0.0
    normal_thresh_deg: Optional[float] = None   # None = no normal rejection
    # Thin the clouds before pairing. Purely a compute knob - the search is
    # linear in both, so a quarter of one against a third of the other is a
    # twelfth of the work. Redrawn on every call, so a fit that re-pairs each
    # iteration still sees all of both surfaces over its run. 0 = use all.
    source_ratio: float = 0.0
    target_ratio: float = 0.0
    seed: Optional[int] = None

    def __post_init__(self):
        if self.forward_weight < 0 or self.backward_weight < 0:
            raise ValueError("direction weights must be >= 0")
        if self.forward_weight == 0 and self.backward_weight == 0:
            raise ValueError("both direction weights are 0; nothing would be matched")
        if self.max_dist is not None and self.max_dist <= 0:
            raise ValueError(f"max_dist must be positive or null, got {self.max_dist}")
        if not 0.0 <= self.trim_ratio < 1.0:
            raise ValueError(f"trim_ratio must be in [0, 1), got {self.trim_ratio}")
        if self.normal_thresh_deg is not None and not 0.0 <= self.normal_thresh_deg <= 180.0:
            raise ValueError(f"normal_thresh_deg must be in [0, 180] or null, got {self.normal_thresh_deg}")
        for ratio, label in ((self.source_ratio, "source_ratio"), (self.target_ratio, "target_ratio")):
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(f"{label} must be in [0, 1] (0 = use all), got {ratio}")

    def _direction(self, src, tgt, rows, cols, dists):
        """Applies the rejection rules to one direction's pairs."""
        keep = np.isfinite(dists)
        if self.max_dist is not None:
            keep &= dists <= self.max_dist
        if self.normal_thresh_deg is not None:
            keep &= _normal_agreement(src.normals, tgt.normals, rows, cols, self.normal_thresh_deg)

        rows, cols, dists = rows[keep], cols[keep], dists[keep]
        return _trim(rows, cols, dists, self.trim_ratio)

    def _sample(self, cloud: PointCloud, ratio: float, rng):
        """A random ``ratio`` of the cloud, with the indices needed to map back."""
        total = len(cloud)
        if ratio <= 0.0 or ratio >= 1.0:
            return cloud, np.arange(total)
        n = max(1, int(round(total * ratio)))
        if n >= total:
            return cloud, np.arange(total)
        keep = np.sort(rng.choice(total, size=n, replace=False))
        return PointCloud(cloud.points[keep]), keep

    def estimate(self, source, target) -> Correspondences:
        src_full = PointCloud.as_cloud(source)
        tgt_full = PointCloud.as_cloud(target)

        rng = np.random.default_rng(self.seed)
        src, src_map = self._sample(src_full, self.source_ratio, rng)
        tgt, tgt_map = self._sample(tgt_full, self.target_ratio, rng)

        rows, cols, dists, weights = [], [], [], []

        if self.forward_weight > 0:
            d, j = tgt.query(src.points, k=1)
            r, c, dd = self._direction(
                src, tgt, np.arange(len(src)), j[:, 0], d[:, 0]
            )
            rows.append(r); cols.append(c); dists.append(dd)
            weights.append(np.full(r.size, self.forward_weight))

        if self.backward_weight > 0:
            d, i = src.query(tgt.points, k=1)
            r, c, dd = self._direction(
                src, tgt, i[:, 0], np.arange(len(tgt)), d[:, 0]
            )
            rows.append(r); cols.append(c); dists.append(dd)
            weights.append(np.full(r.size, self.backward_weight))

        # indices are into the sampled clouds; map them back to the originals
        return Correspondences.from_arrays(
            src_map[np.concatenate(rows)], tgt_map[np.concatenate(cols)],
            np.concatenate(dists), weight=np.concatenate(weights),
            n_source=len(src_full), n_target=len(tgt_full),
        )
