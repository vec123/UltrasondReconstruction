"""Correspondence estimation: which model vertex is paired with which target point.

A :class:`CorrespondenceStrategy` maps (model vertices, target cloud) onto a
:class:`Correspondences` set, or onto ``None`` when the fit is meant to run
without an explicit pairing at all. Strategies are selected by name from a
registry, so the config picks the method and passes its keyword arguments:

    correspondence:
      name: constrained
      kwargs: {dist_thresh: 0.0026, normal_thresh_deg: 90.0}

``None`` and an empty match set mean different things and are kept distinct:
``None`` is "no pairing was asked for", empty is "a pairing was asked for and
nothing survived the filters", which is a configuration error.

This module holds the result, the interface and the registry. The methods are
in ``methods.py``, and their search is in ``compute/``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Tuple

import numpy as np

from src.shared.core.registry import RegistryError, TypedRegistry

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
        # lets np.asarray(corr) and anything built on it (save_matching_lines,
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
    #: settings that are distances, so a caller can convert them from another frame
    distance_kwargs: ClassVar[Tuple[str, ...]] = ()

    @abstractmethod
    def estimate(self, source, target) -> Optional[Correspondences]:
        """Returns the pairing, or ``None`` when this method produces none.

        ``source``/``target`` accept raw ``(N, 3)`` / ``(3N,)`` arrays or an
        existing :class:`~src.shared.correspondence.cloud.PointCloud`; passing clouds reuses
        their cached normals and KD-tree.
        """

    @property
    def produces_matches(self) -> bool:
        """False when ``estimate`` never returns a pairing (correspondence-free)."""
        return True

    def describe(self) -> str:
        return repr(self)


CORRESPONDENCE = TypedRegistry("correspondence method")
register_correspondence = CORRESPONDENCE.register


def available_correspondences() -> List[str]:
    return CORRESPONDENCE.names()


def build_correspondence(name: str, kwargs: Optional[Dict[str, Any]] = None) -> CorrespondenceStrategy:
    return CORRESPONDENCE.build(name, kwargs)

