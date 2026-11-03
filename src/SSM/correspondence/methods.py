"""The correspondence methods, registered by name:

    none          no pairing is produced (``estimate`` returns ``None``). The
                  data term has to associate the surfaces itself - a Chamfer
                  style loss that re-resolves neighbours every iteration.
    nearest       plain 1-NN for every model vertex, nothing rejected.
    constrained   1-NN among the k closest, rejecting matches that are too far
                  away or whose normals disagree.
    symmetric     1-NN in both directions, each direction weighted.

The classes hold and check the settings. The search itself is in
``compute/heuristic_estimate.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.SSM.correspondence.base import (
    Correspondences,
    CorrespondenceStrategy,
    register_correspondence,
)
from src.SSM.correspondence.cloud import PointCloud
from src.SSM.correspondence.compute.heuristic_estimate import (
    constrained_pairs,
    nearest_pairs,
    symmetric_pairs,
)


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
    :func:`~src.SSM.correspondence.compute.heuristic_estimate.deduplicate_pairs`) - off by default because a plain 1-NN pairing is
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
        rows, cols, dists = nearest_pairs(
            src, tgt, max_dist=self.max_dist, trim_ratio=self.trim_ratio,
            one_to_one=self.one_to_one,
        )
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
    (see :func:`~src.SSM.correspondence.compute.heuristic_estimate.deduplicate_pairs`): each target point ends up claimed by at most
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
        rows, cols, dists = constrained_pairs(
            src, tgt, max_dist=self.max_dist, normal_thresh_deg=self.normal_thresh_deg,
            k=self.k, use_target_normals=self.use_target_normals,
            trim_ratio=self.trim_ratio, one_to_one=self.one_to_one,
        )
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

    def estimate(self, source, target) -> Correspondences:
        src = PointCloud.as_cloud(source)
        tgt = PointCloud.as_cloud(target)
        rows, cols, dists, weights = symmetric_pairs(
            src, tgt, forward_weight=self.forward_weight, backward_weight=self.backward_weight,
            max_dist=self.max_dist, trim_ratio=self.trim_ratio,
            normal_thresh_deg=self.normal_thresh_deg, source_ratio=self.source_ratio,
            target_ratio=self.target_ratio, seed=self.seed,
        )
        return Correspondences.from_arrays(
            rows, cols, dists, weight=weights, n_source=len(src), n_target=len(tgt),
        )
