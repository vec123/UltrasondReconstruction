"""Chamfer-family base: a weighted sum of the source-to-target and target-to-source distances."""

from typing import Optional
from torch import Tensor

from .base import DataTerm
from .distance import LossDistanceHandle


class TwoSidedDataTerm(DataTerm):
    """Chamfer-family base: ``w_st * D(source->target) + w_ts * D(target->source)``.

    "source" and "target" are the config's own two shapes (``source:`` and
    ``target:`` in the YAML). The ``pred`` argument of ``forward`` IS the source:
    the source flowed forward, phi(source), which is the only form of it the data
    term ever sees.

    Both weights default to 1.0 -- the plain symmetric Chamfer every subclass
    computed before these knobs existed. Setting one to 0.0 drops that direction
    ENTIRELY: it is never computed, so it costs nothing and builds no graph.

    Zeroing ``source_to_target_weight`` is what a PARTIAL target needs. That
    direction asserts "every source point lies on the target surface", which is
    FALSE when the target is a partial scan -- most of the source has no
    counterpart in it -- and the cheapest way to satisfy it is to collapse the
    whole source into the retained patches. Dropping it leaves only the half that
    is true: target->source, "every OBSERVED point is explained by the source".
    The unobserved region is then shaped by the REGULARIZER alone, so an
    asymmetric data term is only half a fix -- kinetic/isometry become the model
    of everything the scan does not see.

    Under ``direction: bidirectional`` these names stop being literal. One
    instance is called twice with the roles swapped -- as (flowed source, target)
    and as (flowed target, source), see BidirectionalMappingError -- so the
    weights follow the ARGUMENT POSITION, not the shape, and asymmetry drops the
    first argument's direction in both calls. That is not "one-sided
    registration"; asymmetry is meant for ``direction: forward``. The
    field-weighted terms are the exception: the backward call gets
    FieldWeightedCDData.mirrored(), which keeps each side's settings with the
    claim they were configured for.

    No renormalisation by the weight sum, deliberately: (1, 1) has to stay
    bit-for-bit what it was. So zeroing a direction also roughly HALVES the data
    term's magnitude while the kinetic and isometry weights do not change --
    compensate with ``loss.sigma``, which sets the data weight to 1/(2 sigma^2).
    """

    def __init__(self, source_to_target_weight: float = 1.0,
                 target_to_source_weight: float = 1.0,
                 loss_distance_handle: Optional[dict] = None):
        """Initialize the direction weights and the shared LossDistanceHandle.

        The loss_distance_handle block is validated and held as
        ``self.loss_distance_handle``, applied to both directions.

        Args:
            source_to_target_weight: weight on D(phi(source) -> target); 0.0 disables it
            target_to_source_weight: weight on D(target -> phi(source)); 0.0 disables it
            loss_distance_handle: the config block ``{kind, breakpoint, slope,
                tau, power}`` (see LossDistanceHandle.from_dict), or None
                (default) to keep plain squared-distance Chamfer -- bit-for-bit
                the pre-existing behaviour.
                kind: one of ROBUST_KINDS; see _robust_penalty for what each does.
                breakpoint: the distance (not squared) where behaviour changes
                    from the near-zero interior regime, in the same normalised
                    length units as loss.sigma. Required for every kind other
                    than "quadratic".
                slope: "linear" only; None defaults to the interior's slope at
                    the breakpoint (2 * breakpoint for power 2, the
                    C1-continuous Huber). 0 flattens the tail to a constant
                    plateau.
                tau: "exponential" only; None defaults to breakpoint.
                power: the interior is ``d**power``; 2 (default) is squared
                    distance, 1 is L1, which pulls small residuals as hard as
                    large ones. Changes the penalty's units -- retune sigma.
        """
        super().__init__()
        if source_to_target_weight < 0 or target_to_source_weight < 0:
            raise ValueError(
                "direction weights must be >= 0, got source_to_target_weight="
                f"{source_to_target_weight}, target_to_source_weight="
                f"{target_to_source_weight}"
            )
        if source_to_target_weight == 0 and target_to_source_weight == 0:
            raise ValueError(
                "both direction weights are 0 -- the data term would be a constant "
                "zero with no gradient, and training would silently optimize the "
                "regularizers alone. Set at least one > 0."
            )
        self.source_to_target_weight = float(source_to_target_weight)
        self.target_to_source_weight = float(target_to_source_weight)

        self.loss_distance_handle = LossDistanceHandle.from_dict(loss_distance_handle)
        if not self.loss_distance_handle.is_quadratic:
            print(f"[data_term] {self.loss_distance_handle.describe()}")

    def regime_fraction(self, dist_sq: Tensor) -> float:
        """Fraction of points currently beyond the loss_distance_handle's breakpoint.

        The direct way to SEE where the configured breakpoint sits relative to the
        current fit, alongside the printed description at construction time: near
        0 means almost nothing is in the tail yet (breakpoint generous, or the fit
        is already tight); near 1 means most points are in the tail, which means
        either the fit has not converged yet or the breakpoint is too tight for
        the current scale. Returns 0.0 (not an error) when the handle has
        no tail (kind "quadratic"), since then nothing is ever "in the tail".

        Args:
            dist_sq: per-point squared distances (e.g. the same tensor the
                loss_distance_handle is applied to).

        Returns:
            A python float in [0, 1].
        """
        return self.loss_distance_handle.regime_fraction(dist_sq)

    @property
    def scores_source_to_target(self) -> bool:
        """Whether source->target is worth computing at all."""
        return self.source_to_target_weight != 0.0

    @property
    def scores_target_to_source(self) -> bool:
        """Whether target->source is worth computing at all."""
        return self.target_to_source_weight != 0.0

    def _combine(self, source_to_target: Optional[Tensor],
                 target_to_source: Optional[Tensor]) -> Tensor:
        """Weighted sum of the two directions; ``None`` is one the caller skipped.

        A direction with a non-zero weight MUST arrive computed. Passing None
        there is a subclass forgetting to guard on the matching ``scores_*``
        property, which would silently drop half the objective -- assert instead.
        """
        total = None
        if self.scores_source_to_target:
            assert source_to_target is not None, \
                "source->target is weighted but was not computed"
            total = self.source_to_target_weight * source_to_target
        if self.scores_target_to_source:
            assert target_to_source is not None, \
                "target->source is weighted but was not computed"
            weighted = self.target_to_source_weight * target_to_source
            total = weighted if total is None else total + weighted
        return total
