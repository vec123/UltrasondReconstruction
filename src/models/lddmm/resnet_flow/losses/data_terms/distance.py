"""The loss_distance_handle of the Chamfer-family data terms: the settings of the
robust penalty and their validation. The penalty itself is src.shared.losses.robust."""

from typing import Optional
import torch
from torch import Tensor
from src.shared.core.config import reject_unknown_keys
from src.shared.losses.robust import ROBUST_KINDS, robust_penalty


class LossDistanceHandle:
    """How per-point squared distances become per-point penalties: one of
    ROBUST_KINDS, plus the breakpoint/slope/tau that shape its tail and the
    power that shapes its interior.

    Holds the five settings of one ``loss_distance_handle`` config block -- kind,
    breakpoint, slope, tau, power -- together with the checks that make them a
    valid combination, so a data term owns ONE object rather than five loose
    attributes and their validation. The maths itself stays in robust_penalty;
    this only decides whether it applies, and with which parameters.

    ``kind`` None and "quadratic" with the default ``power`` 2 both mean plain
    squared distance: calling the handle then returns ``dist_sq`` itself,
    bit-for-bit. "quadratic" with another power is ``d**power`` everywhere.
    """

    KEYS = ("kind", "breakpoint", "slope", "tau", "power")   # of the config block, see from_dict

    def __init__(self, kind: Optional[str] = None, breakpoint: Optional[float] = None,
                 slope: Optional[float] = None, tau: Optional[float] = None,
                 power: float = 2.0):
        """Validate and store one setting.

        Args:
            kind: one of ROBUST_KINDS, or None for plain squared distance.
            breakpoint: the distance (not squared) where behaviour changes from
                the interior regime, in the same normalised length units as
                loss.sigma. Required for every kind other than None/"quadratic".
            slope: "linear" only; see robust_penalty.
            tau: "exponential" only; see robust_penalty.
            power: exponent of the interior penalty ``d**power``; 2 (default) is
                plain squared distance, 1 is L1. See robust_penalty.

        Raises:
            ValueError: unknown kind, a non-quadratic kind without a breakpoint,
                or power < 1.
        """
        # Messages name the CONFIG keys (the loss_distance_handle block's kind and
        # breakpoint), since that is what the user has to change.
        if kind is not None and kind not in ROBUST_KINDS:
            raise ValueError(f"unknown loss_distance_handle kind {kind!r}; expected "
                              f"one of {ROBUST_KINDS} or None")
        if kind not in (None, "quadratic") and breakpoint is None:
            raise ValueError(
                f"loss_distance_handle kind={kind!r} needs a breakpoint -- the "
                f"distance at which its behaviour changes from the interior "
                f"regime. Pass one, or drop the loss_distance_handle block to disable it."
            )
        # Below 1 the interior is concave: its gradient grows WITHOUT BOUND as
        # d -> 0, so already-fitted points would dominate the gradient.
        if power < 1.0:
            raise ValueError(f"loss_distance_handle power must be >= 1, got {power}")
        self.kind = kind
        self.breakpoint = breakpoint
        self.slope = slope
        self.tau = tau
        self.power = power

    @classmethod
    def from_dict(cls, block: Optional[dict]) -> "LossDistanceHandle":
        """A handle from its ``loss_distance_handle`` config block.

        Args:
            block: ``{kind, breakpoint, slope, tau, power}``, or None when there
                is no block at all -- plain squared distance.

        Returns:
            The validated LossDistanceHandle.

        Raises:
            ValueError: not a mapping, an unknown key, no ``kind``, or anything
                __init__ rejects.
        """
        if block is None:
            return cls()
        if not isinstance(block, dict):
            raise ValueError(
                f"loss_distance_handle must be a mapping such as "
                f"{{kind: truncated, breakpoint: 0.04}}, got {block!r}")
        reject_unknown_keys(block, cls.KEYS, what="loss_distance_handle")
        # Without its kind, a block would silently mean plain squared distance
        # while reading as configured.
        if block.get("kind") is None:
            raise ValueError(
                f"loss_distance_handle needs a kind, one of {ROBUST_KINDS}; drop the "
                f"block entirely for plain squared distance. Got {block!r}")
        params = {key: None if block.get(key) is None else float(block[key])
                  for key in ("breakpoint", "slope", "tau", "power")}
        power = 2.0 if params["power"] is None else params["power"]
        return cls(block["kind"], params["breakpoint"],
                   slope=params["slope"], tau=params["tau"], power=power)

    @property
    def has_tail(self) -> bool:
        """Whether a breakpoint splits interior from tail (kind not None/"quadratic")."""
        return self.kind not in (None, "quadratic")

    @property
    def is_quadratic(self) -> bool:
        """Whether distances pass through untouched (no tail, power 2)."""
        return not self.has_tail and self.power == 2.0

    def describe(self) -> str:
        """One line naming the kind and its parameters, for the build-time log."""
        if self.is_quadratic:
            return f"loss_distance_handle kind={self.kind!r} -- plain squared distance"
        if not self.has_tail:
            return (f"loss_distance_handle kind={self.kind!r} power={self.power:g} "
                    f"-- d**{self.power:g} everywhere")
        extra = {"linear": f" slope={self.slope!r}",
                 "exponential": f" tau={self.tau!r}"}.get(self.kind, "")
        inner = "quadratic" if self.power == 2.0 else f"d**{self.power:g}"
        return (f"loss_distance_handle kind={self.kind!r} breakpoint={self.breakpoint:g} "
                f"power={self.power:g}{extra} -- {inner} below the breakpoint, "
                f"{self.kind!r} above it")

    def __call__(self, dist_sq: Tensor) -> Tensor:
        """Per-point penalty, same shape as ``dist_sq``; see robust_penalty.

        A no-op when quadratic -- returns ``dist_sq`` unchanged, so the default is
        bit-for-bit plain Chamfer. Meant for the ALREADY-REDUCED nearest-neighbour
        distance (post ``.min()``), never the full pairwise matrix: every kind is
        monotonic in ``d`` below the point where it starts flattening/dropping,
        so applying it after argmin gives the identical nearest-neighbour
        assignment, for a fraction of the cost.
        """
        if self.is_quadratic:
            return dist_sq
        return robust_penalty(dist_sq, self.kind or "quadratic", self.breakpoint,
                                slope=self.slope, tau=self.tau, power=self.power)

    def regime_fraction(self, dist_sq: Tensor) -> float:
        """Fraction of points beyond the breakpoint; 0.0 when there is no tail.

        See TwoSidedDataTerm.regime_fraction for how to read it.
        """
        if not self.has_tail:
            return 0.0
        d = dist_sq.clamp_min(0).sqrt()
        return (d > self.breakpoint).float().mean().item()
