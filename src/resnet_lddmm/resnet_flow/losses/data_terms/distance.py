"""Robust per-point penalty on squared distances, shared by the Chamfer-family data terms."""

from typing import Optional
import torch
from torch import Tensor
from src.core.config import reject_unknown_keys


ROBUST_KINDS = ("quadratic", "linear", "truncated", "exponential")


def _robust_penalty(dist_sq: Tensor, kind: str, breakpoint: float,
                     slope: Optional[float] = None, tau: Optional[float] = None,
                     power: float = 2.0, eps: float = 1e-12) -> Tensor:
    """Per-point robust penalty on squared distance: ``d**power`` below
    ``breakpoint``, one of four selectable shapes above it.

    With the default ``power=2`` every kind agrees EXACTLY with plain squared
    distance for ``d <= breakpoint`` (returns ``dist_sq`` itself there, not a
    reconstruction via sqrt -- so the interior regime is bit-for-bit the
    pre-existing Chamfer behaviour). They differ only in the TAIL,
    ``d > breakpoint``, which is where "further away has less importance" is
    actually implemented -- each kind trades a different amount of the far
    point's influence (the gradient dρ/dd) away. Below, ``c = breakpoint``,
    ``v = c**power`` and ``g = power * c**(power - 1)`` are the interior's value
    and slope at the breakpoint (``c**2`` and ``2c`` for the default power):

    - "quadratic": ``d**power`` everywhere, no tail -- with the default power the
      no-op / baseline case, so a run can A/B a loss_distance_handle against plain
      Chamfer via one config value instead of a separate flag.
    - "linear": ``v + slope*(d - c)``. Influence SATURATES at the constant
      ``slope`` instead of growing with ``d``. ``slope=None`` defaults to ``g``,
      matching the interior's slope at the breakpoint -- for power 2 the classic
      C1-continuous Huber. ``slope=0`` flattens the tail to a constant plateau at
      ``v``: every point beyond the breakpoint still counts for that fixed amount,
      but none of them can be pulled on any harder than another.
    - "truncated": ``0`` beyond the breakpoint -- not a plateau, a hard drop from
      ``v`` down to exactly zero. Points past the breakpoint are fully excluded,
      gradient and all, matching trimmed-ICP-style outlier rejection. This is a
      genuine discontinuity in the penalty's VALUE at the breakpoint,
      deliberately -- it is what "excluded" means, not a smoothed approximation
      of it.
    - "exponential": ``v + g*tau*(1 - exp(-(d-c)/tau))``. C1-continuous at the
      breakpoint (value AND slope match the interior), and approaches the finite
      asymptote ``v + g*tau`` as ``d -> inf`` with influence decaying like
      ``exp(-(d-c)/tau)`` -- a genuinely redescending (Welsch-style) kernel: far
      points are not just capped, their pull vanishes. ``tau=None`` defaults to
      ``breakpoint``.

    ``power`` shapes the INTERIOR, i.e. how hard SMALL deviations pull. Squared
    distance (2) has gradient ``2d``, which vanishes as ``d -> 0``, so a small
    residual barely pulls and the regularizer holds it in place. ``power=1`` is
    L1: gradient 1 at every ``d``, so a small residual pulls as hard as a large
    one. The penalty's UNITS change with it (``d``, not ``d**2``), so the
    ``1/(2 sigma**2)`` the caller divides by no longer means what it did --
    retune sigma / base_sigma when changing power.

    A redescending kernel ("exponential") can zero out the gradient from a
    correspondence that is far only because the flow has not reached the right
    pose/alignment YET, not because it is a genuine outlier -- removing exactly the
    long-range signal that would have pulled it there. "linear" and "truncated"
    keep pulling (just less steeply, or not into a plateau), so they carry that
    risk less sharply; prefer them unless "exponential" is specifically wanted.

    Args:
        dist_sq: per-point squared distances, any shape.
        kind: one of ROBUST_KINDS.
        breakpoint: the distance (NOT squared) at which behaviour changes from the
            plain quadratic regime, in the same normalised length units as
            ``loss.sigma``.
        slope: "linear" only; see above.
        tau: "exponential" only; see above.
        power: exponent of the interior penalty ``d**power``; see above.
        eps: floor under the sqrt so ``d`` stays differentiable at ``dist_sq == 0``.

    Returns:
        Per-point penalty, same shape as ``dist_sq``.

    Raises:
        ValueError: unknown ``kind``.
    """
    inner = dist_sq if power == 2.0 else dist_sq.clamp_min(eps).pow(0.5 * power)
    if kind == "quadratic":
        return inner
    if kind not in ROBUST_KINDS:
        raise ValueError(f"unknown robust_kernel {kind!r}; expected one of {ROBUST_KINDS}")

    c = breakpoint
    d = dist_sq.clamp_min(eps).sqrt()
    beyond = d > c
    v = c ** power                  # interior value at the breakpoint
    g = power * c ** (power - 1)    # interior slope at the breakpoint

    if kind == "linear":
        s = g if slope is None else slope
        tail = v + s * (d - c)
    elif kind == "truncated":
        tail = torch.zeros_like(dist_sq)
    else:  # "exponential"
        t = c if tau is None else tau
        tail = v + g * t * (1.0 - torch.exp(-(d - c) / t))

    return torch.where(beyond, tail, inner)


class LossDistanceHandle:
    """How per-point squared distances become per-point penalties: one of
    ROBUST_KINDS, plus the breakpoint/slope/tau that shape its tail and the
    power that shapes its interior.

    Holds the five settings of one ``loss_distance_handle`` config block -- kind,
    breakpoint, slope, tau, power -- together with the checks that make them a
    valid combination, so a data term owns ONE object rather than five loose
    attributes and their validation. The maths itself stays in _robust_penalty;
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
            slope: "linear" only; see _robust_penalty.
            tau: "exponential" only; see _robust_penalty.
            power: exponent of the interior penalty ``d**power``; 2 (default) is
                plain squared distance, 1 is L1. See _robust_penalty.

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
        """Per-point penalty, same shape as ``dist_sq``; see _robust_penalty.

        A no-op when quadratic -- returns ``dist_sq`` unchanged, so the default is
        bit-for-bit plain Chamfer. Meant for the ALREADY-REDUCED nearest-neighbour
        distance (post ``.min()``), never the full pairwise matrix: every kind is
        monotonic in ``d`` below the point where it starts flattening/dropping,
        so applying it after argmin gives the identical nearest-neighbour
        assignment, for a fraction of the cost.
        """
        if self.is_quadratic:
            return dist_sq
        return _robust_penalty(dist_sq, self.kind or "quadratic", self.breakpoint,
                                slope=self.slope, tau=self.tau, power=self.power)

    def regime_fraction(self, dist_sq: Tensor) -> float:
        """Fraction of points beyond the breakpoint; 0.0 when there is no tail.

        See TwoSidedDataTerm.regime_fraction for how to read it.
        """
        if not self.has_tail:
            return 0.0
        d = dist_sq.clamp_min(0).sqrt()
        return (d > self.breakpoint).float().mean().item()
