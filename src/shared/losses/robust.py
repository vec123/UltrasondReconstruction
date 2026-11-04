"""Robust per-point penalty on squared distances, shared by the Chamfer-family data
terms of both models (src/models/lddmm/.../data_terms, src/models/gpmm/nicp/losses/chamfer.py)."""

from typing import Optional

import torch
from torch import Tensor


ROBUST_KINDS = ("quadratic", "linear", "truncated", "exponential")


def robust_penalty(dist_sq: Tensor, kind: str, breakpoint: float,
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
