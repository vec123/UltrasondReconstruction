"""Solvers: how the coefficients are moved to reduce the objective.

Two of them, and the choice is dictated by the loss rather than by taste:

    normal_equations  the exact minimiser of a matched squared-distance term
                      plus a quadratic prior, in one linear solve. Reproduces
                      the original pipeline step. Only valid for that objective.
    gradient          torch autograd, for anything - Chamfer, CPD, composites.
                      The forward model is linear in alpha, so each step costs
                      one matrix product; the nearest-neighbour search inside
                      the loss dominates.
    trust_region      either objective, but CONSTRAINED: it minimises the data
                      term subject to a mahalanobis radius, solving for the
                      constraint's Lagrange multiplier each iteration rather
                      than taking a configured prior weight on faith.

Both own their iteration loop, apply the projection after every step, and can
re-estimate correspondences on the way (``refresh_every``), which is what turns
a single matched solve into an ICP.
"""

from __future__ import annotations

from .base import (  # noqa: F401
    SOLVERS,
    SolveResult,
    Solver,
    _format,
    _should_refresh,
    available_solvers,
    build_solver,
    register_solver,
)
from .gradient import GradientSolver  # noqa: F401
from .normal_equations import NormalEquationsSolver  # noqa: F401
from .trust_region import TrustRegionSolver  # noqa: F401
