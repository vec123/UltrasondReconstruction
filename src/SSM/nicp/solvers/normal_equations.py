"""The exact one-solve minimiser for a matched squared-distance term plus a quadratic prior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import torch
from src.SSM.nicp.problem import FitProblem
from src.shared.core.registry import RegistryError

from .base import SolveResult, Solver, _format, _should_refresh, register_solver


@register_solver("normal_equations")
@dataclass(frozen=True)
class NormalEquationsSolver(Solver):
    """Closed-form Gauss-Newton step on a matched squared-distance term.

    Minimises ``||Phi_I (alpha + dalpha) - y_I||^2 + (alpha+dalpha)^T D (alpha+dalpha)``
    over the matched rows, which gives

        (Phi_I^T Phi_I + D + damping*I) dalpha = Phi_I^T r - D alpha

    with ``D`` the prior's diagonal. The prior therefore penalises the *total*
    coefficients, exactly as it does under the gradient solver - the two are the
    same regularisation expressed two ways. ``damping`` is the separate
    Levenberg-Marquardt knob that shortens the step without pulling alpha
    anywhere.

    The data term is weighted and normalised exactly as ``matched_l2`` is, so
    what this minimises is what gets reported, and a prior weight means the same
    number as it does under the gradient solver. Mind the reduction, though:
    with ``sum`` the data term grows with the number of matched vertices, so a
    prior weight is measured against that total - which is the convention the
    pipeline's gamma values were tuned in. With ``mean`` the same gamma is
    effectively multiplied by the match count.

    The ``- D alpha`` term vanishes at alpha = 0, so starting from the mean with
    ``iterations: 1``, ``reduction: sum`` and an isotropic prior this reduces to
    ``(Phi_I^T Phi_I + gamma I) dalpha = Phi_I^T r`` - precisely
    ShapeInstance.l2_closed_form_step, gamma and all.
    """

    iterations: int = 1
    learning_rate: float = 1.0
    damping: float = 0.0

    def __post_init__(self):
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if self.damping < 0:
            raise ValueError(f"damping must be >= 0, got {self.damping}")

    def validate(self, problem: FitProblem) -> None:
        if problem.loss.name != "matched_l2":
            raise RegistryError(
                f"solver 'normal_equations' minimises a matched squared-distance term in "
                f"closed form and cannot take loss '{problem.loss.name}'. Use solver "
                f"'gradient' for that loss, or loss 'matched_l2' for this solver."
            )
        if problem.prior.quadratic_diagonal(problem.space) is None:
            raise RegistryError(
                f"solver 'normal_equations' needs a quadratic prior; '{problem.prior.name}' "
                f"is not expressible as one. Use solver 'gradient'."
            )

    def solve(self, problem: FitProblem, log=None, on_iteration=None) -> SolveResult:
        self.validate(problem)
        diagonal = problem.prior.quadratic_diagonal(problem.space)

        space = problem.space
        ctx = problem.prepare(self.iterations)
        alpha = problem.alpha0.clone()
        history: List[Dict[str, float]] = []
        last_terms: Dict[str, float] = {}

        for iteration in range(self.iterations):
            ctx.iteration = iteration
            # Emitted before the step, so the snapshot is the state this
            # iteration starts from and index 0 is the initial shape. The terms
            # are the ones evaluated at that alpha at the end of the previous
            # round; at index 0 nothing has been evaluated yet.
            if on_iteration:
                on_iteration(iteration, alpha, dict(last_terms))
            if _should_refresh(problem, iteration):
                ctx.correspondences = problem.estimate_correspondences(alpha)

            corr = ctx.require_correspondences(problem.loss.name)
            verts = space.reconstruct(alpha)

            source = torch.as_tensor(corr.source_idx, dtype=torch.long, device=space.device)
            target = torch.as_tensor(corr.target_idx, dtype=torch.long, device=space.device)

            Phi_I = space.phi_rows(corr.source_idx)                       # (3P, K)
            residual = (problem.target_points[target] - verts[source]).reshape(-1)

            # Weight and normalise exactly as the loss does, so the closed form
            # is the minimiser of the objective that gets reported - not of a
            # differently scaled cousin of it. Under 'mean' the data term shrinks
            # by 1/sum(w), which is precisely what makes a prior weight mean the
            # same number here as it does under the gradient solver.
            weight = torch.as_tensor(corr.weight, dtype=space.dtype, device=space.device)
            scale = 1.0 / weight.sum() if problem.loss.reduction == "mean" else 1.0
            w_coord = weight.repeat_interleave(3)

            lhs = scale * (Phi_I.T @ (w_coord.unsqueeze(1) * Phi_I)) + torch.diag(diagonal + self.damping)
            rhs = scale * (Phi_I.T @ (w_coord * residual)) - diagonal * alpha

            # The K x K system is solved in float64 whatever the working dtype:
            # at 163 x 163 it is free, and it is the one place where float32
            # rounding would show up directly in the fitted coefficients.
            lhs64, rhs64 = lhs.double(), rhs.double()
            try:
                delta = torch.linalg.solve(lhs64, rhs64)
            except RuntimeError:
                # singular, e.g. fewer matched vertices than modes
                delta = torch.linalg.lstsq(lhs64, rhs64.unsqueeze(1)).solution.squeeze(1)
            delta = delta.to(alpha.dtype)

            alpha = alpha + self.learning_rate * delta
            alpha = problem.projection(alpha, space)

            with torch.no_grad():
                ctx.alpha = alpha
                _, terms = problem.evaluate(alpha)
            terms.update({"iteration": iteration, "matches": len(corr),
                          "step": float(delta.norm() * self.learning_rate)})
            history.append(terms)
            last_terms = terms
            if log:
                log(f"    [{iteration + 1}/{self.iterations}] {_format(terms)} "
                    f"matches={len(corr)}")

        return SolveResult(alpha=alpha, history=history, iterations=self.iterations,
                           converged=True, message="closed form")
