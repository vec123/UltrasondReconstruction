"""Solvers: how the coefficients are moved to reduce the objective.

Two of them, and the choice is dictated by the loss rather than by taste:

    normal_equations  the exact minimiser of a matched squared-distance term
                      plus a quadratic prior, in one linear solve. Reproduces
                      the original pipeline step. Only valid for that objective.
    gradient          torch autograd, for anything - Chamfer, CPD, composites.
                      The forward model is linear in alpha, so each step costs
                      one matrix product; the nearest-neighbour search inside
                      the loss dominates.

Both own their iteration loop, apply the projection after every step, and can
re-estimate correspondences on the way (``refresh_every``), which is what turns
a single matched solve into an ICP.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from src.nicp.problem import FitProblem
from src.nicp.registry import Registry, RegistryError

SOLVERS = Registry("solver")
register_solver = SOLVERS.register


def available_solvers() -> List[str]:
    return SOLVERS.names()


def build_solver(name: str, kwargs: Optional[Dict[str, Any]] = None) -> "Solver":
    return SOLVERS.build(name, kwargs)


@dataclass
class SolveResult:
    alpha: torch.Tensor
    history: List[Dict[str, float]] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    message: str = ""

    @property
    def final(self) -> Dict[str, float]:
        return self.history[-1] if self.history else {}


class Solver(ABC):
    name = "?"

    @abstractmethod
    def solve(self, problem: FitProblem, log=None, on_iteration=None) -> SolveResult:
        """``on_iteration(index, alpha, terms)`` is called after every step.

        It is how the runner snapshots the fit without the solver knowing
        anything about files.
        """

    def describe(self) -> str:
        return repr(self)


def _format(terms: Dict[str, float]) -> str:
    """Objective split into its parts, plus the count-independent residual.

    The total alone is misleading once the pairing is re-estimated: it is a sum
    over a set that changes size, so admitting more pairs raises it even when
    every pair got closer. ``rms`` is the per-pair figure and is the one to
    watch across iterations.
    """
    parts = [f"objective={terms.get('objective', float('nan')):.4e}"]
    data = terms.get("matched_l2", terms.get("chamfer", terms.get("cpd_nll")))
    if data is not None:
        parts.append(f"= data {data:.4e} + prior {terms.get('prior', 0.0):.4e}")
    if "rms" in terms:
        parts.append(f"| rms={terms['rms']:.4e}")
    return " ".join(parts)


def _should_refresh(problem: FitProblem, iteration: int) -> bool:
    """Correspondences are always estimated once, then every ``refresh_every``."""
    if iteration == 0:
        return True
    return problem.refresh_every > 0 and iteration % problem.refresh_every == 0


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
    ShapeInstance.fit_to_matching, gamma and all.
    """

    iterations: int = 1
    learning_rate: float = 1.0
    damping: float = 0.0

    def __post_init__(self):
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if self.damping < 0:
            raise ValueError(f"damping must be >= 0, got {self.damping}")

    def solve(self, problem: FitProblem, log=None, on_iteration=None) -> SolveResult:
        if problem.loss.name != "matched_l2":
            raise RegistryError(
                f"solver 'normal_equations' minimises a matched squared-distance term in "
                f"closed form and cannot take loss '{problem.loss.name}'. Use solver "
                f"'gradient' for that loss, or loss 'matched_l2' for this solver."
            )
        diagonal = problem.prior.quadratic_diagonal(problem.space)
        if diagonal is None:
            raise RegistryError(
                f"solver 'normal_equations' needs a quadratic prior; '{problem.prior.name}' "
                f"is not expressible as one. Use solver 'gradient'."
            )

        space = problem.space
        ctx = problem.prepare(self.iterations)
        alpha = problem.alpha0.clone()
        history: List[Dict[str, float]] = []

        for iteration in range(self.iterations):
            ctx.iteration = iteration
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
            if log:
                log(f"    [{iteration + 1}/{self.iterations}] {_format(terms)} "
                    f"matches={len(corr)}")
            if on_iteration:
                on_iteration(iteration, alpha, terms)

        return SolveResult(alpha=alpha, history=history, iterations=self.iterations,
                           converged=True, message="closed form")


@register_solver("gradient")
@dataclass(frozen=True)
class GradientSolver(Solver):
    """First-order optimisation of any differentiable objective.

    ``lr`` is not comparable across losses: a mean-reduced Chamfer term in this
    model's units produces gradients many orders of magnitude smaller than a
    CPD negative log-likelihood. Read the reported objective over the first few
    iterations and scale from there rather than carrying a value over.
    """

    optimizer: str = "adam"
    lr: float = 1e-3
    iterations: int = 100
    tolerance: float = 0.0        # stop when the relative improvement falls below this
    patience: int = 3             # consecutive iterations below tolerance before stopping
    log_every: int = 10

    OPTIMIZERS = ("adam", "sgd", "lbfgs")

    def __post_init__(self):
        if self.optimizer not in self.OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {list(self.OPTIMIZERS)}, got {self.optimizer!r}"
            )
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")

    def _make_optimizer(self, alpha):
        if self.optimizer == "adam":
            return torch.optim.Adam([alpha], lr=self.lr)
        if self.optimizer == "sgd":
            return torch.optim.SGD([alpha], lr=self.lr)
        return torch.optim.LBFGS([alpha], lr=self.lr, max_iter=1, history_size=20)

    def solve(self, problem: FitProblem, log=None, on_iteration=None) -> SolveResult:
        space = problem.space
        ctx = problem.prepare(self.iterations)

        alpha = problem.alpha0.clone().requires_grad_(True)
        optimizer = self._make_optimizer(alpha)

        history: List[Dict[str, float]] = []
        last_terms: Dict[str, float] = {}
        previous = None
        stalled = 0
        converged = False
        message = "iteration budget exhausted"

        for iteration in range(self.iterations):
            ctx.iteration = iteration
            if _should_refresh(problem, iteration) and problem.correspondence.produces_matches:
                ctx.correspondences = problem.estimate_correspondences(alpha.detach())

            def closure():
                optimizer.zero_grad(set_to_none=True)
                ctx.alpha = alpha
                total, terms = problem.evaluate(alpha)
                total.backward()
                last_terms.clear()
                last_terms.update(terms)
                return total

            optimizer.step(closure)

            with torch.no_grad():
                alpha.copy_(problem.projection(alpha.detach(), space))

            terms = dict(last_terms)
            terms["iteration"] = iteration
            terms["grad_norm"] = float(alpha.grad.norm()) if alpha.grad is not None else float("nan")
            history.append(terms)

            if log and (iteration % max(1, self.log_every) == 0 or iteration == self.iterations - 1):
                log(f"    [{iteration + 1}/{self.iterations}] {_format(terms)} "
                    f"|grad|={terms['grad_norm']:.3e}")
            if on_iteration:
                on_iteration(iteration, alpha.detach(), terms)

            current = terms["objective"]
            if self.tolerance > 0 and previous is not None:
                improvement = abs(previous - current) / max(abs(previous), 1e-30)
                stalled = stalled + 1 if improvement < self.tolerance else 0
                if stalled >= self.patience:
                    converged = True
                    message = f"relative improvement below {self.tolerance:g} for {stalled} iterations"
                    break
            previous = current

        return SolveResult(alpha=alpha.detach(), history=history,
                           iterations=len(history), converged=converged, message=message)
