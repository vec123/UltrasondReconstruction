"""Autograd solver for any objective: Chamfer, CPD, composites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import torch
from src.models.gpmm.nicp.problem import FitProblem

from .base import SolveResult, Solver, _format, _should_refresh, register_solver


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

            # The closure evaluates at the alpha the step starts from, so this
            # copy is what `last_terms` actually describes - snapshotting it
            # keeps shape and objective in step with each other, and makes index
            # 0 the initial shape.
            started_from = alpha.detach().clone()
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
                on_iteration(iteration, started_from, terms)

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
