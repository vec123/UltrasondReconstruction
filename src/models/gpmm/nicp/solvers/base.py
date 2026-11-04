"""The solver registry, the result type, the ``Solver`` base class and the helpers the solvers share."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from src.models.gpmm.nicp.problem import FitProblem
from src.shared.core.registry import TypedRegistry


SOLVERS = TypedRegistry("solver")


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

    def validate(self, problem: FitProblem) -> None:
        """Raise if this solver cannot minimise ``problem``. Called by solve(),
        and separately by callers that want to fail before any fitting starts."""

    @abstractmethod
    def solve(self, problem: FitProblem, log=None, on_iteration=None) -> SolveResult:
        """``on_iteration(index, alpha, terms)`` is called before every step.

        ``alpha`` is therefore the state the iteration *starts* from, so index
        ``i`` is the fit after exactly ``i`` updates and index 0 is the initial
        shape. ``terms`` is the objective at that same alpha, empty at index 0
        where nothing has been evaluated yet. The final state is not emitted
        here - it is the solve's return value, which the runner writes anyway.

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
