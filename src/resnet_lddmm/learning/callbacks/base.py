"""Callback base + run context.

Observer / Inversion of Control: the training loop emits events; interested
parties subscribe. The loop does not deciding WHAT happens at step N and only
decides WHEN -- mechanism stays in the orchestrator, policy moves out here.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class TrainingContext:
    """Run-scoped state every callback can read.

    The split of responsibilities is deliberate: run-scoped facts (which stepper,
    where to write, how long) live HERE and are supplied once by whoever starts
    the run, while per-callback POLICY (cadence, patience, how many shapes to
    render) lives on the callback's own constructor. That keeps callback
    construction short and keeps one source of truth for the shared paths.
    """

    stepper: Any
    log_dir: str
    num_steps: int
    # The step number the loop STARTS at -- nonzero only for a run resumed from a
    # checkpoint. num_steps stays the count of steps this run takes, so the last
    # step number is start_step + num_steps - 1.
    start_step: int = 0
    # Every callback attached to this run. ValidationRunner needs it to broadcast
    # on_validation_end to its peers -- a callback that triggers other callbacks.
    callbacks: List["Callback"] = field(default_factory=list)
    # Any callback may set this to end the run early; the orchestrator checks it
    # after each step. This one flag is what lets EarlyStopping exist without the
    # orchestrator knowing early stopping is a concept.
    should_stop: bool = False
    stop_reason: Optional[str] = None

    def stop(self, reason=None):
        """Request that training end after the current step."""
        self.should_stop = True
        self.stop_reason = reason


class Callback:
    """Observer. Override only the hooks you care about; defaults do nothing.

    ``every_n_steps`` is this callback's OWN cadence -- the orchestrator no longer
    holds log_every / save_every / val_every. Use ``self._due(step)`` to honor it.
    """

    def __init__(self, every_n_steps: int = 1):
        self.every_n_steps = every_n_steps

    def _due(self, step: int) -> bool:
        """True when ``step`` lands on this callback's cadence (0 disables)."""
        return bool(self.every_n_steps) and step % self.every_n_steps == 0

    # -- hooks: default no-op bodies matter, so a subclass overrides two, not four --
    def on_train_start(self, ctx):
        ...

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """``metrics``: the composer breakdown plus the total, as {name: float}."""
        ...

    def on_validation_end(self, ctx, step, metrics, batch, pred):
        """Same shape as ``on_step_end`` -- symmetric on purpose, so a callback can
        treat train and validation events with the same code (T11's val/<term>
        mirroring train/<term> is exactly this symmetry, one layer down)."""
        ...

    def on_train_end(self, ctx):
        ...