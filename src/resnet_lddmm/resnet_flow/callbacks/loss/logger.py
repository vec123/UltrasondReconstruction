"""LossLogger: the loss breakdown and diagnostics of each logged step, as JSON (loss_logs/)."""

import os
import json
import torch

from src.resnet_lddmm.learning.callbacks.base import Callback


def _as_float(value):
    """Best-effort JSON-serialisable scalar, or None when the value is not one.

    The metrics dict is open: callbacks earlier in the list drop their own entries
    into it, and nothing guarantees they are Python floats. Coercing here keeps a
    stray tensor or array from taking down the whole log write.
    """
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class LossLogger(Callback):
    """Log the composed loss and its per-term breakdown to JSON.

    The console shows the total and whatever the verbose callback selects; this
    keeps the whole breakdown, per step, in a form that can be replayed and plotted
    after the run.

    Output per due step, ``loss_logs/step_%06d.json``::

        {
          "step": 500,
          "total": 6.7412,
          "terms": {
            "data":    {"value": 0.0134, "weight": 50.0, "contribution": 0.6700},
            "kinetic": {"value": 6.0712, "weight":  1.0, "contribution": 6.0712}
          },
          "skipped": ["code_reg", "isometry"],
          "sum_of_contributions": 6.7412,
          "residual": 0.0,
          "diagnostics": {"diag/det_min": 0.83, "grad_norm/flow": 1.7e-2}
        }

    ``value`` is the term as the composer received it, UNWEIGHTED -- which is what
    ``breakdown`` carries, and the usual misreading of these logs, since the values
    do not sum to the total. ``contribution = weight * value`` is what actually
    entered the objective, and ``residual`` (total minus their sum) is the check
    that the weights logged here are the weights that were optimised. It should be
    0 up to float error; anything else means the composer and this log disagree.

    ``skipped`` names terms the composer is configured with that produced no value
    this step. That is the normal way a term switches itself off -- an unavailable
    pose, a mode that does not apply -- and it is worth seeing explicitly, because a
    term silently returning None looks exactly like a term that is satisfied.

    History goes to ``loss_logs/loss_history.jsonl``, one record per line. Not a CSV
    (as ``PoseLogger`` uses) because the key set is NOT fixed across steps: terms
    that return None vanish from the breakdown, so a header written at step 0 would
    be wrong the first time a term appears or drops out. A line-delimited record is
    schema-free and still streams into pandas via ``read_json(..., lines=True)``.

    Place this LAST in the callback list. ``metrics`` is one mutable dict shared by
    the whole list, so running last is what lets DiagnosticsCallback's ``diag/*``
    and GradientMonitor's ``grad_norm/*`` land in ``diagnostics`` rather than being
    written a step late.
    """

    def __init__(self, every_n_steps=50, subdir="loss_logs"):
        """Initialize the logger.

        Args:
            every_n_steps: cadence
            subdir: directory under ctx.log_dir to write into
        """
        super().__init__(every_n_steps)
        self.subdir = subdir
        self.log_dir = None

    def on_train_start(self, ctx):
        """Create the loss_logs directory at the start of training."""
        self.log_dir = os.path.join(ctx.log_dir, self.subdir)
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[LossLogger] Logging losses to {self.log_dir}")

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Write this step's record, if due."""
        if not self._due(step) or self.log_dir is None:
            return

        try:
            record = self._record(ctx, step, metrics)

            with open(os.path.join(self.log_dir, f"step_{step:06d}.json"), "w") as f:
                json.dump(record, f, indent=2)

            with open(os.path.join(self.log_dir, "loss_history.jsonl"), "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[LossLogger] Failed to log losses at step {step}: {e}")
            import traceback
            traceback.print_exc()

    def _record(self, ctx, step, metrics):
        """Split the metrics dict into weighted terms and everything else.

        Args:
            ctx: TrainingContext, read for ctx.stepper.composer
            step: current step
            metrics: {"loss": float, <term>: float, <diagnostic>: float}

        Returns:
            JSON-serialisable dict, as documented on the class
        """
        weights = self._weights(ctx)
        total = _as_float(metrics.get("loss"))

        terms, diagnostics = {}, {}
        for key, value in metrics.items():
            if key == "loss":
                continue
            scalar = _as_float(value)
            if key in weights:
                weight = float(weights[key])
                terms[key] = {
                    "value": scalar,
                    "weight": weight,
                    "contribution": None if scalar is None else weight * scalar,
                }
            else:
                diagnostics[key] = scalar

        contributions = [t["contribution"] for t in terms.values()
                         if t["contribution"] is not None]
        summed = float(sum(contributions))

        return {
            "step": step,
            "total": total,
            "terms": terms,
            "skipped": sorted(set(weights) - set(terms)),
            "sum_of_contributions": summed,
            "residual": None if total is None else total - summed,
            "diagnostics": diagnostics,
        }

    @staticmethod
    def _weights(ctx):
        """{term name: weight} from the stepper's composer ({} if unavailable).

        The composer is the only authority on the weights actually applied. Reading
        them back off the config instead would log what was requested rather than
        what was used, and would miss the derived ones -- ``data`` is 1/(2 sigma^2),
        never a config field.
        """
        composer = getattr(getattr(ctx, "stepper", None), "composer", None)
        if composer is None:
            return {}
        return {name: weight for name, weight, _ in composer.terms}
