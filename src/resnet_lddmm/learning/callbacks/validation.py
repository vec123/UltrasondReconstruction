"""
Validation-running callback.

This is the one callback that EMITS events as well as consuming them: 
runs the validation pass, then broadcasts ``on_validation_end`` to every callback on the
context. Keeps the orchestrator out of the validation business entirely --
it doesn't know a validation set exists -- while MetricsRecorder, MetricsPlotter
and GeometryVisualizer all react to validation exactly as they react to a step.
"""

from src.learning.callbacks.base import Callback


class ValidationRunner(Callback):
    """Evaluates on ``val_loader`` at its own cadence and broadcasts the result.

    ``num_val_batches`` bounds how many batches to pull -- the loaders here are
    infinite generators (``OneBatchLoader`` / ``ResamplingGraphLoader``), so
    iterating to exhaustion would never return.
    """

    def __init__(self, val_loader, every_n_steps=100, num_val_batches=1):
        super().__init__(every_n_steps)
        self.val_loader = val_loader
        self.num_val_batches = num_val_batches

    def on_step_end(self, ctx, step, metrics, batch, pred):
        if not self._due(step) or self.val_loader is None:
            return
        val_metrics, last_batch, last_pred = self._evaluate(ctx)
        for callback in ctx.callbacks:
            callback.on_validation_end(ctx, step, val_metrics, last_batch, last_pred)

    def _evaluate(self, ctx):
        """Run the pass with the models in eval mode; returns (metrics, batch, pred)."""
        stepper = ctx.stepper

        # Duck-typing: prefer new protocol (eval method) over legacy (encoder/decoder attrs)
        if hasattr(stepper, "eval"):
            stepper.eval()
        else:
            stepper.encoder.eval()
            stepper.decoder.eval()

        try:
            val_iter = iter(self.val_loader)
            losses = []
            # term -> [running sum, batches that produced it]. Counting per term
            # rather than dividing everything by num_val_batches keeps the mean
            # honest if some batch can't produce a term (e.g. no posterior -> no kl).
            term_totals = {}
            last_batch, last_pred = None, None

            for _ in range(self.num_val_batches):
                batch = next(val_iter)
                pred, loss, breakdown = stepper.eval_step(*batch)
                losses.append(loss)
                for name, value in breakdown.items():
                    total, count = term_totals.get(name, (0.0, 0))
                    term_totals[name] = (total + value, count + 1)
                last_batch, last_pred = batch, pred

            metrics = {"loss": sum(losses) / len(losses)}
            metrics.update({name: total / count
                            for name, (total, count) in term_totals.items()})
            return metrics, last_batch, last_pred
        finally:
            if hasattr(stepper, "train"):
                stepper.train()
            else:
                stepper.encoder.train()
                stepper.decoder.train()