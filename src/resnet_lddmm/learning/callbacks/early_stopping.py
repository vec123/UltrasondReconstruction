"""
Early-stopping callback.

"""

from src.learning.callbacks.base import Callback


class EarlyStopping(Callback):
    """Ends the run when a watched metric stops improving.

    ``monitor`` names a key from the validation metrics (e.g. ``"loss"``,
    ``"recon"``). Watching VALIDATION rather than training is the point -- a
    training loss that keeps falling while validation stalls is exactly the
    situation this is for.
    """

    def __init__(self, monitor="loss", patience=5, min_delta=0.0, verbose=True):
        # Cadence is inherited from whoever emits validation events, so this
        # callback does no step-cadence filtering of its own.
        super().__init__(every_n_steps=1)
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best = None
        self.bad_rounds = 0

    def on_validation_end(self, ctx, step, metrics, batch, pred):
        value = metrics.get(self.monitor)
        if value is None:
            return   # metric not produced this run; nothing to watch

        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.bad_rounds = 0
            return

        self.bad_rounds += 1
        if self.bad_rounds >= self.patience:
            reason = (f"early stop at step {step}: val/{self.monitor} did not improve on "
                      f"{self.best:.6f} for {self.bad_rounds} validations")
            if self.verbose:
                print(f"  {reason}")
            ctx.stop(reason)