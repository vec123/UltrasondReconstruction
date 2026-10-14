"""VerboseCallback: the loss (and every non-diagnostic metric) printed to the console at its cadence."""

from src.resnet_lddmm.learning.callbacks.base import Callback


class VerboseCallback(Callback):
    """Log loss progression during training."""

    def __init__(self, log_every=1, pose_only_mode=False):
        super().__init__(every_n_steps=log_every)
        self.pose_only_mode = pose_only_mode

    def on_train_start(self, ctx):
        """Print training mode info at startup."""
        if self.pose_only_mode:
            print("[POSE-ONLY MODE] Flow is frozen at identity; encoder learns pose only.")

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Print loss at cadence."""
        if not self._due(step):
            return
        loss = metrics.get("loss", 0.0)
        # Against the LAST step number, not the count: a resumed run starts at
        # ctx.start_step, and "Step 1301/2000" for 2000 further steps reads as a
        # run that is nearly over when it has barely begun.
        last_step = ctx.start_step + ctx.num_steps
        progress = f"Step {step+1:4d}/{last_step} | loss: {loss:.6f}"

        # Add any other metrics
        for key in sorted(metrics.keys()):
            if key != "loss" and not key.startswith("diag/"):
                progress += f" | {key}: {metrics[key]:.6f}"

        print(progress)

    def _due(self, step):
        """Check if we should log this step."""
        return step % self.every_n_steps == 0 or step == 0
