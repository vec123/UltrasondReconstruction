"""Checkpoint-writing callback."""

import os

import torch

from src.learning.callbacks.base import Callback


class CheckpointWriter(Callback):
    """
    Saves encoder / decoder / optimizer state on its own cadence.
    """

    def __init__(self, every_n_steps=100, directory="checkpoints", verbose=True,
                 save_final=True):
        super().__init__(every_n_steps)
        self.directory = directory
        self.verbose = verbose
        self.save_final = save_final
        self._last_step = None

    def on_step_end(self, ctx, step, metrics, batch, pred):
        self._last_step = step
        if self._due(step):
            self.save(ctx, step)

    def on_train_end(self, ctx):
        """Always leave the FINAL weights behind, wherever the run happened to end.

        Cadence alone loses them whenever the last step doesn't land on a multiple
        of ``every_n_steps`` (num_steps=250 with save_every=100 stops at 249, last
        save at 200) -- and early stopping ends a run at an arbitrary step by
        design. ``final.pt`` is written under a fixed name so it is findable
        without knowing where the run stopped.
        """
        if self.save_final and self._last_step is not None:
            self.save(ctx, self._last_step, filename="final.pt")

    def save(self, ctx, step, filename=None):
        checkpoint_dir = os.path.join(ctx.log_dir, self.directory)
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, filename or f"step_{step}.pt")

        # Duck-typing: prefer new protocol (state_dict method) over legacy (encoder/decoder attrs)
        if hasattr(ctx.stepper, "state_dict"):
            state = ctx.stepper.state_dict()
        else:
            state = {
                "encoder": ctx.stepper.encoder.state_dict(),
                "decoder": ctx.stepper.decoder.state_dict(),
                "optimizer": ctx.stepper.optimizer.state_dict(),
            }

        torch.save({"step": step, **state}, path)
        if self.verbose:
            print(f"  checkpoint -> {path}")