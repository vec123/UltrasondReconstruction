"""ModeAmplitudeLogger: how far each designed deformation mode has been opened.

Kept out of fields/modes.py so the field module does not depend on the
training loop, as callbacks/schedules/fourier.py does for the Fourier bands.
"""

import torch

from src.resnet_lddmm.learning.callbacks.base import Callback


class ModeAmplitudeLogger(Callback):
    """Write ``modes/<name>_mm`` every step: the mode's amplitude averaged over the Euler steps.

    The basis is normalised to a peak of 1 mm per unit time (runner._place_modes),
    so this is roughly the peak displacement the mode contributes over the whole
    flow - 0 means the data never asked for it, a negative value that it runs in
    reverse (e.g. narrowing instead of widening).
    """

    def __init__(self, field):
        """``field``: the run's ModalField."""
        super().__init__(every_n_steps=1)
        self.field = field

    def _values(self):
        with torch.no_grad():
            mean = self.field.amplitude.mean(dim=0).tolist()
        return dict(zip(self.field.names, mean))

    def on_train_start(self, ctx):
        print(f"[modes] amplitudes start at "
              + ", ".join(f"{name} {value:+.3g} mm" for name, value in self._values().items()))

    def on_step_end(self, ctx, step, metrics, batch, pred):
        if metrics is not None:
            for name, value in self._values().items():
                metrics[f"modes/{name}_mm"] = value

    def on_train_end(self, ctx):
        print(f"[modes] final amplitudes: "
              + ", ".join(f"{name} {value:+.3g} mm" for name, value in self._values().items()))
