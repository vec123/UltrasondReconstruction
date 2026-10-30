"""Warm-up of the code penalty (loss.code_reg_warmup)."""

from src.resnet_lddmm.learning.callbacks.base import Callback
from src.resnet_lddmm.learning.losses.composer import LossTerm


class CodeRegWarmup(Callback):
    """Set the composer's code_reg weight from the step: 0 before ``start``, ``weight`` from ``end``.

    Linear in between. An auto-decoder whose flow starts as the identity gets no data
    gradient on its codes at first, so a penalty from step 0 pulls every code to zero
    before the data can shape them, and training sits at the z = 0, v = 0 saddle. With the
    penalty off for the first steps the codes take on the data first.

    Writes the weight used to ``metrics["code_reg/weight"]``.
    """

    def __init__(self, composer, weight: float, start: int, end: int, name: str = "code_reg"):
        super().__init__(every_n_steps=1)
        if start < 0 or end <= start:
            raise ValueError(f"code_reg_warmup needs 0 <= start < end, got start={start}, end={end}")
        if name not in composer.names():
            raise ValueError(f"the loss composer has no {name!r} term")
        self.composer = composer
        self.weight = float(weight)
        self.start = start
        self.end = end
        self.name = name

    def weight_at(self, step: int) -> float:
        return self.weight * min(max((step - self.start) / (self.end - self.start), 0.0), 1.0)

    def _set(self, step: int) -> None:
        self.composer.terms = [LossTerm(n, self.weight_at(step), k) if n == self.name else LossTerm(n, w, k)
                               for n, w, k in self.composer.terms]

    def on_train_start(self, ctx):
        self._set(ctx.start_step)
        print(f"[code_reg] warm-up: weight 0 before step {self.start}, {self.weight:g} from step {self.end}; "
              f"at start {self.weight_at(ctx.start_step):g}")

    def on_step_end(self, ctx, step, metrics, batch, pred):
        if metrics is not None:
            metrics["code_reg/weight"] = self.weight_at(step)
        self._set(step + 1)
