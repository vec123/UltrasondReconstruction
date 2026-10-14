"""FourierAnnealing: opens a field's Fourier bands on a step schedule (coarse to fine).

Kept out of fields/fourier.py so the field module does not depend on the
training loop; the field only exposes ``encoding.set_alpha``.
"""

from src.resnet_lddmm.learning.callbacks.base import Callback


class FourierAnnealing(Callback):
    """Set ``encoding.alpha`` from the step number: 0 before ``start``, n_e from ``end``.

    Linear in between, so band k opens over the steps where alpha passes k (see
    AnnealedFourierFeatures). The step that just ran used alpha(step); this
    callback then sets alpha(step + 1) for the next one. Append it AFTER the
    CheckpointWriter so a checkpoint stores the alpha its weights were trained
    with.

    Never lowers alpha. A resumed run gets the same value from its step count
    anyway; a warm start that restarts the count at 0 keeps the bands the loaded
    weights were trained with rather than suddenly hiding them.

    Writes the alpha used this step to ``metrics[metric]`` ("fourier/alpha" by
    default; one name per schedule when a field has several).
    """

    def __init__(self, encoding, start: int, end: int, metric: str = "fourier/alpha"):
        """Initialize the schedule.

        Args:
            encoding: an AnnealedFourierFeatures (the field's ``.encoding``)
            start: step at which the first band starts opening
            end: step at which every band is fully open; > start
            metric: metrics key the alpha is logged under

        Raises:
            ValueError: end <= start, or start < 0.
        """
        super().__init__(every_n_steps=1)
        if start < 0 or end <= start:
            raise ValueError(f"fourier_anneal needs 0 <= start < end, got start={start}, end={end}")
        self.encoding = encoding
        self.start = start
        self.end = end
        self.metric = metric

    def alpha_at(self, step: int) -> float:
        """Scheduled alpha for ``step``, in [0, n_e]."""
        progress = (step - self.start) / (self.end - self.start)
        return self.encoding.n_e * min(max(progress, 0.0), 1.0)

    def _advance_to(self, step: int) -> None:
        current = float(self.encoding.alpha)
        self.encoding.set_alpha(max(current, self.alpha_at(step)))

    def on_train_start(self, ctx):
        self._advance_to(ctx.start_step)
        print(f"[fourier] {self.metric}: {self.encoding.n_e} bands, opening over steps "
              f"{self.start}..{self.end}; alpha at start {float(self.encoding.alpha):.3g}")

    def on_step_end(self, ctx, step, metrics, batch, pred):
        if metrics is not None:
            metrics[self.metric] = float(self.encoding.alpha)
        self._advance_to(step + 1)
