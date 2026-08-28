"""
Metric recording and plotting callbacks (INSTRUCTIONS.md T12).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # headless backend: save PNGs without a display / GUI event loop
import matplotlib.pyplot as plt

from src.learning.callbacks.base import Callback


class MetricsRecorder(Callback):
    """Persists every metric event, and keeps the history for whoever plots it.

    Writes APPEND-ONLY JSONL -- one ``{"step", "split", "term", "value"}`` record
    per event.

    ``metrics.json`` is  written ONCE at ``on_train_end`` as a summary
    in  ``{"train/recon": [[step, value], ...]}`` shape. The JSONL is
    the durable stream; the JSON is the convenient end-of-run artifact.
    """

    def __init__(self, every_n_steps=1, filename="metrics.jsonl",
                 summary_filename="metrics.json", verbose=True):
        super().__init__(every_n_steps)
        self.filename = filename
        self.summary_filename = summary_filename
        self.verbose = verbose
        # "<split>/<term>" -> [(step, value), ...]; what MetricsPlotter draws.
        self.history = {}
        self._fh = None

    def on_train_start(self, ctx):
        os.makedirs(ctx.log_dir, exist_ok=True)
        self._fh = open(os.path.join(ctx.log_dir, self.filename), "a", encoding="utf-8")

    def on_step_end(self, ctx, step, metrics, batch, pred):
        if not self._due(step):
            return
        self._record(ctx, step, "train", metrics)

    def on_validation_end(self, ctx, step, metrics, batch, pred):
        # No cadence check: validation events are already rate-limited by whoever
        # emitted them (ValidationRunner's own every_n_steps). Re-filtering here
        # would silently drop validation points on a mismatched cadence.
        self._record(ctx, step, "val", metrics)

    def _record(self, ctx, step, split, metrics):
        if self.verbose:
            print(f"[step {step}] {split}: "
                  + ", ".join(f"{k}={v:.6f}" for k, v in metrics.items()))
        for term, value in metrics.items():
            self.history.setdefault(f"{split}/{term}", []).append((int(step), float(value)))
            if self._fh is not None:
                self._fh.write(json.dumps({
                    "step": int(step), "split": split,
                    "term": term, "value": float(value),
                }) + "\n")
        if self._fh is not None:
            # Flush per event, not per line: a killed run keeps everything it printed.
            self._fh.flush()

    def on_train_end(self, ctx):
        os.makedirs(ctx.log_dir, exist_ok=True)
        with open(os.path.join(ctx.log_dir, self.summary_filename), "w") as f:
            json.dump(self.history, f, indent=2)
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class MetricsPlotter(Callback):
    """Draws every recorded series on one step axis.

    Reads from a ``MetricsRecorder`` rather than from disk: an explicit object
    dependency beats an implicit shared-file convention, and it keeps the plot
    format free to change without touching how metrics are stored.
    """

    def __init__(self, recorder, every_n_steps=100, filename="metrics.png", verbose=True):
        super().__init__(every_n_steps)
        self.recorder = recorder
        self.filename = filename
        self.verbose = verbose

    def on_validation_end(self, ctx, step, metrics, batch, pred):
        # Refresh on every validation so a long run shows live train-vs-val curves.
        self.plot(ctx)

    def on_step_end(self, ctx, step, metrics, batch, pred):
        if self._due(step):
            self.plot(ctx)

    def on_train_end(self, ctx):
        # Always leave a final plot behind, even if the last step missed the cadence.
        self.plot(ctx)

    def plot(self, ctx):
        series = {name: pts for name, pts in self.recorder.history.items() if pts}
        if not series:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, pts in sorted(series.items()):
            steps, values = zip(*pts)
            ax.plot(steps, values, marker=".", label=name)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        # Losses span orders of magnitude early in training; log scale is only valid
        # when every point is positive (chamfer + laplacian are, but guard anyway).
        if all(v > 0 for pts in series.values() for _, v in pts):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        os.makedirs(ctx.log_dir, exist_ok=True)
        path = os.path.join(ctx.log_dir, self.filename)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        if self.verbose:
            print(f"  saved metrics plot -> {path}")