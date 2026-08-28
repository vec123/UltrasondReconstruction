"""Geometry-export callback.

Unlike the other four callbacks, this one DELEGATES rather than reimplements: VTP
rendering is ~150 lines of geometry/field-marshalling code in ``TrainingLogger``
(``_save_graph_vtp``, ``_save_true_verts``, ``visualize_results``), and rewriting
it would be a large, risky change that has nothing to do with the Observer pattern
T12 is about. So ``TrainingLogger`` survives here as the geometry RENDERER; what
this callback owns is the policy -- when to render, into which subdirectory, and
how many shapes.

Its metrics/checkpoint halves are superseded by MetricsRecorder and
CheckpointWriter and are no longer driven by the orchestrator.
"""

from src.learning.callbacks.base import Callback
from src.learning.logger.train_logs import TrainingLogger


class GeometryVisualizer(Callback):
    """Writes input graph / target / prediction VTPs at its own cadence.

    Validation renders go to ``vtk/validation`` so they never overwrite the
    training VTPs written at the same step number.
    """

    def __init__(self, every_n_steps=100, max_num=4, renderer=None, val_viz_random=False):
        super().__init__(every_n_steps)
        self.max_num = max_num
        self._renderer = renderer
        self.val_viz_random = val_viz_random

    def _get_renderer(self, ctx):
        # Built lazily from ctx so log_dir has exactly one source of truth.
        if self._renderer is None:
            self._renderer = TrainingLogger(log_dir=ctx.log_dir)
        return self._renderer

    def on_step_end(self, ctx, step, metrics, batch, pred):
        if not self._due(step) or batch is None or pred is None:
            return
        self._get_renderer(ctx).visualize_batch(batch, pred, step, max_num=self.max_num)

    def on_validation_end(self, ctx, step, metrics, batch, pred):
        if batch is None or pred is None:
            return
        self._get_renderer(ctx).visualize_val_batch(batch, pred, step, max_num=self.max_num,
                                                     random_sample=self.val_viz_random)