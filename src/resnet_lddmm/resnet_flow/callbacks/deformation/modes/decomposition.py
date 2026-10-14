"""DecompositionExporter: at the end of training, the run split into network and modes as .vtp files.

The same files as scripts/LDDMM/decompose_modes.py writes for a finished run
(export.py), made from the weights still in memory instead of a checkpoint
read back. Switched on by field.modes.decompose.
"""

from pathlib import Path

from src.resnet_lddmm.learning.callbacks.base import Callback
from src.resnet_lddmm.resnet_flow.callbacks.deformation.modes.export import (
    OUT_DIR, check_frame, write_decomposition,
)


class DecompositionExporter(Callback):
    """On train end, write ``<log_dir>/mode_decomposition/`` from the trained flow.

    Acts once, at the end, so the training steps cost the same. Put it after
    CheckpointWriter and LossLogger: if the export fails, the weights and the
    logs are already on disk.
    """

    def __init__(self, source, target, transform, trajectories: bool = True):
        """``source`` / ``target``: the run's pair as runner.load_pair returns it (landmarks kept);
        ``trajectories``: also every Euler step, not only the end shapes.

        Raises:
            ValueError: the frame has a rotation (check_frame) - before training, not after it.
        """
        super().__init__(every_n_steps=0)
        check_frame(transform)
        self.source, self.target, self.transform = source, target, transform
        self.trajectories = trajectories

    def on_train_end(self, ctx):
        flow = ctx.stepper.flow
        training = flow.training
        flow.eval()
        try:
            print("[modes] the trained flow, split into network and modes:")
            write_decomposition(flow, self.source, self.target, self.transform, Path(ctx.log_dir) / OUT_DIR,
                                trajectories=self.trajectories)
        finally:
            flow.train(training)
