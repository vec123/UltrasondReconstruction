"""Checkpoints: written during training, found and read again by apply and by resumed runs.

    CheckpointWriter    the callback that writes them - generic, shared with the other
                        trainer, so it lives in ``src.resnet_lddmm.learning.callbacks``
                        and is re-exported here as part of this concern
    files.py            the reader for its layout: ``step_<N>.pt`` / ``final.pt`` in
                        ``<output_dir>/checkpoints`` - resolve, load, step numbers
"""

from src.resnet_lddmm.learning.callbacks.checkpointing import CheckpointWriter
from src.resnet_lddmm.resnet_flow.callbacks.checkpoints.files import (
    CheckpointRef, checkpoint_step, load_checkpoint, resolve_checkpoint,
)

__all__ = ["CheckpointWriter", "CheckpointRef", "checkpoint_step", "load_checkpoint", "resolve_checkpoint"]
