"""Entry points: build, restore, then train."""

import torch
from src.resnet_lddmm.resnet_flow.config import ExperimentCfg
from src.resnet_lddmm.learning.trainers.E3_end2end import TrainingOrchestrator

from .build import build
from .callbacks import complete_callbacks, default_callbacks
from .restore import restore_from


def run(cfg: ExperimentCfg, callbacks=None):
    """Load config, build stepper, run orchestrator.

    Args:
        cfg: ExperimentCfg instance
        callbacks: optional base list of callbacks (default: ``default_callbacks``);
            ``complete_callbacks`` adds the ones the model and config need

    Returns:
        TrainingContext from orchestrator.run
    """
    # Every tensor the build and the training loop create lands on train.device
    # (torch's default-device context); data read from numpy is moved there explicitly.
    with torch.device(cfg.train.device):
        return _run(cfg, callbacks)


def _run(cfg: ExperimentCfg, callbacks=None):
    if callbacks is None:
        callbacks = default_callbacks(cfg)

    stepper, loader, transform = build(cfg)

    # Weights, if the config names a checkpoint to start from. AFTER build (the
    # modules and optimizer have to exist to load into) and BEFORE the callback
    # list is finalised, so a bad init.checkpoint fails before any output is
    # written rather than a few steps in.
    start_step = restore_from(cfg, stepper)
    callbacks = complete_callbacks(cfg, callbacks, stepper, transform)

    orchestrator = TrainingOrchestrator(
        stepper=stepper,
        dataloader=loader,
        callbacks=callbacks,
        log_dir=cfg.output_dir,
    )
    # train.steps is how many steps to RUN, so a resumed run takes that many more
    # rather than stopping the moment the counter passes it.
    return orchestrator.run(num_steps=cfg.train.steps, start_step=start_step)
