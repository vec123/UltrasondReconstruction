"""The loss, as training sees it: per-term values, weights and contributions of each logged step.

    logger.py    LossLogger: the breakdown of each logged step as JSON (loss_logs/)
    console.py   VerboseCallback: the loss printed to the console at its cadence
"""

from src.resnet_lddmm.resnet_flow.callbacks.loss.console import VerboseCallback
from src.resnet_lddmm.resnet_flow.callbacks.loss.logger import LossLogger

__all__ = ["LossLogger", "VerboseCallback"]
