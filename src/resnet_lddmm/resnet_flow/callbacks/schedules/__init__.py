"""Callbacks that change the model on a step schedule rather than record it."""

from src.resnet_lddmm.resnet_flow.callbacks.schedules.fourier import FourierAnnealing

__all__ = ["FourierAnnealing"]
