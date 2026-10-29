"""Callbacks that change the model on a step schedule rather than record it."""

from src.resnet_lddmm.resnet_flow.callbacks.schedules.code_prior_em import CodePriorEM
from src.resnet_lddmm.resnet_flow.callbacks.schedules.code_reg import CodeRegWarmup
from src.resnet_lddmm.resnet_flow.callbacks.schedules.fourier import FourierAnnealing

__all__ = ["CodePriorEM", "CodeRegWarmup", "FourierAnnealing"]
