"""Augmentation strategies for random group transformations on source shapes."""

from src.models.lddmm.resnet_flow.augmentation.base import Augmentation
from src.models.lddmm.resnet_flow.augmentation.none import NoAugmentation
from src.models.lddmm.resnet_flow.augmentation.so3 import SO3Augmentation
from src.models.lddmm.resnet_flow.augmentation.se3 import SE3Augmentation

__all__ = [
    "Augmentation",
    "NoAugmentation",
    "SO3Augmentation",
    "SE3Augmentation",
]
