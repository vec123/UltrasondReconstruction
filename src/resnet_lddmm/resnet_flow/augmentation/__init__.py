"""Augmentation strategies for random group transformations on source shapes."""

from src.resnet_lddmm.augmentation.base import Augmentation
from src.resnet_lddmm.augmentation.none import NoAugmentation
from src.resnet_lddmm.augmentation.so3 import SO3Augmentation
from src.resnet_lddmm.augmentation.se3 import SE3Augmentation

__all__ = [
    "Augmentation",
    "NoAugmentation",
    "SO3Augmentation",
    "SE3Augmentation",
]
