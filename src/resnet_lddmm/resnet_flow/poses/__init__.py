"""Rigid pose sources, held by a code source rather than being one."""

from src.resnet_lddmm.resnet_flow.poses.base import PoseSource
from src.resnet_lddmm.resnet_flow.poses.free import FreeSE3, FreeSO3

# Pose kinds that require a translation bound to be meaningful. Used to reject
# `max_translation` under a rotation-only kind, where it would be ignored in
# silence.
TRANSLATING_KINDS = frozenset({"se3"})

__all__ = [
    "PoseSource",
    "FreeSE3",
    "FreeSO3",
    "TRANSLATING_KINDS",
]
