"""Helpers shared by the combinators: parameter namespacing and the child check."""

from __future__ import annotations

from typing import Dict, List, Sequence
from src.SSM.kernels.base import Kernel


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _namespaced(children: Sequence[Kernel]) -> Dict[str, float]:
    """Child parameters, prefixed by position: ``0.log_sigma``, ``1.log_scale``."""
    out: Dict[str, float] = {}
    for index, child in enumerate(children):
        for key, value in child.free_parameters().items():
            out[f"{index}.{key}"] = value
    return out


def _split_namespaced(theta: Dict[str, float], count: int) -> List[Dict[str, float]]:
    """Inverse of :func:`_namespaced`; unprefixed keys are ignored."""
    parts: List[Dict[str, float]] = [{} for _ in range(count)]
    for key, value in theta.items():
        head, _, rest = key.partition(".")
        if not rest or not head.isdigit():
            continue
        index = int(head)
        if 0 <= index < count:
            parts[index][rest] = value
    return parts


def _check_children(children, who: str) -> List[Kernel]:
    children = list(children or [])
    if len(children) < 1:
        raise ValueError(f"{who} needs at least one child kernel")
    for child in children:
        if not isinstance(child, Kernel):
            raise TypeError(f"{who} children must be Kernels, got {type(child).__name__}")
    return children
