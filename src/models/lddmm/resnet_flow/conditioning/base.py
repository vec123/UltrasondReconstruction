"""Conditioning ABC: how a shape code enters a velocity field."""

import abc
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor


class Conditioning(nn.Module, abc.ABC):
    """Turns a GLOBAL per-shape code into PER-POINT code features.

    The field decides WHERE those features enter its stack; the conditioning
    decides HOW a global vector becomes a per-point one.
    """

    dim: int  # per-point feature dimension; 0 for NoConditioning

    @abc.abstractmethod
    def forward(self, x: Tensor, code: Optional[Tensor]) -> Optional[Tensor]:
        """Map positions and code to per-point conditioning features.

        Args:
            x: [B, N, 3] point positions
            code: [B, N_z] per-shape code (ignored for unconditioned fields)

        Returns:
            [B, N, dim] per-point features, or None iff dim == 0
        """


class NoConditioning(Conditioning):
    """Null Object for unconditioned fields (no code injection)."""

    def __init__(self):
        super().__init__()
        self.dim = 0

    def forward(self, x: Tensor, code: Optional[Tensor]) -> None:
        return None
