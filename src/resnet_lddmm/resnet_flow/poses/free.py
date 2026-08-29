"""Free rigid poses: the pose IS the parameters, optimised straight from the loss.

The same auto-decoding pattern AutoDecoderCodes uses for latent codes -- a table
of free parameters, one row per shape, with the gradient arriving directly from
the data term -- applied to SE(3) instead of to R^n. No network stands between
the objective and the quantity being solved for, which for a single pair is the
whole of what there is to learn.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.resnet_lddmm.poses.base import PoseSource


def skew(w: Tensor) -> Tensor:
    """[..., 3] axis-angle vectors -> [..., 3, 3] skew-symmetric matrices."""
    zero = torch.zeros_like(w[..., 0])
    return torch.stack([
        torch.stack([zero, -w[..., 2], w[..., 1]], dim=-1),
        torch.stack([w[..., 2], zero, -w[..., 0]], dim=-1),
        torch.stack([-w[..., 1], w[..., 0], zero], dim=-1),
    ], dim=-2)


def bound_norm(v: Tensor, limit: Optional[float]) -> Tensor:
    """Smooth per-row radial squash of ``v`` to ``||v|| <= limit``; None passes through.

    ``v * limit * tanh(||v||/limit) / ||v||``, rather than a clamp, for two reasons
    that both bite in practice:

      * it is the IDENTITY near zero, so imposing a bound does not distort the
        search in the region where every run starts;
      * it has no dead-gradient region. A clamped parameter that walks past its
        boundary receives zero gradient there and can never walk back.

    Radial, not per-component: the constraint on a rotation is on the ANGLE, which
    is ``||omega||``. Clamping components separately would bound a CUBE in
    axis-angle space, making the reachable rotation depend on its axis.
    """
    if limit is None:
        return v
    norm = v.norm(dim=-1, keepdim=True) + 1e-12
    return v * (limit * torch.tanh(norm / limit) / norm)


def _select(parameter: Tensor, shape_ids: Optional[Tensor]) -> Tensor:
    """The rows of a per-shape parameter table that this batch asks for."""
    return parameter if shape_ids is None else parameter[shape_ids.view(-1)]


class FreeSO3(PoseSource):
    """A rotation per shape, held as a free axis-angle vector.

    ``R = exp([omega]_x)``, which earns its keep three ways:

      * exactly orthogonal with det +1 for EVERY parameter value, so the pose
        never needs re-projecting and cannot drift off the manifold;
      * the identity at ``omega = 0``, where the parameters are initialised -- a
        run starts from "no rotation", not from a random one;
      * ``||omega||`` IS the rotation angle, so "at most 30 degrees" is a bound on
        that one vector's norm, applied by bound_norm.

    matrix_exp rather than Rodrigues' formula, which divides by ``||omega||`` and
    is therefore singular at exactly the initial value.
    """

    def __init__(self, num_shapes: int = 1, max_angle_deg: Optional[float] = None):
        """Initialize a table of free rotations.

        Args:
            num_shapes: rows in the table; 1 for pair mode
            max_angle_deg: largest rotation the fit may reach, in degrees, or None
                for unbounded. Worth setting whenever the two shapes are known to
                start roughly aligned: it confines the search to the basin around
                the identity, and a chamfer objective on its own does not reliably
                prefer the correct alignment over a large flip.

        Raises:
            ValueError: on a non-positive num_shapes or bound
        """
        super().__init__()
        if num_shapes < 1:
            raise ValueError(f"num_shapes must be >= 1, got {num_shapes}")
        if max_angle_deg is not None and max_angle_deg <= 0:
            raise ValueError(f"max_angle_deg must be > 0, got {max_angle_deg}")

        self.num_shapes = num_shapes
        self.max_angle = None if max_angle_deg is None else math.radians(max_angle_deg)
        self.omega = nn.Parameter(torch.zeros(num_shapes, 3))

    def rotation(self, shape_ids: Optional[Tensor] = None) -> Tensor:
        """[B, 3, 3] rotations for the requested shapes."""
        return torch.matrix_exp(skew(bound_norm(_select(self.omega, shape_ids),
                                                self.max_angle)))

    def translation(self, shape_ids: Optional[Tensor] = None) -> Optional[Tensor]:
        """None: SO(3) is rotation only, and None is how downstream code knows."""
        return None

    def forward(self, shape_ids: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """(rotation, translation) for the requested shapes."""
        return self.rotation(shape_ids), self.translation(shape_ids)

    def angle_degrees(self) -> list:
        """Each row's current rotation angle, for logging and bound checks."""
        with torch.no_grad():
            return torch.rad2deg(
                bound_norm(self.omega, self.max_angle).norm(dim=-1)).tolist()


class FreeSE3(FreeSO3):
    """FreeSO3 plus a free translation per shape.

    Translations are in NORMALISED units: joint_normalize fits the pair into a
    domain of width 1, so ``max_translation: 0.1`` is a tenth of the pair's
    bounding box -- not a tenth of a millimetre.
    """

    def __init__(self, num_shapes: int = 1, max_angle_deg: Optional[float] = None,
                 max_translation: Optional[float] = None):
        """Initialize a table of free rigid transforms.

        Args:
            num_shapes: rows in the table; 1 for pair mode
            max_angle_deg: see FreeSO3
            max_translation: largest translation in normalised units, or None

        Raises:
            ValueError: on a non-positive bound
        """
        super().__init__(num_shapes=num_shapes, max_angle_deg=max_angle_deg)
        if max_translation is not None and max_translation <= 0:
            raise ValueError(f"max_translation must be > 0, got {max_translation}")
        self.max_translation = max_translation
        self.tau = nn.Parameter(torch.zeros(num_shapes, 3))

    def translation(self, shape_ids: Optional[Tensor] = None) -> Tensor:
        """[B, 3] translations, bounded the same radial way as the rotations."""
        return bound_norm(_select(self.tau, shape_ids), self.max_translation)

    def translation_norms(self) -> list:
        """Each row's current translation length, for logging and bound checks."""
        with torch.no_grad():
            return self.translation().norm(dim=-1).tolist()
