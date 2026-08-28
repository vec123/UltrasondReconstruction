"""PoseSource: where a rigid pose comes from.

Split out from ShapeCode because pose and code are orthogonal. A code conditions
the velocity FIELD; a pose transforms GEOMETRY, through mapping_error.apply_pose.
They live on different manifolds (R^n vs SE(3)), carry different constraints
(``||z||^2`` vs a bound on the rotation angle), and are consumed at different
points in the forward pass -- so a code source HOLDS a pose rather than being one.

That composition is what keeps the config from going combinatorial: any code kind
can carry any pose kind, instead of needing an ``auto_decoder_with_pose`` the
first time someone wants both.
"""

import abc
from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor


class PoseSource(nn.Module, abc.ABC):
    """Supplies a rigid pose in the pipeline's ``points @ R + t`` convention.

    Implementations decide WHERE the pose comes from -- free parameters optimised
    directly, or (in EncoderCodes' case) a network's prediction. The mapping error
    never knows which.

    ``num_shapes`` mirrors AutoDecoderCodes' embedding table: one pose per shape,
    so a cohort member gets its own. Pair mode is the ``num_shapes == 1`` case.
    """

    @abc.abstractmethod
    def forward(self, shape_ids: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """Produce the pose.

        Args:
            shape_ids: [B] indices of the shapes to pose, or None for all of them
                (which is every shape in pair mode, where there is exactly one)

        Returns:
            (rotation [B, 3, 3], translation [B, 3] or None). Rotation is always
            present; translation is None for a rotation-only pose, which is how
            SO(3) is distinguished from SE(3) downstream.
        """
