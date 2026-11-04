"""Shape code abstractions for amortised vs per-shape inference."""

import abc
from typing import Optional, Tuple
import torch.nn as nn
from torch import Tensor


class ShapeCode(nn.Module, abc.ABC):
    """Provides per-shape conditioning codes, and optionally a rigid pose.

    Maps a batch object to per-shape latent codes (None if no codes).
    The code_source decides HOW shapes get codes (auto-decoder table lookup,
    encoder forward pass, or none); the stepper never knows which.

    A code source may also carry a POSE, supplied by an injected PoseSource. Pose
    and code are orthogonal -- the code conditions the velocity field, the pose
    transforms geometry -- so this is composition, not a second responsibility:
    any code kind can be given any pose kind. ``get_pose`` below is the whole of
    the integration, and it is what PairRegistration and the pose callbacks
    duck-type on. EncoderCodes overrides it, because its pose is PREDICTED from
    the sample rather than stored.
    """

    def __init__(self, pose=None):
        """Initialize, optionally with a pose source.

        Args:
            pose: PoseSource instance, or None for a code source with no pose.
                Assigning it here registers it as a child module, so its
                parameters appear in ``self.parameters()`` and reach the optimizer.
        """
        super().__init__()
        self.pose = pose

    @abc.abstractmethod
    def forward(self, batch) -> Optional[Tensor]:
        """Extract codes from batch.

        Args:
            batch: batch-like object with shape information (e.g., shape_ids)

        Returns:
            [B, n_z] tensor of per-shape codes, or None if no codes
        """

    @abc.abstractmethod
    def penalty(self) -> Optional[Tensor]:
        """Compute code regularisation term (e.g., L2 on embeddings).

        Returns:
            Scalar loss tensor, or None if no regularisation applies
        """

    def get_pose(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """(rotation, translation) in the ``points @ R + t`` convention, or (None, None).

        ``(None, None)`` when this code source carries no pose -- the mapping
        errors treat that as "no pose" and leave the geometry alone, so a pose-less
        code source needs no special case anywhere.

        Returns:
            (rotation [B,3,3] or None, translation [B,3] or None)
        """
        return (None, None) if self.pose is None else self.pose()
