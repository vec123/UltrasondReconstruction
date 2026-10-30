"""The pipeline's one pose convention: ``points @ R + t``."""

from typing import Optional
import torch
from torch import Tensor


def apply_pose(points: Tensor, rotation: Optional[Tensor],
               translation: Optional[Tensor]) -> Tensor:
    """Apply an SE(3) pose to a batch of points: ``points @ R + t``.

    The ONE definition of the pipeline's pose convention -- ``points @ R``, not
    ``R @ points``. Everything that moves geometry by a pose goes through here
    (both mapping errors, and apply's client), because a handedness that differs
    between training and inference is invisible in the shapes' extents and only
    shows up as a mirrored result. See EncoderCodes.get_pose for why this
    convention, and not its transpose, is the one the encoder is built against.

    einsum rather than SE3_transform: it preserves gradients through the
    rotation, which is the only path the pose head has to train on.

    Args:
        points: [B, N, 3] batched points
        rotation: [B, 3, 3] or None
        translation: [B, 3] or None

    Returns:
        [B, N, 3] transformed points (the input tensor itself when both are None)
    """
    if rotation is not None:
        # 'bij,bjk->bik': for each batch b, [N,3] @ [3,3] = [N,3]
        points = torch.einsum('bij,bjk->bik', points, rotation)
    if translation is not None:
        points = points + translation.unsqueeze(1)  # [B,1,3] + [B,N,3]
    return points
