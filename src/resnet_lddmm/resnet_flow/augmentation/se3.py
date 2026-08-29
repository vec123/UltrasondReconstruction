"""SE3Augmentation: random rotation + translation augmentation."""

from typing import Optional
import torch
from torch import Tensor
from src.resnet_lddmm.augmentation.base import Augmentation
from src.transforms.group_transforms import SE3_transform


class SE3Augmentation(Augmentation):
    """Apply random SE(3) transformation (rotation + translation) to each shape in batch.

    Samples a random rotation and translation for each shape independently.
    Rotations use quaternion-based uniform sampling; translations use uniform
    bounds specified by translation_scale.
    """

    def __init__(self, translation_scale: float = 0.2, seed: Optional[int] = None):
        """Initialize SE(3) augmentation.

        Args:
            translation_scale: Bounds for uniform translation sampling.
                Translations sampled uniformly from [-translation_scale, +translation_scale]³.
                Default 0.2 means each component in [-0.2, 0.2].
            seed: Random seed for reproducibility. If None, non-deterministic.
        """
        super().__init__()
        self.translation_scale = translation_scale
        self.seed = seed
        self.rng: Optional[torch.Generator] = None
        if seed is not None:
            self.rng = torch.Generator()
            self.rng.manual_seed(seed)

    def _sample_random_rotations(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
        """Sample batch_size random rotation matrices from SO(3) via quaternions.

        Returns:
            [batch_size, 3, 3] rotation matrices
        """
        # Sample unit quaternions (batch_size, 4)
        q = torch.randn(batch_size, 4, device=device, dtype=dtype, generator=self.rng)
        q = q / q.norm(dim=1, keepdim=True)

        # Extract components
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # Convert quaternion to rotation matrix (3x3 for each quaternion)
        rotations = torch.stack([
            torch.stack([
                1 - 2*(y**2 + z**2),
                2*(x*y - w*z),
                2*(x*z + w*y)
            ], dim=1),
            torch.stack([
                2*(x*y + w*z),
                1 - 2*(x**2 + z**2),
                2*(y*z - w*x)
            ], dim=1),
            torch.stack([
                2*(x*z - w*y),
                2*(y*z + w*x),
                1 - 2*(x**2 + y**2)
            ], dim=1)
        ], dim=1)  # [batch_size, 3, 3]

        return rotations

    def _sample_random_translations(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
        """Sample batch_size random translations uniformly in [-translation_scale, +translation_scale]³.

        Returns:
            [batch_size, 3] translation vectors
        """
        # Uniform distribution in [-translation_scale, translation_scale]
        t = torch.empty(batch_size, 3, device=device, dtype=dtype).uniform_(
            -self.translation_scale, self.translation_scale, generator=self.rng
        )
        return t

    def forward(self, points: Tensor) -> Tensor:
        """Apply random SE(3) transformation to each shape in batch.

        Args:
            points: [B, N, 3] point cloud batch

        Returns:
            [B, N, 3] transformed point cloud
        """
        batch_size, num_points, _ = points.shape
        device = points.device
        dtype = points.dtype

        # Sample random rotations and translations
        rotations = self._sample_random_rotations(batch_size, device, dtype)  # [B, 3, 3]
        translations = self._sample_random_translations(batch_size, device, dtype)  # [B, 3]

        # Record the drawn element so a supervision term can target it
        self._record_element(rotations, translations)

        # Flatten points: [B*N, 3]
        flat_points = points.reshape(-1, 3)

        # Create n_node array: [B] where each element = N
        n_node = torch.full((batch_size,), num_points, device=device, dtype=torch.long)

        # Apply SE(3) transformation
        transformed_flat = SE3_transform(flat_points, n_node, rotations, translations)

        # Reshape back to [B, N, 3]
        return transformed_flat.reshape(batch_size, num_points, 3)
