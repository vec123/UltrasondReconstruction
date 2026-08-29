"""SO3Augmentation: random rotation-only augmentation."""

from typing import Optional
import torch
from torch import Tensor
from src.resnet_lddmm.augmentation.base import Augmentation
from src.transforms.group_transforms import SE3_transform


class SO3Augmentation(Augmentation):
    """Apply random SO(3) rotation to each shape in batch.

    Samples a random rotation matrix for each shape independently, then applies
    via SE3_transform with zero translation.

    Uses quaternion-based uniform sampling on SO(3) for mathematically clean
    distribution (avoids gimbal lock and ensures uniform coverage).
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize SO3 augmentation.

        Args:
            seed: Random seed for reproducibility. If None, non-deterministic.
        """
        super().__init__()
        self.seed = seed
        self.rng: Optional[torch.Generator] = None
        if seed is not None:
            self.rng = torch.Generator()
            self.rng.manual_seed(seed)

    def _sample_random_rotations(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
        """Sample batch_size random rotation matrices from SO(3) via quaternions.

        Method:
        1. Sample quaternions q ∈ ℝ⁴ from standard normal
        2. Normalize: q̂ = q / ‖q‖ (uniform on S³)
        3. Convert to rotation matrix (3×3)

        Returns:
            [batch_size, 3, 3] rotation matrices
        """
        # Sample unit quaternions (batch_size, 4)
        q = torch.randn(batch_size, 4, device=device, dtype=dtype, generator=self.rng)
        q = q / q.norm(dim=1, keepdim=True)

        # Extract components
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # Convert quaternion to rotation matrix (3x3 for each quaternion)
        # Using standard conversion formulas
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

    def forward(self, points: Tensor) -> Tensor:
        """Apply random SO(3) rotation to each shape in batch.

        Args:
            points: [B, N, 3] point cloud batch

        Returns:
            [B, N, 3] rotated point cloud
        """
        batch_size, num_points, _ = points.shape
        device = points.device
        dtype = points.dtype

        # Sample random rotations
        rotations = self._sample_random_rotations(batch_size, device, dtype)  # [B, 3, 3]

        # Zero translations (rotation only)
        translations = torch.zeros(batch_size, 3, device=device, dtype=dtype)

        # Record the drawn element so a supervision term can target it
        self._record_element(rotations, None)

        # Flatten points: [B*N, 3]
        flat_points = points.reshape(-1, 3)

        # Create n_node array: [B] where each element = N
        n_node = torch.full((batch_size,), num_points, device=device, dtype=torch.long)

        # Apply rotation via SE3_transform
        rotated_flat = SE3_transform(flat_points, n_node, rotations, translations)

        # Reshape back to [B, N, 3]
        return rotated_flat.reshape(batch_size, num_points, 3)
