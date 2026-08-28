import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FoldingDecoder(nn.Module):
    def __init__(self, num_samples=256, latent_dim=8, n_freqs=4, verbose=True):
        super().__init__()
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.n_freqs = n_freqs
        self.verbose = verbose
        self.grid_size = int(math.sqrt(num_samples))

        # forward() hard-requires latent.shape[1] == 1 (see below). Exposed so a
        # factory can check encoder/decoder compatibility on the constructed
        # objects (INSTRUCTIONS.md T7 step 4).
        self.expects_tokens = 1

        # Input dimension calculation:
        # grid: 2 channels * 2 * n_freqs = 4 * n_freqs
        # latent: latent_dim
        # total = (4 * n_freqs) + latent_dim
        input_dim_1 = (4 * n_freqs) 
        input_dim_2 = (4 * n_freqs)  

        # Layers
        self.dense1_1 = nn.Linear(input_dim_1, 128)
        self.film1 = nn.Linear(latent_dim, 128*2)
        self.norm1_1 = nn.LayerNorm(128)
        self.dense1_2 = nn.Linear(128, 128)
        self.norm1_2 = nn.LayerNorm(128)
        self.fold1 = nn.Linear(128, 3)

        self.dense2_1 = nn.Linear(input_dim_2, 128)
        self.film2 = nn.Linear(latent_dim, 128*2)
        self.film3 = nn.Linear(3, 128*2)
        self.norm2_1 = nn.LayerNorm(128)
        self.dense2_2 = nn.Linear(128, 128)
        self.norm2_2 = nn.LayerNorm(128)
        self.fold2 = nn.Linear(128, 3)

        # Initialize weights (Variance Scaling)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m is self.fold1 or m is self.fold2:
                    nn.init.normal_(m.weight, std=0.01)

    def positional_encoding(self, coords):
        # coords: [batch, samples, 2]
        freqs = 2.0 ** torch.arange(self.n_freqs, device=coords.device)
        scaled = coords.unsqueeze(-1) * freqs * math.pi
        encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return encoded.view(coords.shape[0], coords.shape[1], -1)

    def forward(self, latent, manual_inv=False):
        # latent: [B, 1, D] -- one D-dimensional latent vector per sample. A plain
        # [B, D] is promoted to [B, 1, D] for backward compatibility.
        if latent.dim() == 2:
            latent = latent.unsqueeze(1)
        if latent.shape[1] != 1:
            raise ValueError(
                f"FoldingDecoder expects a single latent per sample ([B, 1, D]); got "
                f"{latent.shape[1]} tokens. Reduce/pool the latent set to one token first."
            )

        batch_size = latent.shape[0]

        latent_raw = torch.zeros_like(latent) if manual_inv else latent
        latent_tiled = latent_raw.expand(-1, self.num_samples, -1)   # [B, num_samples, D]

        # Create grid
        u = torch.linspace(-1, 1, self.grid_size, device=latent.device)
        v = torch.linspace(-1, 1, self.grid_size, device=latent.device)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        grid = torch.stack([uu.flatten(), vv.flatten()], dim=-1)
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1)
        
        encoded_grid = self.positional_encoding(grid)
        
        # --- First Fold ---
        #x = torch.cat([encoded_grid, latent_tiled], dim=-1)
        #h = self.norm1_1(F.silu(self.dense1_1(x)))
        # Replace cat conditioning with FiLM conditioning
        h = self.norm1_1(F.silu(self.dense1_1(encoded_grid)))   # grid pathway, normed
        g, b = self.film1(latent_tiled).chunk(2, dim=-1)        # latent -> scale/shift
        h = (1 + g) * h + b 
        h = self.norm1_2(F.silu(self.dense1_2(h) + h))
        points_coarse = self.fold1(h)
        
        # --- Second Fold ---
        # x = torch.cat([encoded_grid, points_coarse, latent_tiled, x], dim=-1)
        # Replace cat conditioning with FiLM conditioning
        h = self.norm2_1(F.silu(self.dense2_1(encoded_grid)))
        
        g, b = self.film2(latent_tiled).chunk(2, dim=-1)  # latent -> scale/shift
        h = (1 + g) * h + b 

        g, b = self.film3(points_coarse).chunk(2, dim=-1)  # latent -> scale/shift
        h = (1 + g) * h + b 

        h = self.norm2_2(F.silu(self.dense2_2(h) + h))
        points_final = points_coarse + self.fold2(h)

        return points_final


class SphereFoldingDecoder(nn.Module):
    """Folds a genus-0 SPHERE (S^2) base into 3D -- a topology-matched decoder for CLOSED
    surfaces (sphere, ellipsoid, box, pyramid) where the flat-patch ``FoldingDecoder`` must
    tear a seam to wrap them.

    The base is a Fibonacci-sphere point set: near-uniform, no pole clustering, and valid
    for ANY ``num_samples`` (no perfect-square requirement). Because the base is already a
    closed genus-0 surface, the two folds only have to DEFORM it toward the target -- there
    is no seam to fake, and the network starts (small fold init + residual-on-base) as the
    unit sphere itself, a strong prior for these primitives.

    Interface matches ``FoldingDecoder`` exactly: takes the invariant latent ``[B, D]`` (or
    ``[B, 1, D]``), returns ``[B, num_samples, 3]`` in canonical space. Conditioning is the
    invariant latent only -- rotation/translation are applied downstream, not here.

    NOTE: the sample order is a spiral, NOT a 2D grid, so the grid-based ``laplacian_loss``
    in losses.py is meaningless for this decoder. Train it with
    ``combined_surface_loss(..., laplacian_weight=0.0)`` (pure Chamfer) or a knn/graph
    smoothness term built on ``base_points`` instead.
    """

    def __init__(self, num_samples=256, latent_dim=8, n_freqs=4, verbose=True):
        super().__init__()
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.n_freqs = n_freqs
        self.verbose = verbose

        # forward() hard-requires latent.shape[1] == 1 (see below). Exposed so a
        # factory can check encoder/decoder compatibility on the constructed
        # objects (INSTRUCTIONS.md T7 step 4).
        self.expects_tokens = 1

        # Fixed Fibonacci-sphere base [num_samples, 3]. Non-persistent buffer: deterministic
        # from num_samples (no need to checkpoint) but still moves with ``.to(device)``.
        self.register_buffer("base_points", self._fibonacci_sphere(num_samples),
                             persistent=False)

        # 3D base coords -> Fourier features: 3 channels * 2 (sin, cos) * n_freqs.
        input_dim = 6 * n_freqs

        # --- First fold (deform the sphere) ---
        self.dense1_1 = nn.Linear(input_dim, 128)
        self.film1 = nn.Linear(latent_dim, 128 * 2)
        self.norm1_1 = nn.LayerNorm(128)
        self.dense1_2 = nn.Linear(128, 128)
        self.norm1_2 = nn.LayerNorm(128)
        self.fold1 = nn.Linear(128, 3)

        # --- Second fold (residual refinement) ---
        self.dense2_1 = nn.Linear(input_dim, 128)
        self.film2 = nn.Linear(latent_dim, 128 * 2)
        self.film3 = nn.Linear(3, 128 * 2)
        self.norm2_1 = nn.LayerNorm(128)
        self.dense2_2 = nn.Linear(128, 128)
        self.norm2_2 = nn.LayerNorm(128)
        self.fold2 = nn.Linear(128, 3)

        # Small final-fold init -> starts near the base sphere, then learns the deformation.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m is self.fold1 or m is self.fold2:
                    nn.init.normal_(m.weight, std=0.01)

    @staticmethod
    def _fibonacci_sphere(n):
        """``n`` near-uniform points on the unit sphere via the golden-angle spiral."""
        i = torch.arange(n, dtype=torch.float32)
        z = 1.0 - 2.0 * (i + 0.5) / n                         # (+1..-1), poles excluded
        radius = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        golden = math.pi * (3.0 - math.sqrt(5.0))             # golden angle
        theta = golden * i
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        return torch.stack([x, y, z], dim=-1)                 # [n, 3]

    def positional_encoding(self, coords):
        # coords: [B, N, 3] on the unit sphere -> [B, N, 6*n_freqs]
        freqs = 2.0 ** torch.arange(self.n_freqs, device=coords.device)
        scaled = coords.unsqueeze(-1) * freqs * math.pi
        encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return encoded.view(coords.shape[0], coords.shape[1], -1)

    def forward(self, latent, manual_inv=False):
        # latent: [B, 1, D] -- one D-dimensional latent per sample; [B, D] is promoted.
        if latent.dim() == 2:
            latent = latent.unsqueeze(1)
        if latent.shape[1] != 1:
            raise ValueError(
                f"SphereFoldingDecoder expects a single latent per sample ([B, 1, D]); got "
                f"{latent.shape[1]} tokens. Reduce/pool the latent set to one token first."
            )

        batch_size = latent.shape[0]

        latent_raw = torch.zeros_like(latent) if manual_inv else latent
        latent_tiled = latent_raw.expand(-1, self.num_samples, -1)      # [B, num_samples, D]

        base = self.base_points.to(latent.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        encoded_base = self.positional_encoding(base)

        # --- First Fold: residual deformation on the base sphere ---
        h = self.norm1_1(F.silu(self.dense1_1(encoded_base)))
        g, b = self.film1(latent_tiled).chunk(2, dim=-1)               # latent -> scale/shift
        h = (1 + g) * h + b
        h = self.norm1_2(F.silu(self.dense1_2(h) + h))
        points_coarse = base + self.fold1(h)

        # --- Second Fold: residual refinement, conditioned on latent + coarse geometry ---
        h = self.norm2_1(F.silu(self.dense2_1(encoded_base)))
        g, b = self.film2(latent_tiled).chunk(2, dim=-1)
        h = (1 + g) * h + b
        g, b = self.film3(points_coarse).chunk(2, dim=-1)
        h = (1 + g) * h + b
        h = self.norm2_2(F.silu(self.dense2_2(h) + h))
        points_final = points_coarse + self.fold2(h)

        return points_final