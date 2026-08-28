"""EquivariantStationaryField: SO(3)-equivariant velocity field using e3nn tensor products."""

from typing import Optional, List
import torch
import torch.nn as nn
from torch import Tensor

from e3nn import o3

from src.resnet_lddmm.fields.base import VelocityField
from src.resnet_lddmm.conditioning.base import Conditioning, NoConditioning
from src.learning.modules.equivariant.interaction import SelfInteraction


class EquivariantStationaryField(VelocityField):
    """SO(3)-equivariant stationary velocity field using stacked e3nn tensor products.

    Architecture:
      Input: x [B, N, 3] (1o irreps)
      → Layer 1: SelfInteraction (x ⊗ x) → layer_irreps[0]
      → Layer 2: SelfInteraction (h ⊗ h) → layer_irreps[1]
      → ... (more layers if specified)
      → Output layer: final_irreps → 1o (velocity)

    Equivariance guaranteed by construction: v'(Rx) = R @ v(x) for any SO(3) rotation R.
    """

    def __init__(
        self,
        in_irreps: str = "1x1o",
        hidden_irreps: str = "4x0e + 2x1o",
        out_irreps: str = "1x1o",
        layers_cfg: Optional[List[dict]] = None,
        conditioning: Optional[Conditioning] = None,
        use_tensor_product_self: bool = True,
        gate_hidden_dim: int = 64,
        verbose: bool = False,
    ):
        """Initialize equivariant stationary field.

        Args:
            in_irreps: input irreps (should be "1x1o" for positions)
            hidden_irreps: intermediate feature representation (e.g., "4x0e + 2x1o")
                          (ignored if layers_cfg provided)
            out_irreps: output irreps (should be "1x1o" for velocities)
            layers_cfg: list of dicts specifying layer stack, e.g.:
                       [{"target_irreps": "32x0e + 16x1o"}, {"target_irreps": "64x0e + 32x1o"}]
                       If provided, builds multiple SelfInteraction layers instead of single hidden_irreps
            conditioning: optional conditioning module (preserves equivariance if outputs scalars)
            use_tensor_product_self: whether to use SelfInteraction (V ⊗ V + gating) vs simple Linear
            gate_hidden_dim: hidden dim for SelfInteraction gating network
            verbose: print shape/module info during initialization
        """
        super().__init__()
        self.in_irreps_str = in_irreps
        self.hidden_irreps_str = hidden_irreps
        self.out_irreps_str = out_irreps
        self.layers_cfg = layers_cfg or []
        self.conditioning = conditioning or NoConditioning()
        self.use_tensor_product_self = use_tensor_product_self
        self.verbose = verbose

        # Parse input/output irreps
        self.in_irreps = o3.Irreps(in_irreps)
        self.out_irreps = o3.Irreps(out_irreps)

        if verbose:
            print(f"EquivariantStationaryField:")
            print(f"  Input: {self.in_irreps}")
            print(f"  Conditioning dim: {self.conditioning.dim}")

        # Build layer stack
        if self.layers_cfg:
            # Multi-layer: thread output of each layer to input of next
            self.layers = nn.ModuleList()
            current_irreps_str = self.in_irreps_str

            for i, layer_cfg in enumerate(self.layers_cfg):
                # Extract layer config parameters
                layer_in_irreps = layer_cfg.get("in_irreps", current_irreps_str)
                target_irreps = layer_cfg["target_irreps"]
                sh_lmax = layer_cfg.get("sh_lmax", 1)

                # Validate chain: each layer's expected input matches previous output
                if i > 0 and layer_in_irreps != current_irreps_str:
                    raise ValueError(
                        f"Layer {i} input {layer_in_irreps} doesn't match "
                        f"previous output {current_irreps_str}"
                    )

                if verbose:
                    print(f"  Layer {i}: {layer_in_irreps} → {target_irreps} (sh_lmax={sh_lmax})")

                if use_tensor_product_self:
                    layer = SelfInteraction(
                        in_irreps=layer_in_irreps,
                        target_irreps=target_irreps,
                        sh_lmax=sh_lmax,
                        gate_hidden_dim=gate_hidden_dim,
                        verbose=False,  # Avoid spam from nested layers
                    )
                else:
                    # Simple linear projection
                    in_ir = o3.Irreps(layer_in_irreps)
                    out_ir = o3.Irreps(target_irreps)
                    layer = o3.Linear(in_ir, out_ir)

                self.layers.append(layer)
                current_irreps_str = target_irreps

            # Final linear layer: last layer output → 1o
            final_irreps = o3.Irreps(current_irreps_str)
            if verbose:
                print(f"  Output: {current_irreps_str} → {self.out_irreps_str}")

        else:
            # Single layer: input → hidden_irreps
            self.hidden_irreps = o3.Irreps(hidden_irreps)
            current_irreps_str = hidden_irreps

            if verbose:
                print(f"  Hidden: {self.hidden_irreps}")
                print(f"  Output: {self.hidden_irreps_str} → {self.out_irreps_str}")

            if use_tensor_product_self:
                self.layers = nn.ModuleList([
                    SelfInteraction(
                        in_irreps=self.in_irreps_str,
                        target_irreps=self.hidden_irreps_str,
                        sh_lmax=1,
                        gate_hidden_dim=gate_hidden_dim,
                        verbose=verbose,
                    )
                ])
            else:
                self.layers = nn.ModuleList([
                    o3.Linear(self.in_irreps, self.hidden_irreps)
                ])

            final_irreps = self.hidden_irreps

        # Final linear layer: last layer output → 1o
        self.linear_out = o3.Linear(final_irreps, self.out_irreps)
        # Scale down initialization for stable training (identity flow at start)
        with torch.no_grad():
            self.linear_out.weight.mul_(0.01)

    def forward(
        self, x: Tensor, step: Optional[int] = None, code: Optional[Tensor] = None
    ) -> Tensor:
        """Compute SO(3)-equivariant velocity.

        Args:
            x: [B, N, 3] point positions (1o irreps)
            step: ignored (stationary field uses same velocity at all times)
            code: [B, N_z] per-shape code (passed to conditioning)

        Returns:
            [B, N, 3] velocity field (1o irreps)
        """
        B, N, _ = x.shape

        # Conditioning: code → per-point scalar features (0e irreps)
        zbar = self.conditioning(x, code)  # [B, N, c_dim] or None

        # Flatten for processing: [B, N, 3] → [B*N, 3]
        x_flat = x.reshape(B * N, 3)  # [B*N, 3]

        if zbar is not None:
            zbar_flat = zbar.reshape(B * N, -1)  # [B*N, c_dim]
            # Conditioning adds scalars: (1o) ⊕ (c_dim × 0e)
            h = torch.cat([x_flat, zbar_flat], dim=-1)  # [B*N, 3 + c_dim]
        else:
            h = x_flat  # [B*N, 3]

        # Thread through layer stack
        for layer in self.layers:
            h = layer(h)  # [B*N, layer_dim]

        # Final linear projection to output (1o)
        v_flat = self.linear_out(h)  # [B*N, 3]

        # Reshape back to [B, N, 3]
        return v_flat.reshape(B, N, 3)


class EquivariantStationaryFieldSimple(VelocityField):
    """Simplified version: x → MLP → gating → v (minimal e3nn, for comparison)."""

    def __init__(
        self,
        width: int = 64,
        hidden_irreps: str = "4x0e + 2x1o",
        conditioning: Optional[Conditioning] = None,
        verbose: bool = False,
    ):
        """Minimal equivariant field for baseline comparison."""
        super().__init__()
        self.conditioning = conditioning or NoConditioning()
        self.verbose = verbose

        # Input MLP: 3 → width → width
        self.lift = nn.Linear(3 + self.conditioning.dim, width)

        # Intermediate features (simplified, no full tensor product)
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.hidden_proj = nn.Linear(width, self.hidden_irreps.dim)

        # Extract scalar/vector parts
        self.hidden_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in self.hidden_irreps if ir.l == 0]
        )
        self.hidden_vectors = o3.Irreps(
            [(mul, ir) for mul, ir in self.hidden_irreps if ir.l > 0]
        )

        # Gating
        if self.hidden_vectors.dim > 0:
            self.gate_net = nn.Sequential(
                nn.Linear(self.hidden_scalars.dim, 32),
                nn.ReLU(),
                nn.Linear(32, self.hidden_vectors.dim),
                nn.Tanh(),
            )
        else:
            self.gate_net = None

        # Output: [B*N, 3]
        self.output = nn.Linear(self.hidden_irreps.dim, 3, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(
        self, x: Tensor, step: Optional[int] = None, code: Optional[Tensor] = None
    ) -> Tensor:
        B, N, _ = x.shape
        zbar = self.conditioning(x, code)

        x_flat = x.reshape(B * N, 3)
        if zbar is not None:
            zbar_flat = zbar.reshape(B * N, -1)
            h = torch.cat([x_flat, zbar_flat], dim=-1)
        else:
            h = x_flat

        # MLP projection
        h = torch.relu(self.lift(h))
        h_hidden = self.hidden_proj(h)

        # Gating
        if self.gate_net is not None:
            scalars = h_hidden[:, : self.hidden_scalars.dim]
            vectors = h_hidden[:, self.hidden_scalars.dim :]
            gates = self.gate_net(scalars)
            h_hidden = torch.cat([scalars, vectors * gates], dim=-1)

        # Output
        v_flat = self.output(h_hidden)
        return v_flat.reshape(B, N, 3)
