"""EquiformerV3 attention as a drop-in equivariant refinement block (e3nn + PyG).

Wraps the vendored EquiformerV2/V3 attention subsystem (``.equiformer_v3``, from Meta's
fairchem, no fairchem dependency) so it can be applied to this project's flat
``[N, irreps.dim]`` e3nn features, like ``SE3Transformer``. It is *in-place*
(``irreps_in == irreps_out``): a residual around the block keeps the output layout.

EquiformerV3 works on dense SO(3) embeddings ``[N, (lmax+1)^2, C]`` (uniform channel
count ``C`` across all degrees) and uses the eSCN / SO(2) trick — rotate each edge's
features into an edge-aligned frame (per-edge Wigner-D), do cheap SO(2) linear ops, and
rotate back. Its payoff is efficient HIGH-degree (lmax >= 4) attention; at the low lmax
of typical supernode features it is heavier than ``SE3Transformer`` but fully valid, and
lets you push the internal ``lmax`` above the model's irreps for richer geometry.

Adapter responsibilities per forward:
  * build a within-shape fully-connected graph (shared with ``SE3Transformer``);
  * project the flat irreps -> a uniform SO(3) embedding (equivariant ``o3.Linear`` +
    a reshape/transpose that is exact because both use e3nn's real-SH basis);
  * compute per-edge rotation matrices + Wigner-D (``init_edge_rot_mat`` / ``set_wigner``)
    and a Gaussian radial expansion, then run the transformer blocks;
  * project the SO(3) embedding back to the flat irreps and add the residual.

Equivariance is exact (verified): every step is equivariant, and the eSCN construction
is invariant to the (random) choice of in-plane edge frame.
"""

import torch
import torch.nn as nn
from e3nn import o3

from src.learning.modules.equivariant.transformer import within_shape_edges
from src.learning.modules.equivariant.equiformer_v3.transformer_block import TransBlockV3
from src.learning.modules.equivariant.equiformer_v3.so3 import SO3Rotation
from src.learning.modules.equivariant.equiformer_v3.edge_rot_mat import init_edge_rot_mat
from src.learning.modules.equivariant.equiformer_v3.radial_function import GaussianSmearing
from src.learning.modules.equivariant.equiformer_v3.layer_norm import get_normalization_layer


def _uniform_irreps(lmax, channels):
    """``channels`` multiplicities of every degree 0..lmax with spherical-harmonic parity."""
    return o3.Irreps([(channels, (l, (-1) ** l)) for l in range(lmax + 1)])


class EquiformerTransformer(nn.Module):
    """Stack of EquiformerV3 transformer blocks over a point set, in-place on the irreps.

    Attends over a within-shape fully-connected graph (from ``batch``) by default; pass an
    explicit ``edge_index`` to override. ``lmax`` defaults to the max degree present in
    ``irreps``; raising it gives the blocks higher-degree internal features (built up from
    the edge geometry) that are projected back down to ``irreps`` on output.
    """

    def __init__(self, irreps, num_layers=2, num_channels=16, num_heads=4,
                 attn_alpha_channels=8, attn_value_channels=8, attn_hidden_channels=None,
                 ffn_hidden_channels=None, lmax=None, mmax=None, num_radial_basis=16,
                 cutoff=2.0, norm_type='sep_layer_norm', attn_activation='gate',
                 ffn_activation='gate', use_grid_mlp=False, final_norm=True, verbose=False):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.C = num_channels
        self.verbose = verbose

        input_degrees = sorted({ir.l for _, ir in self.irreps})
        max_in_degree = max(input_degrees)
        self.lmax = max_in_degree if lmax is None else lmax
        self.mmax = self.lmax if mmax is None else mmax
        assert self.lmax >= max_in_degree, \
            f"lmax ({self.lmax}) must be >= max input degree ({max_in_degree})"
        attn_hidden_channels = attn_hidden_channels or num_channels
        ffn_hidden_channels = ffn_hidden_channels or num_channels

        # Uniform SO(3) fibers: full set of degrees 0..lmax for the embedding / out proj,
        # and the subset also present in the input for the in proj (others start at zero).
        self.uniform_irreps = _uniform_irreps(self.lmax, self.C)
        self.in_irreps = o3.Irreps([(self.C, (l, (-1) ** l)) for l in range(self.lmax + 1)
                                    if l in input_degrees])
        self.in_proj = o3.Linear(self.irreps, self.in_irreps)
        self.out_proj = o3.Linear(self.uniform_irreps, self.irreps)

        # Per-edge Wigner-D rotations + Gaussian radial expansion.
        self.so3_rotation = SO3Rotation(self.lmax, self.mmax, use_rotation_mask=False)
        self.gauss = GaussianSmearing(0.0, cutoff, num_radial_basis, 2.0)
        edge_channels_list = [self.gauss.num_output, num_channels, num_channels]

        self.blocks = nn.ModuleList([
            TransBlockV3(
                num_in_channels=self.C, attn_hidden_channels=attn_hidden_channels,
                num_heads=num_heads, attn_alpha_channels=attn_alpha_channels,
                attn_value_channels=attn_value_channels, ffn_hidden_channels=ffn_hidden_channels,
                num_out_channels=self.C, lmax=self.lmax, mmax=self.mmax,
                so3_rotation=self.so3_rotation,
                attn_grid_resolution_list=[2 * (self.lmax + 1), 2 * (self.mmax + 1) + 1],
                ffn_grid_resolution_list=[2 * (self.lmax + 1), 2 * (self.lmax + 1) + 1],
                max_num_elements=1, edge_channels_list=edge_channels_list,
                use_atom_edge_embedding=False, attn_activation=attn_activation,
                ffn_activation=ffn_activation, use_grid_mlp=use_grid_mlp,
                norm_type=norm_type, drop_path_rate=0.0,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = get_normalization_layer(norm_type, lmax=self.lmax,
                                                  num_channels=self.C) if final_norm else None

    def _flat_to_embedding(self, h, irreps):
        """Flat [N, irreps.dim] (mul==C per degree) -> SO(3) embedding [N, (lmax+1)^2, C]."""
        N = h.shape[0]
        emb = h.new_zeros(N, (self.lmax + 1) ** 2, self.C)
        for (mul, ir), sl in zip(irreps, irreps.slices()):
            l = ir.l
            block = h[:, sl].reshape(N, mul, ir.dim)          # [N, C, 2l+1]
            emb[:, l * l:(l + 1) * (l + 1), :] = block.transpose(1, 2)
        return emb

    def _embedding_to_flat(self, emb, irreps):
        """SO(3) embedding [N, (lmax+1)^2, C] -> flat [N, irreps.dim] (mul==C per degree)."""
        N = emb.shape[0]
        out = []
        for (mul, ir), sl in zip(irreps, irreps.slices()):
            l = ir.l
            block = emb[:, l * l:(l + 1) * (l + 1), :]        # [N, 2l+1, C]
            out.append(block.transpose(1, 2).reshape(N, mul * ir.dim))
        return torch.cat(out, dim=-1)

    def forward(self, x, pos, batch, edge_index=None):
        """x: [N, irreps.dim], pos: [N, 3], batch: [N] -> [N, irreps.dim]."""
        if edge_index is None:
            edge_index = within_shape_edges(batch)
        if edge_index.shape[1] == 0:
            return x                                          # no edges -> identity

        src, dst = edge_index[0], edge_index[1]
        edge_vec = pos[src] - pos[dst]
        self.so3_rotation.set_wigner(init_edge_rot_mat(edge_vec))
        edge_distance = self.gauss(edge_vec.norm(dim=1))

        emb = self._flat_to_embedding(self.in_proj(x), self.in_irreps)
        for block in self.blocks:
            emb = block(emb, None, None, edge_distance, edge_index, None, batch)
        if self.final_norm is not None:
            emb = self.final_norm(emb)

        out = self.out_proj(self._embedding_to_flat(emb, self.uniform_irreps))
        if self.verbose:
            print("--------------EquiformerTransformer --------------")
            print("irreps:", self.irreps, "| lmax:", self.lmax, "| C:", self.C,
                  "| edges:", edge_index.shape[1])
            print("out.shape:", out.shape)
            print("--------------Finished --------------")
        return x + out                                        # residual keeps it in-place
