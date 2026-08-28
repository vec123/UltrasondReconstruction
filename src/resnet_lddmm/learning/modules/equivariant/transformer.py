"""SE(3)-equivariant graph Transformer (e3nn + PyG port of NVIDIA's fast model).

This is a native re-implementation of the attention block of NVIDIA's fast
SE(3)-Transformer (``Theory/DeepLearningExamples/.../SE3Transformer``), rewritten to
run in this project's stack — flat ``[N, irreps.dim]`` e3nn tensors and
``torch_geometric`` graphs — instead of DGL. DGL ships no Windows CUDA wheel for
torch 2.4 (ABI mismatch), so the DGL implementation cannot run here; the speed-ups it
provides are almost all NOT DGL-specific and are reproduced below:

  * geometry (spherical harmonics + radial basis) is computed ONCE per forward and
    shared across all layers (the "precompute / cached bases" idea);
  * keys and values are produced by a SINGLE fused TFN tensor-product conv per layer
    (NVIDIA's fused K+V optimisation);
  * per-receiver softmax uses ``torch_geometric.utils.softmax`` (the direct analogue of
    DGL ``edge_softmax``), replacing the python node loop in ``EquivariantAttention``;
  * weighted value aggregation uses ``torch_scatter.scatter`` (analogue of DGL
    ``copy_e_sum``);
  * runs under ``torch.cuda.amp.autocast`` on Tensor-Core GPUs.

The module is *in-place*: ``irreps_in == irreps_out``, so it can be dropped in right
after supernode aggregation and before the invariant scalars are filtered out, without
changing any downstream shapes.

Equivariance is exact (up to float tolerance) by construction: attention logits are
invariant per-head Euclidean dot products of query/key (real Wigner-D matrices are
orthogonal, so the dot within each irrep is a rotation invariant); values are
equivariant and are only ever scaled by those invariant weights and summed.
"""

import math

import torch
import torch.nn as nn
from e3nn import o3
from torch_geometric.utils import softmax as edge_softmax
from torch_scatter import scatter

from src.learning.modules.equivariant.layer_norm import EquivariantLayerNorm
from src.learning.modules.pos_encodings.radial_fourier import RadialFourier


def _distinct_irreps(irreps):
    """Distinct ``o3.Irrep`` types (l, p) present in ``irreps``, order preserved."""
    seen = []
    for _, ir in o3.Irreps(irreps):
        if ir not in seen:
            seen.append(ir)
    return seen


def _hidden_fiber(irreps, channels):
    """A fiber with ``channels`` multiplicities of every distinct irrep type in ``irreps``."""
    return o3.Irreps([(channels, ir) for ir in _distinct_irreps(irreps)])


def within_shape_edges(batch):
    """Fully-connected (no self-loops) edges within each graph of the batch.

    Returns ``edge_index`` [2, E] with row 0 = sender, row 1 = receiver. Both directions
    are present, so the sender/receiver convention is symmetric. Shared by the SE(3) and
    Equiformer transformer wrappers.
    """
    device = batch.device
    order = torch.argsort(batch)
    counts = torch.bincount(batch)
    src_all, dst_all, start = [], [], 0
    for cnt in counts.tolist():
        if cnt > 1:
            idx = order[start:start + cnt]
            ii, jj = torch.meshgrid(idx, idx, indexing='ij')
            mask = ii != jj
            src_all.append(ii[mask])
            dst_all.append(jj[mask])
        start += cnt
    if src_all:
        return torch.stack([torch.cat(src_all), torch.cat(dst_all)], dim=0)
    return torch.empty(2, 0, dtype=torch.long, device=device)


class SE3AttentionLayer(nn.Module):
    """One SE(3)-equivariant multi-head graph-attention block (PyG-native).

    Mirrors ``AttentionBlockSE3`` of the NVIDIA SE(3)-Transformer: a query is a linear
    projection of the receiver node features; keys and values are produced by a single
    geometric (TFN) tensor-product convolution over the sender features; attention is an
    invariant softmax over incoming edges; the aggregated value is concatenated with the
    residual node features and projected back to ``irreps``.
    """

    def __init__(self, irreps, sh_irreps, num_heads, hidden_channels, radial_dim,
                 verbose=False):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.num_heads = num_heads
        self.verbose = verbose

        assert hidden_channels % num_heads == 0, \
            f"hidden_channels ({hidden_channels}) must be divisible by num_heads ({num_heads})"
        self.head_c = hidden_channels // num_heads

        # Key/query and value hidden fibers (same irrep types as the input, wider).
        self.kq_irreps = _hidden_fiber(self.irreps, hidden_channels)
        self.v_irreps = _hidden_fiber(self.irreps, hidden_channels)

        # Query: plain linear projection of receiver node features -> kq fiber.
        self.to_q = o3.Linear(self.irreps, self.kq_irreps)

        # Keys + Values: ONE fused geometric tensor product (sender feats (x) SH),
        # with per-edge weights from a radial network (like SpatialConvolution).
        self.kv_out_irreps = (self.v_irreps + self.kq_irreps)
        self.tp_kv = o3.FullyConnectedTensorProduct(
            self.irreps, o3.Irreps(sh_irreps), self.kv_out_irreps,
            shared_weights=False, internal_weights=False,
        )
        self.kv_weight_net = nn.Sequential(
            nn.Linear(radial_dim, 32), nn.SiLU(),
            nn.Linear(32, self.tp_kv.weight_numel),
        )

        # Output: concat(residual node feats, aggregated value) -> irreps.
        self.to_out = o3.Linear(self.irreps + self.v_irreps, self.irreps)

        # Precomputed (slice, dim) per irrep block for head-wise reshaping.
        self._kq_blocks = [(sl, ir.dim) for (mul, ir), sl
                           in zip(self.kq_irreps, self.kq_irreps.slices())]
        self._v_blocks = [(sl, mul, ir.dim) for (mul, ir), sl
                          in zip(self.v_irreps, self.v_irreps.slices())]
        # Per-head key/query dimension (for the 1/sqrt(d) scaling).
        self.d_head = sum(self.head_c * dim for (_, dim) in self._kq_blocks)

    def _headwise_dot(self, q_e, k_e):
        """Invariant per-head attention logits: [E, kq_dim], [E, kq_dim] -> [E, heads].

        Computed block-by-block so each head sums the Euclidean dot over whole irrep
        components (which is a rotation invariant) — no cross-irrep mixing.
        """
        E = q_e.shape[0]
        H, c = self.num_heads, self.head_c
        logits = q_e.new_zeros(E, H)
        for sl, dim in self._kq_blocks:
            qb = q_e[:, sl].reshape(E, H, c, dim)
            kb = k_e[:, sl].reshape(E, H, c, dim)
            logits = logits + (qb * kb).sum(dim=(2, 3))
        return logits

    def forward(self, x, edge_index, edge_sh, radial_basis):
        # edge_index: [2, E] with row 0 = sender (src), row 1 = receiver (dst).
        src, dst = edge_index[0], edge_index[1]
        N = x.shape[0]

        # Query per receiver node; fused K+V per edge from the sender features.
        q = self.to_q(x)                                   # [N, kq_dim]
        w = self.kv_weight_net(radial_basis)               # [E, weight_numel]
        kv = self.tp_kv(x[src], edge_sh, w)                # [E, v_dim + kq_dim]
        v, k = kv[:, :self.v_irreps.dim], kv[:, self.v_irreps.dim:]

        # Invariant per-head logits, softmax over incoming edges of each receiver.
        logits = self._headwise_dot(q[dst], k) / math.sqrt(max(self.d_head, 1))
        alpha = edge_softmax(logits, dst, num_nodes=N)     # [E, heads]

        # Weighted value aggregation at receivers, per head, then merge heads.
        H, c = self.num_heads, self.head_c
        out_blocks = []
        for sl, mul, dim in self._v_blocks:
            vb = v[:, sl].reshape(-1, H, c, dim) * alpha[:, :, None, None]
            agg = scatter(vb, dst, dim=0, dim_size=N, reduce='add')   # [N, H, c, dim]
            out_blocks.append(agg.reshape(N, mul * dim))
        v_agg = torch.cat(out_blocks, dim=-1)              # [N, v_dim]

        out = self.to_out(torch.cat([x, v_agg], dim=-1))   # [N, irreps.dim]
        if self.verbose:
            print("--------------SE3AttentionLayer --------------")
            print("irreps:", self.irreps, "| heads:", self.num_heads,
                  "| edges:", edge_index.shape[1])
            print("out.shape:", out.shape)
            print("--------------Finished --------------")
        return out


class SE3Transformer(nn.Module):
    """Stack of SE(3)-equivariant attention layers over a point set.

    In-place on the irreps (``irreps_in == irreps_out``). By default it attends over a
    graph that is fully connected WITHIN each shape (built from ``batch``, no
    self-loops), matching the "attend only within your own shape" convention used
    elsewhere in the encoders; pass an explicit ``edge_index`` to override.

    ``precompute_bases`` (default ``False``) toggles caching of the (parameter-free)
    geometry — spherical harmonics + radial basis — across forwards. Keep it OFF when
    inputs are augmented per step (random rotations, point dropout, jitter): every new
    position tensor invalidates the cache, so leaving it off is both correct and free.
    """

    def __init__(self, irreps, num_layers=2, num_heads=4, num_channels=16,
                 lmax=2, radial_freqs=8, r_max=2.0, norm=True,
                 precompute_bases=False, verbose=False):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.num_heads = num_heads
        self.precompute_bases = precompute_bases
        self.verbose = verbose

        # hidden_channels must divide evenly into heads for every degree; round up.
        if num_channels % num_heads != 0:
            num_channels = ((num_channels + num_heads - 1) // num_heads) * num_heads

        # Geometry (shared across all layers): SH of relative positions + radial basis.
        self.sh = o3.SphericalHarmonics([l for l in range(1, lmax + 1)], normalize=True)
        self.radial = RadialFourier(num_freqs=radial_freqs, r_max=r_max)
        radial_dim = 2 * radial_freqs

        self.layers = nn.ModuleList([
            SE3AttentionLayer(self.irreps, self.sh.irreps_out, num_heads,
                              num_channels, radial_dim, verbose=verbose)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            EquivariantLayerNorm(self.irreps, verbose=False) if norm else nn.Identity()
            for _ in range(num_layers)
        ])

        self._basis_cache = None

    def _geometry(self, pos, edge_index):
        """Spherical-harmonic + radial basis of the edges (parameter-free; cacheable)."""
        if self.precompute_bases and self._basis_cache is not None:
            pos_ptr, ei_ptr, esh, rb = self._basis_cache
            if pos_ptr == pos.data_ptr() and ei_ptr == edge_index.data_ptr():
                return esh, rb
        src, dst = edge_index[0], edge_index[1]
        rel_pos = pos[src] - pos[dst]                      # sender - receiver
        dist = rel_pos.norm(dim=-1, keepdim=True)
        edge_sh = self.sh(rel_pos)
        radial_basis = self.radial(dist)
        if self.precompute_bases:
            self._basis_cache = (pos.data_ptr(), edge_index.data_ptr(), edge_sh, radial_basis)
        return edge_sh, radial_basis

    def forward(self, x, pos, batch, edge_index=None):
        """x: [N, irreps.dim], pos: [N, 3], batch: [N] -> [N, irreps.dim]."""
        if edge_index is None:
            edge_index = within_shape_edges(batch)

        if edge_index.shape[1] == 0:
            # No edges (e.g. every shape has a single node): identity pass-through.
            return x

        edge_sh, radial_basis = self._geometry(pos, edge_index)
        for layer, norm in zip(self.layers, self.norms):
            x = norm(layer(x, edge_index, edge_sh, radial_basis))
        return x


def build_equivariant_transformer(transformer_type, irreps, cfg=None, verbose=False):
    """Factory for the post-aggregation equivariant transformer.

    ``transformer_type``: ``None``/``'none'`` (disabled), ``'se3'`` (this module's
    SE(3)-Transformer port) or ``'equiformer'`` (the vendored EquiformerV3 block).
    ``cfg`` is forwarded as kwargs to the chosen module.
    """
    if transformer_type in (None, 'none', False):
        return None
    cfg = dict(cfg or {})
    if transformer_type == 'se3':
        return SE3Transformer(irreps, verbose=verbose, **cfg)
    if transformer_type == 'equiformer':
        # Lazy import to avoid a circular import (equiformer imports from this module).
        from src.learning.modules.equivariant.equiformer import EquiformerTransformer
        return EquiformerTransformer(irreps, verbose=verbose, **cfg)
    raise ValueError(
        f"unknown transformer_type {transformer_type!r} (expected None | 'se3' | 'equiformer')")
