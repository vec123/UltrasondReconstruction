import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
import torch
from torch_geometric.utils import degree
from typing import Tuple  

from src.learning.modules.equivariant.irreps_utils import (
    scalar_features,
    expand_per_irrep_gate,
)
from src.learning.modules.pos_encodings.radial_fourier import RadialFourier

class GatingBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        # Defining the layers in the constructor
        self.dense1 = nn.Linear(input_dim, hidden_dim) # Note: requires input dimension
        self.dense2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # Implementation of the MLP gate logic
        x = self.dense1(x)
        x = F.silu(x)
        x = self.dense2(x)

        return x


class SelfInteraction(nn.Module):
    def __init__(self, in_irreps, target_irreps, sh_lmax=1, gate_hidden_dim=64, verbose=True):
        super().__init__()
        # self.in_irreps = o3.Irreps(in_irreps)
        self.in_irreps = self.limit_irreps(in_irreps, sh_lmax)
        self.target_irreps = o3.Irreps(target_irreps)
        self.sh_lmax = sh_lmax
        self.gate_hidden_dim = gate_hidden_dim
        self.verbose = verbose

        # Tensor Product: V ⊗ V (equivariant self-interaction / "square").
        self.tp = o3.FullyConnectedTensorProduct(
            self.in_irreps, self.in_irreps,
            self.in_irreps,  # Output irreps of the interaction
            internal_weights=True
        )

        # Concatenated features V ⊕ (V ⊗ V); order preserved to match torch.cat.
        self.concat_irreps = self.in_irreps + self.in_irreps

        # MLP gating: one gate per irrep, computed from the invariant scalars.
        self.num_scalars = self.concat_irreps.count("0e")
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.num_scalars, gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(gate_hidden_dim, self.concat_irreps.num_irreps)  # one gate per irrep
        )

        # Final Linear projection: (V ⊕ V⊗V) -> target.
        self.linear_out = o3.Linear(self.concat_irreps, self.target_irreps)

    def limit_irreps(self, irreps_str, max_l):
        """Filters an irreps string to keep only l <= max_l."""
        irreps = o3.Irreps(irreps_str)
        filtered = [ir for ir in irreps if ir.ir.l <= max_l]
        return o3.Irreps(filtered)

    def forward(self, node_features):
        # Step 1: V ⊗ V, then concatenate with V -> V ⊕ (V ⊗ V).
        interaction = self.tp(node_features, node_features)
        v_concat = torch.cat([node_features, interaction], dim=-1)

        # Step 2: per-irrep gate from the invariant scalars (keeps equivariance).
        scalars = scalar_features(v_concat, self.concat_irreps)
        gate = torch.sigmoid(self.gate_mlp(scalars))
        gate = expand_per_irrep_gate(gate, self.concat_irreps)
        gated = v_concat * gate

        # Step 3: linear projection to the target irreps.
        out = self.linear_out(gated)
        if self.verbose:
            print("--------------SelfInteraction --------------")
            print("in_irreps: ", self.in_irreps)
            print("target_irreps: ", self.target_irreps)
            print("out.shape: ", out.shape)
            print("self.linear_out.irreps_out: ", self.linear_out.irreps_out)
            print("--------------Finished --------------")
        return out


class EquivariantSpatialConv(MessagePassing):
    """Shared base for equivariant spatial message passing (Tensor-Field-Network style).

    Builds the per-edge equivariant message ``lin_msg( [x_j ⊕ (x_j ⊗ Y(rel_pos))] * gate )``
    and aggregates it at the receivers with an AREA-WEIGHTED mean::

        out_i = ( Σ_j a_j · m_ij ) / ( Σ_j a_j )

    where ``a_j`` is the sender's surface area (mass). Passing no area is equivalent to
    ``a_j ≡ 1``, which is exactly the plain ``1/degree`` mean — so the original numerics are
    reproduced when areas are absent. Area is an invariant (0e) scalar, so weighting the
    messages by it preserves equivariance.

    Subclasses provide only ``forward`` (the graph convention): ``SpatialConvolution`` is a
    homogeneous graph with a self-residual; ``BipartiteSpatialConvolution`` is a bipartite
    source→target graph with featureless targets.
    """

    def __init__(self, in_irreps, target_irreps, sh_lmax=4, r_max=2, verbose=True):
        super().__init__(aggr='add')  # "add" aggregation; normalization is explicit below
        self.verbose = verbose
        self.in_irreps = o3.Irreps(in_irreps)
        self.target_irreps = o3.Irreps(target_irreps)
        self.sh_lmax = sh_lmax

        # Spherical Harmonics of the relative position (geometric part of the message).
        self.sh = o3.SphericalHarmonics([l for l in range(1, sh_lmax + 1)], normalize=True)

        # Tensor Product for messages; weights supplied per-edge by the radial net.
        self.tp = o3.FullyConnectedTensorProduct(
            self.in_irreps, self.sh.irreps_out, self.in_irreps,
            shared_weights=False, internal_weights=False,
        )
        self.num_weights = self.tp.weight_numel
        # Radial network: message weights as a function of edge length.
        self.radial = RadialFourier(num_freqs=8, r_max=r_max)
        self.weight_net = nn.Sequential(
            nn.Linear(2 * 8, 32), nn.SiLU(), nn.Linear(32, self.num_weights)
        )
        # Message features x_j ⊕ (x_j ⊗ Y); order preserved to match torch.cat.
        self.geo_irreps = self.in_irreps + self.in_irreps
        self.lin_msg = o3.Linear(self.geo_irreps, self.target_irreps)

        # Gating: one gate per message irrep, from sender + receiver invariant scalars.
        self.num_scalars = self.in_irreps.count("0e")
        self.gate_net = GatingBlock(
            input_dim=2 * self.num_scalars,
            hidden_dim=64,
            out_dim=self.geo_irreps.num_irreps,
        )

    def message(self, x_i, x_j, pos_i, pos_j, area_j):
        # x_j: sender features, x_i: receiver features; area_j: sender area (or ones).
        rel_pos = pos_j - pos_i
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        edge_sh = self.sh(rel_pos)

        # Equivariant message: (x_j ⊗ Y) with radial weights, concatenated with x_j.
        weights = self.weight_net(self.radial(dist))
        tp_msg = self.tp(x_j, edge_sh, weights)
        geo_features = torch.cat([x_j, tp_msg], dim=-1)

        # Per-irrep gate from sender + receiver invariant scalars (keeps equivariance).
        x_i_0e = scalar_features(x_i, self.in_irreps)
        x_j_0e = scalar_features(x_j, self.in_irreps)
        gate = torch.sigmoid(self.gate_net(torch.cat([x_i_0e, x_j_0e], dim=-1)))
        gate = expand_per_irrep_gate(gate, self.geo_irreps)

        # Weight the message by the sender's area (a_j ≡ 1 -> the ordinary message).
        return self.lin_msg(geo_features * gate) * area_j

    @staticmethod
    def _node_area(area, n_nodes, ref):
        """Per-node area column ``[n, 1]`` on ref's device/dtype; ``None`` -> ones."""
        if area is None:
            return torch.ones(n_nodes, 1, device=ref.device, dtype=ref.dtype)
        return area.reshape(-1, 1).to(device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _inv_area_norm(area_edge, receiver_index, dim_size):
        """``1 / Σ_j a_j`` per receiver, clamped exactly like the original ``1/degree``.
        ``area_edge`` is the per-edge sender area ``[E]``; all-ones gives ``1/degree``."""
        Z = scatter(area_edge, receiver_index, dim=0, dim_size=dim_size, reduce='add')
        return (1.0 / torch.clamp(Z, min=1.0)).view(-1, 1)


class SpatialConvolution(EquivariantSpatialConv):
    """Equivariant message passing over a homogeneous graph, with a self-residual.
    ``forward`` optionally takes a per-node ``area`` (surface measure); when given, the
    neighbour aggregation is an area-weighted mean instead of a ``1/degree`` mean. Omitting
    it reproduces the original behavior exactly.
    """

    def forward(self, x, pos, edge_index, area=None):
        src, dst = edge_index[0], edge_index[1]
        area_node = self._node_area(area, x.size(0), x)             # [N, 1]

        # Aggregate area-weighted messages at the receivers (self-residual added in update).
        out = self.propagate(edge_index, x=x, pos=pos, area=area_node)
        inv = self._inv_area_norm(area_node[src].view(-1), dst, x.size(0))
        if self.verbose:
            print("--------------SpatialConvolution --------------")
            print("target_irreps: ", self.target_irreps)
            print("out.shape: ", out.shape)
            print("self.lin_msg.irreps_out: ", self.lin_msg.irreps_out)
            print("--------------Finished --------------")
        return out * inv  # Scale the (aggregated + residual) features by 1 / Σ a_j

    def update(self, aggr_out, x):
        # aggr_out is the (area-weighted) sum of messages; add a simplified self-residual.
        return aggr_out + x


class BipartiteSpatialConvolution(EquivariantSpatialConv):
    """Equivariant message passing over a bipartite (source -> target) graph.

    Aggregates SOURCE node features (e.g. full-graph nodes, carrying ``in_irreps``)
    onto TARGET nodes (e.g. supernodes) that have positions but NO input features.
    The message is the same equivariant construction as ``SpatialConvolution``
    (``x_j (x) Y`` with a per-edge radial-net weight, concatenated with ``x_j`` and
    per-irrep gated), summed at the targets and normalized by (area-weighted) in-degree.
    There is no self-residual, since the targets are featureless.

    ``forward`` takes ``edge_index`` in the bipartite ``Data`` convention produced by
    ``build_bipartite_graph`` — ``row 0 = target (supernode)``, ``row 1 = source
    (full node)`` — and flips it internally to PyG's ``source_to_target`` layout.
    """

    def forward(self, x_src, pos_src, pos_dst, edge_index, area_src=None):
        """
        x_src     : [F, in_irreps.dim]  source (full-node) features
        pos_src   : [F, 3]              source positions
        pos_dst   : [S, 3]              target (supernode) positions
        edge_index: [2, E] with row0=target (super), row1=source (full)
                    (the build_bipartite_graph convention)
        area_src  : [F] optional per-source-node area (surface measure); None -> 1/degree
        returns   : [S, target_irreps.dim]  supernode features
        """
        num_target = pos_dst.size(0)
        # Flip to PyG source_to_target: [source (full), target (super)].
        edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        src, col = edge_index[0], edge_index[1]

        area_node = self._node_area(area_src, x_src.size(0), x_src)          # [F, 1]
        # Targets are featureless; zero placeholders feed the receiver-scalar gate + area.
        x_dst = x_src.new_zeros(num_target, self.in_irreps.dim)
        area_dst = x_src.new_ones(num_target, 1)                            # unused (target side)

        out = self.propagate(
            edge_index, x=(x_src, x_dst), pos=(pos_src, pos_dst),
            area=(area_node, area_dst), size=(x_src.size(0), num_target),
        )
        inv = self._inv_area_norm(area_node[src].view(-1), col, num_target)
        if self.verbose:
            print("--------------BipartiteSpatialConvolution --------------")
            print("in_irreps: ", self.in_irreps, " target_irreps: ", self.target_irreps)
            print("out.shape: ", out.shape)
            print("--------------Finished --------------")
        return out * inv


def _sample_neighbors(
    edge_index: torch.Tensor,
    area_src: torch.Tensor,
    num_target: int,
    num_samples: int,
    generator: torch.Generator = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs vectorized weighted random sampling (Efraimidis-Spirakis)
    without replacement per target node.
    """
    device = edge_index.device
    targets, sources = edge_index[0], edge_index[1]
    E = edge_index.size(1)

    # 1. Calculate target node degrees (K_i)
    deg_target = degree(targets, num_nodes=num_target, dtype=area_src.dtype)

    # 2. Compute Efraimidis-Spirakis keys: u^(1/a_j)
    edge_areas = area_src[sources].clamp_min(1e-12)
    u = torch.rand(E, generator=generator, device=device)
    keys = u.pow(1.0 / edge_areas)

    # 3. Sort keys globally (descending), then group stably by target ID
    order = keys.argsort(descending=True)
    targets_sorted = targets[order]
    group_idx = torch.argsort(targets_sorted, stable=True)
    
    targets_grouped = targets_sorted[group_idx]
    sampled_edge_ids = order[group_idx]

    # 4. Filter top-M edges per target using grouped offsets
    counts = torch.bincount(targets_grouped, minlength=num_target)
    offsets = counts.cumsum(0) - counts
    rank = torch.arange(E, device=device) - offsets[targets_grouped]

    max_rank = torch.clamp(deg_target[targets_grouped], max=num_samples)
    keep = rank < max_rank

    return sampled_edge_ids[keep], deg_target


class MonteCarloBipartiteSpatialConvolution(EquivariantSpatialConv):
    """
    Bipartite spatial convolution using a Monte Carlo approximation 
    of a surface integral via reproducible, vectorized neighborhood sampling.
    """

    def forward(self, x_src, pos_src, pos_dst, edge_index, area_src=None, num_samples=30, seed=1):
        num_target, num_source = pos_dst.size(0), x_src.size(0)
        
        if edge_index.size(1) == 0:
            out_dim = self.out_irreps.dim if hasattr(self, 'out_irreps') else x_src.size(-1)
            return x_src.new_zeros(num_target, out_dim)

        # Initialize local generator for reproducible random state
        gen = None
        if seed is not None:
            gen = torch.Generator(device=edge_index.device).manual_seed(seed)

        area_node = self._node_area(area_src, num_source, x_src).view(-1)
        if num_samples is None:
            # No sampling: aggregate over EVERY neighbour, which is what makes the
            # result exactly SE(3)-invariant. Sampling cannot be, even with a fixed
            # seed: _sample_neighbors draws u positionally over edge_index, and a
            # radius graph emits the same edge SET in a rotation-dependent ORDER, so
            # the same seed selects a different subset for a rotated input.
            sampled_edges = torch.arange(edge_index.size(1), device=edge_index.device)
            deg_target = degree(edge_index[0], num_nodes=num_target, dtype=x_src.dtype)
        else:
            sampled_edges, deg_target = _sample_neighbors(
                edge_index, area_node, num_target, num_samples, gen
            )

        sub_edge_index = edge_index[:, sampled_edges]

        # Flip to PyG source-to-target layout: [source (full), target (super)]
        sub_edge_index = torch.stack([sub_edge_index[1], sub_edge_index[0]], dim=0)
        col = sub_edge_index[1]

        # Message Passing
        x_dst = x_src.new_zeros(num_target, self.in_irreps.dim)
        out = self.propagate(
            sub_edge_index, 
            x=(x_src, x_dst), 
            pos=(pos_src, pos_dst),
            area=(area_node.unsqueeze(-1), x_src.new_ones(num_target, 1)), 
            size=(num_source, num_target),
        )

        # Scale output by (K_total / M_sampled) for unbiased importance sampling
        m_sampled = torch.bincount(col, minlength=num_target).to(x_src.dtype).clamp_min(1.0)
        scale = (deg_target / m_sampled).unsqueeze(-1)
        
        return out * scale