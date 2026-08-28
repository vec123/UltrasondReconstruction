from dataclasses import replace

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import softmax as scatter_softmax
from e3nn import o3
from src.learning.layers.equivariant.Self_Spatial_layer import EquiLayer
from src.learning.modules.equivariant.interaction import (
    BipartiteSpatialConvolution, MonteCarloBipartiteSpatialConvolution)

from src.learning.modules.equivariant.transformer import build_equivariant_transformer
from src.learning.modules.equivariant.irreps_utils import scalar_features, vector_features

from src.learning.registry import Registry

class GroupEncoder(nn.Module):
    def __init__(self, layers_cfg,
                 latent_dim: int = 5,
                 output_irreps: str = None,
                 readout: str = "mean",
                 readout_heads: int = 1,
                 supernode_sh_lmax: int = 4,
                 transformer_type: str = "se3",
                 transformer_cfg: dict = None,
                 area_pool: bool = False,
                 latent_mode: str = "gaussian",
                 verbose: bool = False):
        
        """``layers_cfg``: a non-empty list of per-layer dicts, one EquiLayer each,
        threaded in order (layer i's output feeds layer i+1's input):

            {"in_irreps": ..., "target_irreps": ..., "spatial_sh_lmax": ...,
             "interaction_sh_lmax": ...}   # interaction_sh_lmax optional, default 4

        ``spatial_sh_lmax`` -- every layer's caller
        must state it explicitly, no default, no silent inherit of an encoder-wide
        value

        ``latent_mode`` selects the LatentHead strategy from the registry:
        ``"gaussian"`` (VAE: mu/logvar) or ``"deterministic"`` (auto-encoder: a
        plain latent). Both emit [B, latent_dim], so nothing downstream branches.
        """

        super().__init__()
        self.latent_dim = latent_dim
        self.readout = readout
        self.area_pool = area_pool
        self.verbose = verbose

        if not layers_cfg:
            raise ValueError("GroupEncoder requires at least one entry in layers_cfg.")

        self.in_irreps_str = layers_cfg[0]["in_irreps"]
        self.intermediate_irreps_str = layers_cfg[-1]["target_irreps"]
        self.output_irreps_str = output_irreps or f"{latent_dim}x0e + 2x1o"

        # Each layer's output must match the next layer's input -- otherwise the
        # tensor-product chain inside EquiLayer fails with an opaque shape
        # mismatch deep in the stack instead of a clear message here.
        for i in range(len(layers_cfg) - 1):
            out_ir = o3.Irreps(layers_cfg[i]["target_irreps"])
            next_in_ir = o3.Irreps(layers_cfg[i + 1]["in_irreps"])
            if out_ir != next_in_ir:
                raise ValueError(
                    f"layers_cfg[{i}]['target_irreps'] ({layers_cfg[i]['target_irreps']!r}) "
                    f"must equal layers_cfg[{i + 1}]['in_irreps'] "
                    f"({layers_cfg[i + 1]['in_irreps']!r})."
                )

        self.layers = nn.ModuleList([
            EquiLayer(in_irreps=l["in_irreps"],
                        target_irreps=l["target_irreps"],
                        spatial_sh_lmax=l["spatial_sh_lmax"],
                        interaction_sh_lmax=l.get("interaction_sh_lmax", 4),
                        verbose=verbose)
            for l in layers_cfg
        ])

         # Optional supernode aggregation: an equivariant bipartite conv maps the
        # full-node features onto the supernodes (in_irreps == out_irreps, so the
        # scalar count is unchanged). When enabled, the readout below pools over
        # supernodes instead of nodes.
        # Attention: if supergraph is None these are dead weights 
        self.supernode_in_irreps = o3.Irreps(self.intermediate_irreps_str)   
        self.supernode_out_irreps = o3.Irreps(self.intermediate_irreps_str)   
        self.supernode_conv = MonteCarloBipartiteSpatialConvolution(
                in_irreps=self.supernode_in_irreps, target_irreps=self.supernode_out_irreps,
                sh_lmax=supernode_sh_lmax, verbose=verbose,
            )
                
        # Optional equivariant transformer refinement of the pooled features, applied
        # after supernode aggregation and before the scalars are filtered out. Selectable
        # backend ('se3' | 'equiformer'); in-place on out_irreps so nothing downstream
        # changes. Disable with transformer_type=None.
       
        self.equi_transformer_irreps = o3.Irreps(self.intermediate_irreps_str)    
        self.equi_transformer = build_equivariant_transformer(
            transformer_type, self.equi_transformer_irreps, transformer_cfg, verbose=verbose,
        )

       
        self.out_irreps = o3.Irreps(self.output_irreps_str)
        self.final_linear = o3.Linear(self.intermediate_irreps_str, self.output_irreps_str)
         
        # Latent head: owns the readout AND the distribution the
        # pooled scalars parameterize. Resolved through the lazy Registry, 
        # adding a third mode is a registration line, not an edit here.
        #
        # Constructed HERE, between final_linear and weight_net, and internally in
        # the order readout_pool -> mu_net -> var_net: 
        # nn.Linear draws from the global RNG at construction,
        # moving this call would change every seeded init and break the
        # characterization baseline.
        self.latent_mode = latent_mode
        self.latent_head = Registry.create(
            "latent_head", latent_mode,
            latent_dim=latent_dim, readout=readout, readout_heads=readout_heads,
        )

        # Per-token attention logits. Stays on the encoder (NOT the head) because
        # the pose head below shares the same weights (might change in future) -- see forward.
        self.weight_net = nn.Linear(latent_dim, 1)
        # Separate attention network for rotation vectors: takes scalars + vector norms
        # This allows rotation pooling weights to be sensitive to vector magnitude/equivariance
        self.vector_weight_net = nn.Linear(latent_dim + 2, 1)

        self.mu_bn = nn.BatchNorm1d(latent_dim)
        self.mu_ln = nn.LayerNorm(latent_dim)

        # Both readouts ("mean" and "attention") collapse to one token per shape
        # (see forward, below). Exposed so a factory can check encoder/decoder
        # compatibility on the constructed objects, multi-token not yet supported.
        self.n_tokens = 1


    def forward(self,graph, supergraph, monte_carlo_reg = True):

        x = graph.x
        pos = graph.pos
        
        edge_index = graph.edge_index
        batch_idx = graph.batch
        node_area = getattr(graph, 'area', None) if self.area_pool else None
        node_normal = getattr(graph, 'normal', None)

        if node_normal is not None:
            x = torch.cat([x, node_normal], dim=-1)
     
        if supergraph is not None:
            super_pos=supergraph.pos
            super_batch=supergraph.batch
            super_edge_index=supergraph.edge_index

        # Message Passing on the full graph.
        for i, layer in enumerate(self.layers):
            if self.verbose:
                print(f"---------Layer {i} with: "
                     f" x: {x.shape}, "
                     f" pos: {pos.shape}, "
                     f" edge index: {edge_index.shape}")

            x = layer(x, pos, edge_index, area=node_area)
            if self.verbose:
                print(f"Layer {i} output shape: {x.shape}")

        # Optional supernode aggregation
        if supergraph is not None:
            if super_pos is None or super_batch is None or super_edge_index is None:
                raise ValueError(
                    "use_supernodes=True requires super_pos, super_batch and super_edge_index."
                )
            if not monte_carlo_reg:
                edge_sampling_seed = 1 
            else:
                edge_sampling_seed = None
            feat = self.supernode_conv(x, pos, super_pos, super_edge_index,
                                       area_src=node_area, seed= edge_sampling_seed )  # [S, out_irreps.dim]
            pool_batch = super_batch
            pool_pos = super_pos
            pool_area = getattr(supergraph, 'area', None)          # supernode mass, if provided
        else:
            feat = x
            pool_batch = batch_idx
            pool_pos = pos
            pool_area = getattr(graph, 'area', None)               # per-vertex area, if provided

        # Area weighting turns sums over the pooled set into surface integrals; 
        #  Off unless area_pool and an 'area' attribute are present 
        # -> falls back to the uniform behavior.
        if not self.area_pool or pool_area is None:
            pool_area = None
        elif pool_area.dim() == 1:
            pool_area = pool_area.reshape(-1, 1)                   # [n_pool, 1]

        # Equivariant transformer refinement over the pooled set (supernodes or nodes),
        # on the full irreps BEFORE the invariant scalars are filtered out.
        # and before the irreps are reduced to the final encoding set
        if self.equi_transformer is not None:
            feat = self.equi_transformer(feat, pool_pos, pool_batch)

        # reduce to the final encoding set
        feat = self.final_linear(feat)

        # Global Pooling (VAE Latent Space) over the invariant scalars.
        scalars = scalar_features(feat, self.out_irreps)          # [n_nodes, #0e]
        n_saclars = scalars.shape[1]
        assert n_saclars == self.latent_dim

        # Pass size= to every pool so a non-contiguous pool_batch can't drop a row.
        num_graphs = int(pool_batch.max().item()) + 1

        # Per-graph attention weights: softmax is taken WITHIN each shape (grouped by
        # pool_batch), so the weights sum to 1 per shape and don't leak across shapes.
        # Computed once and shared by the scalar readout and the vector/pose head. Use
        # with global_add_pool so the softmax is the single normalization (a following
        # global_mean_pool would divide again -> a per-shape latent crushed toward 0).
        scalar_logit_gate = self.weight_net(scalars)
        if pool_area is not None:
            # Area-weighted attention: softmax(logit + log a) = a·e^logit / Σ a·e^logit,
            # so a denser sampling no longer over-counts (a per-shape scale of a cancels).
            scalar_logit_gate = scalar_logit_gate + torch.log(pool_area.clamp_min(1e-12))
        weights = scatter_softmax(scalar_logit_gate, pool_batch)                     # [n_pool, 1]

        # Latent head: the readout AND the distribution live here. Returns an
        # EncoderOutput carrying only the latent fields -- gaussian -> mu/logvar,
        # deterministic -> latent -- which the pose fields are added to below.
        latent_out = self.latent_head(scalars, weights, pool_batch, num_graphs)

        # Equivariant Output (Rotation & Translation)
        vectors = vector_features(feat, self.out_irreps, '1o')    #[n_nodes, n_vec, 3]
        n_vec = vectors.shape[1]
        assert n_vec == 2

        # Compute vector magnitude as input to attention (makes rotation vectors "speak for themselves")
        # This breaks the pure scalar->weight dependency and allows weights to adapt to vector scale
        vector_norms = torch.norm(vectors, dim=-1)  # [n_nodes, n_vec]
        vector_norms_flat = vector_norms.reshape(vector_norms.shape[0], -1)  # [n_nodes, n_vec]
        vector_features_for_attention = torch.cat([scalars, vector_norms_flat], dim=-1)  # [n_nodes, latent_dim + n_vec]

        vector_logit_gate = self.vector_weight_net(vector_features_for_attention)
        weights = scatter_softmax(vector_logit_gate, pool_batch)
        # Weighted sum over tokens (same per-shape weights), then reshape to [B, n_vec, 3].
        vec_weighted = (weights.unsqueeze(-1) * vectors).reshape(vectors.shape[0], -1)
        vec_graph = global_add_pool(vec_weighted, pool_batch, size=num_graphs).reshape(-1, n_vec, 3)

        # Extract v1, v2 for rotation matrix (2 vectors -> 3x3 R)
        v1, v2 = vec_graph[:, 0, :], vec_graph[:, 1, :]

        rot_matrix = self.get_rotation_matrix_from_two_vectors(v1, v2)

        # Translation: center of mass (over the pooled token set). Area-weighted when
        # areas are available and differentiable -> the true surface centroid (a plain mean
        # over-weights densely sampled regions). Falls back to unweighted mean if areas
        # don't have gradients (e.g., SO(3)-only training).
        # Skip translation for SO(3)-only mode to avoid breaking gradient flow
        # (translation will be re-enabled for SE(3) training in the future)

        if pool_area is not None and pool_area.requires_grad:
            denom = global_add_pool(pool_area, pool_batch, size=num_graphs).clamp_min(1e-12)
            transl = global_add_pool(pool_area * pool_pos, pool_batch, size=num_graphs) / denom
        else:
            # Fall back to unweighted mean to preserve gradient flow
            transl = global_mean_pool(pool_pos, pool_batch, size=num_graphs)

        # For SO(3) training, translation often doesn't require gradients (it's not used anyway)
        # This is expected and OK—only rotation matters for SO(3)

        # Store the raw pose vectors (v1, v2) in aux for logging
        aux_data = latent_out.aux.copy() if latent_out.aux else {}
        aux_data['v1'] = v1
        aux_data['v2'] = v2

        # Attach the pose to whatever latent fields the head produced, without this
        # method needing to know which kind of head it holds.
        return replace(latent_out, rotation=rot_matrix, translation=transl, aux=aux_data)
    

    def get_rotation_matrix_from_two_vectors(self, v1, v2):
        """Compute rotation matrix from two vectors using Gram-Schmidt orthogonalization.

        IMPORTANT: Encoder must learn non-zero pose vectors. If vectors remain near-zero,
        this will raise an error to alert the user to fix the encoder initialization.

        Args:
            v1: [B, 3] first vector
            v2: [B, 3] second vector

        Returns:
            [B, 3, 3] rotation matrix with orthonormal columns

        Raises:
            RuntimeError: if vectors remain near-zero after multiple steps (encoder not learning)
        """
        B = v1.shape[0]
        device = v1.device
        dtype = v1.dtype

        # Compute vector norms
        v1_norm = torch.norm(v1, dim=-1, keepdim=True)  # [B, 1]
        v2_norm = torch.norm(v2, dim=-1, keepdim=True)  # [B, 1]

        # Check if vectors are too small (encoder hasn't learned pose yet)
        # Threshold: 1e-4 is quite generous
        min_norm_threshold = 1e-4
        is_dead = (v1_norm < min_norm_threshold) | (v2_norm < min_norm_threshold)  # [B, 1]

        if is_dead.any():
            n_dead = is_dead.sum().item()
            print(f"[ROT_ERROR] {n_dead}/{B} samples have near-zero pose vectors!")
            print(f"  v1_norm: {v1_norm.view(-1).tolist()}")
            print(f"  v2_norm: {v2_norm.view(-1).tolist()}")
            raise RuntimeError(
                f"Encoder pose vectors are too small ({n_dead}/{B} samples). "
                f"The encoder is not learning the pose. This usually means:\n"
                f"  1. Encoder irreps must have 1o (pseudo-vector) components for pose\n"
                f"  2. Check that readout is configured correctly\n"
                f"  3. Try different initialization or learning rate\n"
                f"  4. Verify that pose is being used in the loss computation"
            )

        # Normalize with epsilon to avoid division by zero
        eps = 1e-8
        u = v1 / (v1_norm + eps)  # [B, 3], normalized first vector

        # Gram-Schmidt orthogonalization
        dot = torch.sum(u * v2, dim=-1, keepdim=True)  # [B, 1]
        w_raw = v2 - dot * u  # [B, 3]
        w_norm = torch.norm(w_raw, dim=-1, keepdim=True)  # [B, 1]
        w = w_raw / (w_norm + eps)  # [B, 3], normalized orthogonal vector

        # Third basis vector via cross product
        v3 = torch.cross(u, w, dim=-1)  # [B, 3]

        # Stack into rotation matrix
        R = torch.stack([u, w, v3], dim=-1)  # [B, 3, 3]

        return R