"""Equivariant GNN backbone + scalar self-attention + Perceiver latent readout.

Pipeline (per graph in the batch):

  1. EquiLayer GNN over the point graph            -> node features (mixed irreps)
  2. Split off the invariant scalar (0e) channels  -> X in R^{n_nodes x n_scalar}
  3. Multi-head self-attention transformer (nodes) -> R^{n_nodes x n_scalar}
  4. Perceiver cross-attention with n_latent learned queries:
         Q: (n_latent x d_shared)  <- learned latent array . W_Q
         K: (n_nodes  x n_scalar)  . W_K -> (n_nodes x d_shared)
         V: (n_nodes  x n_scalar)  . W_V -> (n_nodes x d_shared)
     -> latent array  R^{n_latent x d_shared}   (d_shared == the latent dim)

Two optional post-readout stages let the same encoder serve both decoder paths:

  * ``reduce_stages`` — a PerceiverReducer that iteratively shrinks the token count,
    e.g. ``[8, 4, 2]`` -> ... -> 2 (or -> 1). Use this to collapse toward a single
    latent (or a mean/var pair) for the FoldingDecoder.
  * ``vae_mode``      — a LatentVAEHead applied to whatever token set remains, turning
    it into a sampled ``z`` + ``(mu, logvar, kl)`` regardless of token count. Leave
    the token set large (no reduction) to feed a PerceiverDecoder.

``forward`` returns an ``EncoderOutput``: ``latent`` is the (sampled) latent set; when
a VAE head is active ``mu``/``logvar`` are set and ``aux['kl']`` / ``aux['latent_set']``
carry the KL term and the pre-VAE tokens.

Attention runs one graph at a time (indexed by ``batch``), so nodes only ever attend
within their own shape and each shape gets its own latent array — no cross-shape
leakage. Only the invariant scalar channels feed the readout, so the latent is
rotation/translation invariant; the higher-order (l>0) channels are dropped here.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from e3nn import o3

from src.learning.layers.equivariant.Self_Spatial_layer import EquiLayer
from src.learning.models.encoder_output import EncoderOutput
from src.learning.modules.equivariant.irreps_utils import scalar_features
from src.learning.modules.latent_vae import LatentVAEHead
from src.learning.modules.transformers.perceiver_encoder import (
    PerceiverEncoder,
    PerceiverReducer,
)
from src.learning.modules.transformers.self_attention import SelfAttentionBlock
from src.learning.modules.equivariant.interaction import BipartiteSpatialConvolution
from src.learning.modules.equivariant.transformer import build_equivariant_transformer

class GroupPerceiverEncoder(nn.Module):
    def __init__(
        self,
        irreps_cfg,
        n_latent=16,
        d_shared=32,
        self_attn_heads=4,
        cross_attn_heads=4,
        n_self_layers=1,
        widening_factor=2,
        reduce_stages=None,
        reduce_heads=4,
        vae_mode=None,
        sh_lmax=1,
        supernode_sh_lmax =1,
        interaction_sh_lmax=2,
        n_perceiver_layers = 4,
        perceiver_weight_sharing = True,
        transformer_type="se3",
        transformer_cfg=None,
        area_pool=False,
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose
        self.area_pool = area_pool
        self.perceiver_weight_sharing = perceiver_weight_sharing
        self.n_latent = n_latent
        self.d_shared = d_shared

        in_irreps  = irreps_cfg.get("input_irreps", "1x0e")
        mid_irreps = irreps_cfg.get("intermediate_irreps", "32x0e + 32x1o")
        out_irreps = irreps_cfg.get("output_irreps", "16x0e + 2x1o")
        self.out_irreps = o3.Irreps(out_irreps)

        # Equivariant GNN backbone (same stack shape as GroupEncoder).
        self.layers = nn.ModuleList([
                EquiLayer(in_irreps=in_irreps,
                            target_irreps=out_irreps,
                            spatial_sh_lmax = sh_lmax,
                            interaction_sh_lmax = interaction_sh_lmax,
                            verbose=verbose)
               # EquiLayer(in_irreps=mid_irreps, 
               #         target_irreps=out_irreps,
               #         spatial_sh_lmax = sh_lmax,
               #         interaction_sh_lmax = interaction_sh_lmax,
               #             verbose=verbose)
            ])

        # Number of invariant scalar channels the GNN emits (the transformer width).
        n_scalar = self.out_irreps.count(o3.Irrep("0e"))
        if n_scalar == 0:
            raise ValueError(f"output_irreps '{out_irreps}' has no 0e channels; nothing to attend over.")
        if n_scalar % self_attn_heads != 0:
            raise ValueError(f"n_scalar ({n_scalar}) must be divisible by self_attn_heads ({self_attn_heads}).")
        if d_shared % cross_attn_heads != 0:
            raise ValueError(f"d_shared ({d_shared}) must be divisible by cross_attn_heads ({cross_attn_heads}).")
        self.n_scalar = n_scalar

        self.supernode_conv = BipartiteSpatialConvolution(
                in_irreps=self.out_irreps, target_irreps=self.out_irreps,
                sh_lmax=supernode_sh_lmax, verbose=verbose,
            )

        # Optional equivariant transformer refinement of the pooled features, applied
        # after supernode aggregation and before the scalar channels are split off.
        # Selectable backend ('se3' | 'equiformer'); in-place on out_irreps. Disable with
        # transformer_type=None.
        self.equi_transformer = build_equivariant_transformer(
            transformer_type, self.out_irreps, transformer_cfg, verbose=verbose,
        )

        # Scalar self-attention transformer over the nodes (keeps width n_scalar).
        # Standard pre-norm multi-head self-attention block (torch.nn.MultiheadAttention).
        self.self_attn = nn.ModuleList([
            SelfAttentionBlock(dim=n_scalar, num_heads=self_attn_heads, widening_factor=widening_factor)
            for _ in range(n_self_layers)
        ])

        # Perceiver stack: n_latent learned queries cross-attend to node scalars.
        # PerceiverLayer adds the query residual, so v_channels must equal d_shared.
        self.latents = nn.Parameter(torch.randn(n_latent, d_shared) * 0.02)
        if not self.perceiver_weight_sharing:
            self.perceivers = nn.ModuleList([
            PerceiverEncoder(
                input_dim=n_scalar, latent_dim=d_shared,
                qk_channels=d_shared, v_channels=d_shared,
                num_heads=cross_attn_heads, widening_factor=widening_factor,
            )
            for _ in range(n_perceiver_layers)
            ])
        else:
            self.perceiver = PerceiverEncoder(
                    input_dim=n_scalar, latent_dim=d_shared,
                    qk_channels=d_shared, v_channels=d_shared,
                    num_heads=cross_attn_heads, widening_factor=widening_factor,
                )
            self.perceivers = nn.ModuleList([self.perceiver
                                             for _ in range(n_perceiver_layers)
            ])

        # Optional: iteratively reduce the token count (n_latent -> ... -> reduce_stages[-1]).
        # The reducer has learnable latent queries in each channel.
        self.reducer = None
        if reduce_stages:
            self.reducer = PerceiverReducer(
                d_shared=d_shared, stages=list(reduce_stages),
                num_heads=reduce_heads, widening_factor=widening_factor,
            )

        # Optional: VAE head over whatever token set remains (any token count).
        self.vae = LatentVAEHead(d_shared, mode=vae_mode) if vae_mode else None

        self.mu_bn = nn.BatchNorm1d(d_shared)
        self.mu_ln = nn.LayerNorm(d_shared)

    def forward(self, graph, supergraph):
        x = graph.x
        pos = graph.pos
        edge_index = graph.edge_index
        batch_idx = graph.batch
        # Per-node surface measure for area-weighted aggregation; None unless area_pool is
        # on and the data carries 'area' -> convs fall back to 1/degree, unchanged.
        node_area = getattr(graph, 'area', None) if self.area_pool else None
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

        #  Aggregate into supernodes
        feat_pos = pos
        if supergraph is not None:
            if super_pos is None or super_batch is None or super_edge_index is None:
                raise ValueError(
                    "use_supernodes=True requires super_pos, super_batch and super_edge_index."
                )
            x = self.supernode_conv(x, pos, super_pos, super_edge_index, area_src=node_area)
            batch_idx = super_batch
            feat_pos = super_pos

        # Equivariant transformer refinement over the pooled set (supernodes or nodes),
        # on the full irreps BEFORE the scalar channels are split off.
        if self.equi_transformer is not None:
            x = self.equi_transformer(x, feat_pos, batch_idx)

        # Split off the invariant scalar channels.
        scalars = scalar_features(x, self.out_irreps)          # [N_total, n_scalar]

        # Run the transformer + Perceiver readout once per graph
        #  nodes only attend within their own shape. 
        # Batch dim of 1 keeps the (B, N, C) attention
        # ops happy without padding masks.
        graph_splits = torch.bincount(batch_idx).tolist()
        nodes_list = torch.split(scalars, graph_splits, dim=0)
        # Pad them into a single dense tensor: [B, max_N_b, n_scalar]
        nodes_padded = pad_sequence(nodes_list, batch_first=True)

        # Create a key_padding_mask to prevent nodes from attending to padding
        # Shape: [B, max_N_b] (True where padding exists)
        B, max_N_b, _ = nodes_padded.shape
        mask = torch.arange(max_N_b, device=scalars.device).unsqueeze(0) >= torch.tensor(graph_splits, device=scalars.device).unsqueeze(1)

        # Run your Transformer blocks (pass the key_padding_mask to your attention blocks)
        for block in self.self_attn:
            nodes_padded = block(nodes_padded, key_padding_mask=mask) 

        # Run Perceiver cross-attention
        # Pass the same mask to the Perceiver so queries don't attend to padded nodes
        queries = self.latents.unsqueeze(0).expand(B, -1, -1)
        tokens = self.perceiver(nodes_padded, queries, key_padding_mask=mask)
        
        # Optional iterative reduction
        if self.reducer is not None:
            tokens = self.reducer(tokens)                      # [B, n_final, d_shared]

        # Optional VAE head over the remaining token set.
        if self.vae is not None:
            z, mu, logvar, kl = self.vae(tokens)
            mu = mu.squeeze(1)
            mu = self.mu_bn(mu)
            mu = mu.unsqueeze(1)
            #mu = self.mu_ln(mu)
            return EncoderOutput(latent = z, mu=mu, logvar=logvar,
                                 aux={"kl": kl, "latent_set": tokens})
       
        return EncoderOutput(latent=tokens)
