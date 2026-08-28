import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from src.learning.modules.transformers.layers import PerceiverLayer
from src.learning.modules.transformers.self_attention import SelfAttentionBlock


class PerceiverEncoder(nn.Module):
    """Perceiver cross-attention: latent queries attend to an input set.

    The queries can be supplied per call (pass ``latent_embeddings`` to ``forward``)
    or owned by the module: pass ``num_latents`` to allocate a learnable
    ``[num_latents, latent_dim]`` query array, then call ``forward(input)`` with no
    queries. Owning them turns a fixed-size readout/reduction into a single module.
    """
    def __init__(
        self,
        input_dim,
        latent_dim,
        qk_channels,
        v_channels,
        num_heads,
        widening_factor,
        num_latents=None,
    ):
        super().__init__()

        self.latents = None
        if num_latents is not None:
            self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)

        self.encoder = PerceiverLayer(
            input_kv_dim=input_dim,
            input_q_dim=latent_dim,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            widening_factor=widening_factor,
            is_cross_attention=True,
        )

    def forward(self, input, latent_embeddings=None, key_padding_mask = None):
        if latent_embeddings is None:
            if self.latents is None:
                raise ValueError(
                    "PerceiverEncoder has no owned latents; pass latent_embeddings "
                    "to forward(), or construct it with num_latents."
                )
            latent_embeddings = self.latents.unsqueeze(0).expand(input.shape[0], -1, -1)
        out = self.encoder(input_kv=input, input_q=latent_embeddings, key_padding_mask= key_padding_mask)
        return out
    


class Perceiver(nn.Module):
    def __init__(self, latent_dim, qk_channels, v_channels, num_heads, widening_factor):
        super().__init__()

        self.layer = PerceiverLayer(
            input_kv_dim=None,
            input_q_dim=latent_dim,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            widening_factor=widening_factor,
            is_cross_attention=False,
        )

    def forward(self, latent):
        return self.layer(input_kv=latent, input_q=latent)


class PerceiverReducer(nn.Module):
    """Iteratively reduce a latent token set to fewer tokens.

    Each stage is a ``PerceiverEncoder`` that owns its learnable query array
    (``num_latents`` = the stage's target token count) and cross-attends to the
    previous stage's tokens, plus an optional latent self-attention. So the reducer
    is just a stack of Perceiver readouts with the queries held inside each stage.
    ``stages=[8, 4, 2]`` turns ``[B, K, d]`` into ``[B, 8, d] -> [B, 4, d] -> [B, 2, d]``;
    all tokens share width ``d_shared``.
    """
    def __init__(self, d_shared, stages, num_heads=4, widening_factor=2, self_attend=True):
        super().__init__()
        if d_shared % num_heads != 0:
            raise ValueError(f"d_shared ({d_shared}) must be divisible by num_heads ({num_heads}).")

        self.cross = nn.ModuleList([
            PerceiverEncoder(
                input_dim=d_shared, latent_dim=d_shared,
                qk_channels=d_shared, v_channels=d_shared,
                num_heads=num_heads, widening_factor=widening_factor,
                num_latents=n,
            )
            for n in stages
        ])
        self.self_attend = self_attend
        self.self_blocks = nn.ModuleList([
            SelfAttentionBlock(dim=d_shared, num_heads=num_heads, widening_factor=widening_factor)
            for _ in stages
        ]) if self_attend else None

    def forward(self, tokens, key_padding_mask = None):
        """tokens: [B, K, d] -> [B, stages[-1], d]."""
        for i, cross in enumerate(self.cross):
            tokens = cross(tokens, key_padding_mask = key_padding_mask)                    # owned query -> [B, n_i, d]
            if self.self_attend:
                tokens = self.self_blocks[i](tokens)
        return tokens