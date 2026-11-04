"""Standard pre-norm multi-head self-attention transformer block.

A plain, unambiguous self-attention encoder block over a token set ``[B, N, dim]``:

    x -> LN -> MultiheadAttention(self) -> +x -> LN -> MLP -> +x

Built on ``torch.nn.MultiheadAttention`` (Q, K, V all from the same input), so there
is no doubt it is self-attention. Width is preserved (input dim == output dim).
"""

import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, widening_factor=2, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * widening_factor),
            nn.GELU(),
            nn.Linear(dim * widening_factor, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        """x: [B, N, dim] -> [B, N, dim]. Optional key_padding_mask: [B, N] (True = pad)."""
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
