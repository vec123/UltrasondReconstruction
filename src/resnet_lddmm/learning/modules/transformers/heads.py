import math

import torch
from torch import nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(
        self,
        is_cross_attention,
        input_kv_dim,
        input_q_dim,
        qk_channels_per_head,
        v_channels_per_head,
        attention_prob_dropout_prob=0.1,
    ):
        super().__init__()
        self.is_cross_attention = is_cross_attention
        if not is_cross_attention:
            input_kv_dim = input_q_dim

        self.q = nn.Linear(input_q_dim, qk_channels_per_head)
        self.k = nn.Linear(input_kv_dim, qk_channels_per_head)
        self.v = nn.Linear(input_kv_dim, v_channels_per_head)
        self.dropout = nn.Dropout(attention_prob_dropout_prob)

    def forward(self, input_kv, input_q, key_padding_mask=None):
        query = self.q(input_q)

        if self.is_cross_attention and (input_kv is not None):
            key = self.k(input_kv)
            value = self.v(input_kv)
        else:
            key = self.k(input_q)
            value = self.v(input_q)

        scale = 1.0 / math.sqrt(query.size(-1))
        scores = torch.bmm(query, key.transpose(-1, -2)) * scale

        # --- Apply Key Padding Mask ---
        if key_padding_mask is not None:
            # key_padding_mask shape: [B, N_kv]
            # Unsqueeze to [B, 1, N_kv] so it correctly broadcasts over N_q dimension
            mask = key_padding_mask.unsqueeze(1)
            # Mask out padded positions by filling them with a very large negative float
            scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return torch.bmm(weights, value)
    
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        is_cross_attention,
        input_kv_dim,
        input_q_dim,
        qk_channels,
        v_channels,
        num_heads,
    ):
        super().__init__()

        qk_channels_per_head = qk_channels // num_heads
        v_channels_per_head = v_channels // num_heads

        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    is_cross_attention,
                    input_kv_dim,
                    input_q_dim,
                    qk_channels_per_head,
                    v_channels_per_head,
                )
                for _ in range(num_heads)
            ]
        )

        self.linear = nn.Linear(v_channels, v_channels)

    def forward(self, input, latent_embedding, key_padding_mask = None):
        x = torch.cat([h(input, latent_embedding, key_padding_mask) for h in self.heads], dim=-1)
        return self.linear(x)