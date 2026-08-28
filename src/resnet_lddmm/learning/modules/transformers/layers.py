import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from src.learning.modules.transformers.heads import MultiHeadAttention



class MLP(nn.Module):
    def __init__(self, input_size, widening_factor, dropout_prob=0.5):
        super().__init__()
        self.dense1 = nn.Linear(input_size, input_size * widening_factor)
        self.dense2 = nn.Linear(input_size * widening_factor, input_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dense2(x)
        return self.dropout(x)

class PerceiverLayer(nn.Module):
    def __init__(
        self,
        input_kv_dim,
        input_q_dim,
        qk_channels,
        v_channels,
        num_heads,
        widening_factor,
        is_cross_attention,
    ):
        super().__init__()

        if input_kv_dim is None:
            input_kv_dim = input_q_dim

        self.layer_norm_1 = nn.LayerNorm(input_kv_dim)
        self.layer_norm_2 = nn.LayerNorm(input_q_dim)

        self.attention = MultiHeadAttention(
            is_cross_attention=is_cross_attention,
            input_kv_dim=input_kv_dim,
            input_q_dim=input_q_dim,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
        )

        self.mlp = MLP(v_channels, widening_factor=widening_factor)

    def forward(self, input_kv, input_q, key_padding_mask=None):
        input_kv_norm = self.layer_norm_1(input_kv)
        input_q_norm = self.layer_norm_2(input_q)
        x_qkv = self.attention(
            input_kv_norm, 
            input_q_norm, 
            key_padding_mask=key_padding_mask
        )
        x_qkv = x_qkv + input_q
        x_qkv = x_qkv + self.mlp(input_q_norm)
        return x_qkv