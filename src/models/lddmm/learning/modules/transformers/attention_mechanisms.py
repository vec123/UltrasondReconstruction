import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        n_channels,
    ):
        super().__init__()
        self.q = nn.Linear(input_dim, n_channels)
        self.k = nn.Linear(input_dim, n_channels)
        self.v = nn.Linear(input_dim, n_channels)

    def forward(self, input):
        # (N, input_dim) . (input_dim, qk_channels) -> (N, qk_channels)
        query = self.q(input)
        # (N, input_dim) . (input_dim, qk_channels) -> (N, qk_channels)
        key = self.k(input)
        # (N, input_dim) . (input_dim, v_channels) -> (N, v_channels)
        value = self.v(input)

        scale = 1.0 / math.sqrt(query.size(-1))
        # (N, qk_channels) . (qk_channels, N) -> (N, N)
        scores = torch.bmm(query, key.transpose(-1, -2)) * scale
        print(f"Attention score shape: {scores.shape}")
        weights = F.softmax(scores, dim=-1)
        # (N, N) . (N, v_channels) -> (N, v_channels)
        return torch.bmm(weights, value)
    


class CrossAttention(nn.Module):
    def __init__(
        self,
        input_kv_dim,
        input_q_dim,
        qk_channels,
        v_channels,
    ):
        super().__init__()
        self.q = nn.Linear(input_q_dim, qk_channels)
        self.k = nn.Linear(input_kv_dim, qk_channels)
        self.v = nn.Linear(input_kv_dim, v_channels)

    def forward(self, input_kv, input_q):
        # (M, input_q_dim) . (input_q_dim, qk_channels) -> (N, qk_channels)
        query = self.q(input_q)
        # (N, input_kv_dim) . (input_kv_dim, qk_channels) -> (N, qk_channels)
        key = self.k(input_kv)
        # (N, input_kv_dim) . (input_kv_dim, v_channels) -> (N, v_channels)
        value = self.v(input_kv)

        scale = 1.0 / math.sqrt(query.size(-1))
        # (M, qk_channels) . (qk_channels, N) -> (M, N)
        scores = torch.bmm(query, key.transpose(-1, -2)) * scale
        print(f"Attention score shape: {scores.shape}")
        weights = F.softmax(scores, dim=-1)
        # (M, N) . (N, v_channels) -> (M, v_channels)
        return torch.bmm(weights, value)