import torch
from torch import nn
import torch.nn.functional as F

from src.learning.modules.transformers.layers import PerceiverLayer

class PerceiverDecoder(nn.Module):
    def __init__(
        self,
        num_output_channels,
        latent_dim,
        query_dim,
        qk_channels,
        v_channels,
        num_heads,
        widening_factor,
    ):
        super().__init__()

        self.decoder = PerceiverLayer(
            input_kv_dim=latent_dim,
            input_q_dim=query_dim,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            widening_factor=widening_factor,
            is_cross_attention=True,
        )

        self.dense = nn.Linear(query_dim, num_output_channels)

    def forward(self, latent, query):
        attn_output = self.decoder(latent, query)
        logit = self.dense(attn_output)
        return logit