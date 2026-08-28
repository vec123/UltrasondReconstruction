"""VAE head over a latent token set of arbitrary size.

Given a latent set ``[B, K, d]`` (K = number of latent tokens, d = per-token dim),
turn it into a sampled latent plus ``(mu, logvar, kl)`` in the usual VAE style. The
head works for ANY K, so it can sit after the Perceiver readout regardless of how
many latents there are (1, 2, or many).

Modes (how mu/logvar are derived from the tokens):
  * ``per_token``     - each token -> its own mu (d) and logvar (d) via linears;
                        z ~ N(mu, exp logvar)              -> z, mu, logvar : [B, K, d]
  * ``split_channel`` - split each token's d channels in half -> mu, logvar (d//2);
                        z                                   -> [B, K, d//2]
  * ``two_token``     - interpret exactly K==2 tokens as (mu, logvar);
                        z                                   -> [B, 1, d]
"""

import torch
import torch.nn as nn

from src.learning.losses.losses import kl_divergence_loss


class LatentVAEHead(nn.Module):
    def __init__(self, d, mode="per_token"):
        super().__init__()
        self.d = d
        self.mode = mode
        if mode == "per_token":
            self.to_mu = nn.Linear(d, d)
            self.to_logvar = nn.Linear(d, d)
        elif mode == "split_channel":
            if d % 2 != 0:
                raise ValueError(f"split_channel mode needs even d, got d={d}")
        elif mode == "two_token":
            pass
        else:
            raise ValueError(f"unknown vae mode {mode!r} (per_token | split_channel | two_token)")

    def _mu_logvar(self, latents):
        if self.mode == "per_token":
            return self.to_mu(latents), self.to_logvar(latents)
        if self.mode == "split_channel":
            mu, logvar = latents.chunk(2, dim=-1)
            return mu, logvar
        # two_token: the two tokens ARE (mu, logvar)
        if latents.shape[1] != 2:
            raise ValueError(f"two_token mode needs exactly K==2 latents, got K={latents.shape[1]}")
        return latents[:, 0:1, :], latents[:, 1:2, :]

    def forward(self, latents):
        """latents: [B, K, d] -> (z, mu, logvar, kl)."""
        mu, logvar = self._mu_logvar(latents)
        # Clamp logvar (same defensive floor as the trainer's finite-loss guard) so an
        # extreme token can't blow std up to inf and NaN the reparameterization.
        logvar = logvar.clamp(-10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        kl = kl_divergence_loss(mu, logvar)   # sum over last dim, mean over B (and K)
        return z, mu, logvar, kl
