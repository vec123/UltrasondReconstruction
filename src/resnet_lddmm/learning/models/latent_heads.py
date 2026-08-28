"""LatentHead strategies (INSTRUCTIONS.md T9).

Strategy: 
pooled invariant scalars -> the latent an ``EncoderOutput`` carries.
``GaussianLatentHead`` is  VAE behavior (mu/logvar); 
``DeterministicLatentHead`` is the auto-encoder sibling (a plain latent). 
Both emit ``[B, latent_dim]``, 
the decoder contract is unaffected and ``latent_mode`` is a simple switch --

The base class owns the readout ("mean" | "attention"), which is ORTHOGONAL to the
distribution: readout decides HOW per-node scalars collapse to one token per shape (i.e. graph),
The subclass decides WHAT distribution that token parameterizes. 
Keeping the readout here is what stops 2 heads x 2 readouts becoming four 
copies of the same pooling.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool

from src.learning.models.encoder_output import EncoderOutput
from src.learning.modules.transformers.perceiver_encoder import PerceiverReducer


class LatentHead(nn.Module, ABC):
    """Strategy base: (scalars, weights, batch, num_graphs) -> EncoderOutput.

    ``scalars``   [n_pool, latent_dim]  invariant per-token features
    ``weights``   [n_pool, 1]           per-shape attention weights, summing to 1
                                        within each shape (computed by the encoder,
                                        which also shares them with its pose head)
    ``batch``     [n_pool]              which shape each token belongs to
    ``num_graphs``                      number of shapes, passed explicitly so a
                                        non-contiguous ``batch`` can't drop a row
    """

    def __init__(self, latent_dim: int, readout: str = "mean", readout_heads: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.readout = readout

        # Attention readout: one learned query cross-attends to a shape's scalars,
        # collapsing them to a single [latent_dim] token (PerceiverReducer, stages=[1]).
        # The mean-pool path is kept for ablation (readout="mean").
        self.readout_pool = None
        if readout == "attention":
            self.readout_pool = PerceiverReducer(
                d_shared=latent_dim, stages=[1],
                num_heads=readout_heads, self_attend=False,
            )
        elif readout != "mean":
            raise ValueError(f"readout must be 'attention' or 'mean', got {readout!r}")

    def _attention_pooled(self, scalars, batch, num_graphs):
        """Collapse each shape's tokens to one, per-graph. 
        Returns [B, latent_dim], or None on the mean path.

        Computed ONCE per forward and shared by every net the subclass reduces,
        because this is a python loop over shapes -- calling it per-net (once for
        mu, again for var) would double an already expensive path.
        """
        if self.readout != "attention":
            return None
        pooled = []
        for b in range(num_graphs):
            toks = scalars[batch == b].unsqueeze(0)         # [1, n_b, latent_dim]
            pooled.append(self.readout_pool(toks))          # [1, 1, latent_dim]
        return torch.cat(pooled, dim=0).squeeze(1)          # [B, latent_dim]

    def _reduce(self, net, scalars, weights, batch, num_graphs, pooled):
        """Apply ``net`` and reduce to [B, latent_dim].

        The ORDER differs per readout, and that difference is load-bearing, not
        incidental -- it is exactly what the pre-T9 GroupEncoder did:
          * attention -> pool first, then apply the net to the pooled token
          * mean      -> apply the net per-token, then weighted-sum
        For a linear net the two orders agree (weights sum to 1 per shape), but
        ``var_net`` ends in a Softplus, so for it they genuinely differ. Preserved
        as-is: T9 step 3 requires this move to be behavior-preserving.
        """
        if self.readout == "attention":
            return net(pooled)
        # Weights already sum to 1 per shape -> global_add_pool is the single
        # normalization (a global_mean_pool here would divide twice and crush
        # the per-shape latent toward 0).
        return global_add_pool(weights * net(scalars), batch, size=num_graphs)

    @abstractmethod
    def forward(self, scalars, weights, batch, num_graphs):
        """-> EncoderOutput carrying the latent fields only (no pose; the encoder
        fills rotation/translation in)."""
        ...


class GaussianLatentHead(LatentHead):
    """VAE head -- today's behavior. -> EncoderOutput(mu=..., logvar=...)"""

    def __init__(self, latent_dim: int, readout: str = "mean", readout_heads: int = 1):
        super().__init__(latent_dim, readout, readout_heads)
        # Construction order (readout_pool -> mu_net -> var_net) mirrors the
        # pre-T9 GroupEncoder.__init__ exactly. nn.Linear draws from the global
        # RNG at construction, so reordering these would change every seeded
        # initialization and break the T2 characterization baseline.
        self.mu_net = nn.Linear(latent_dim, latent_dim)
        self.var_net = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Softplus())

    def forward(self, scalars, weights, batch, num_graphs):
        pooled = self._attention_pooled(scalars, batch, num_graphs)
        mu = self._reduce(self.mu_net, scalars, weights, batch, num_graphs, pooled)
        var = self._reduce(self.var_net, scalars, weights, batch, num_graphs, pooled)
        # var_net ends in Softplus, so var > 0; the epsilon guards log(0) when it underflows.
        logvar = torch.log(var + 1e-8)
        return EncoderOutput(mu=mu, logvar=logvar)


class DeterministicLatentHead(LatentHead):
    """Auto-encoder head. -> EncoderOutput(latent=z), mu=None, logvar=None.

    Same [B, latent_dim] output shape as the Gaussian head, so the decoder
    contract is unaffected and latent_mode becomes a pure ablation switch.

    Regularization is not this head's business: with ``mu`` unset, ``kl()``
    returns None and the composer skips a ``kl`` term (T8), so the Frobenius
    term takes over purely by loss config (``frobenius_latent_loss``, losses.py).
    """

    def __init__(self, latent_dim: int, readout: str = "mean", readout_heads: int = 1):
        super().__init__(latent_dim, readout, readout_heads)
        # The Gaussian head's mu_net, minus the var_net and the sampling.
        self.latent_net = nn.Linear(latent_dim, latent_dim)

    def forward(self, scalars, weights, batch, num_graphs):
        pooled = self._attention_pooled(scalars, batch, num_graphs)
        z = self._reduce(self.latent_net, scalars, weights, batch, num_graphs, pooled)
        return EncoderOutput(latent=z)