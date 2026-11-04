from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class EncoderOutput:
    """Standard container for encoder outputs, decoupling encoders from the trainer.

    All fields are optional so a range of encoders satisfy one contract:
      * deterministic encoder -> set ``latent``
      * VAE encoder           -> set ``mu`` / ``logvar`` (the trainer reparameterizes)
      * equivariant encoder   -> additionally set ``rotation`` / ``translation`` (pose)

    ``aux`` carries anything extra (e.g. the raw equivariant vectors) without
    widening the contract.
    """

    latent: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    logvar: Optional[torch.Tensor] = None
    rotation: Optional[torch.Tensor] = None
    translation: Optional[torch.Tensor] = None
    aux: dict = field(default_factory=dict)

    def sample(self, deterministic=False, generator=None):
        """Return the latent to feed the decoder — the single consumption point.

        Precedence (works for every encoder without the caller knowing which it is):
          * ``mu`` is set (probabilistic / VAE encoder):
              - ``deterministic`` (or no ``logvar``)  -> return ``mu`` (the mean)
              - otherwise                             -> reparameterized sample
                                                         ``mu + eps * exp(0.5*logvar)``
          * only ``latent`` is set (deterministic encoder) -> return ``latent``.

        So the same call handles "sampled from mu/var" and "deterministic mu", and
        the GroupEncoder (always mu/logvar) and the GroupPerceiverEncoder (mu/logvar
        in VAE mode, or a plain latent set otherwise) go through identical code.
        """
        if self.mu is None:
            return self.latent
        if deterministic or self.logvar is None:
            return self.mu
        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn(std.shape, dtype=std.dtype, device=std.device, generator=generator)
        return self.mu + eps * std

    def kl(self):
        """Gaussian KL to N(0, I), or ``None`` when the encoder is not probabilistic.

        Prefers a precomputed ``aux['kl']`` (e.g. from ``LatentVAEHead``); otherwise
        derives it from ``mu``/``logvar``. Summed over the last dim, mean over the rest.
        """
        if "kl" in self.aux:
            return self.aux["kl"]
        if self.mu is None or self.logvar is None:
            return None
        return -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp(), dim=-1).mean()
