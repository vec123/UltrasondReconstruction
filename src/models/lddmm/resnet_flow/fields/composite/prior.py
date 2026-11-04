"""A frozen prior field plus a trainable residual, each conditioned on its own part of the code.

    v(x, [z | w]) = v_prior(x, z) + v_net(x, w)

``v_prior`` is the field of a finished run (e.g. a cohort flow trained on BabySSM
samples), loaded once and never trained again: it holds what that cohort taught
about faces. ``v_net`` is a new network that learns what the prior cannot do
(e.g. how fetal faces differ from baby faces). The per-shape code is one vector
``[z | w]``: the first ``prior_n_z`` entries go to the prior, the rest to the
network, so code tables, code fits and apply need no second code source.

The price follows ``fields/modes.py``: ``split`` returns the prior's part as the
priced part "prior", so the kinetic term charges only the network
(``Trajectory.free_velocities``) and ``losses.kinetic.PartKineticLoss`` charges
the prior at ``discount * kinetic_weight``. A low discount makes a motion the
prior knows cheap, and any other motion costs what it always did.

The prior's weights are buffers of this module in all but name: they are frozen
(``requires_grad`` off) but kept in the state dict, so every checkpoint carries
the prior it was trained with and apply needs no second checkpoint.
"""

from typing import Dict, Optional, Tuple

from torch import Tensor

from src.models.lddmm.resnet_flow.fields.base import VelocityField, split_velocity


class PriorResidualField(VelocityField):
    """``prior(x, z) + network(x, w)``, the prior frozen, the code split as ``[z | w]``."""

    def __init__(self, prior: VelocityField, network: VelocityField, prior_n_z: int):
        """Initialize the field.

        Args:
            prior: the field of a finished run; frozen here, its weights loaded later
                (runner._load_prior_field) or restored from a checkpoint
            network: the trainable residual field
            prior_n_z: how many leading code entries the prior reads; 0 for an
                unconditioned prior
        """
        super().__init__()
        if prior_n_z < 0:
            raise ValueError(f"prior_n_z must be >= 0, got {prior_n_z}")
        self.prior = prior
        self.network = network
        self.prior_n_z = int(prior_n_z)
        for param in self.prior.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True):
        """Train the network only: the prior stays in eval mode."""
        super().train(mode)
        self.prior.eval()
        return self

    def split_code(self, code: Optional[Tensor]) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """[B, n_z] -> (z [B, prior_n_z] or None, w [B, n_z - prior_n_z] or None).

        Raises:
            ValueError: the prior needs a code and there is none, or the code is
                shorter than the prior's part.
        """
        if code is None:
            if self.prior_n_z > 0:
                raise ValueError(f"the prior field reads a code of {self.prior_n_z} entries, "
                                 f"but the flow got no code")
            return None, None
        if code.shape[-1] < self.prior_n_z:
            raise ValueError(f"code of {code.shape[-1]} entries is shorter than the prior's "
                             f"{self.prior_n_z}")
        z = code[..., :self.prior_n_z] if self.prior_n_z > 0 else None
        w = code[..., self.prior_n_z:] if code.shape[-1] > self.prior_n_z else None
        return z, w

    def split(self, x: Tensor, step: Optional[int] = None,
              code: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        """(total velocity, priced parts): the prior's part as "prior", plus the network's own.

        Integrators record the parts so the kinetic cost can charge each apart
        from the rest, as for a ModalField.
        """
        z, w = self.split_code(code)
        prior = self.prior(x, step=step, code=z)
        velocity, parts = split_velocity(self.network, x, step=step, code=w)
        return prior + velocity, {"prior": prior, **parts}

    def forward(self, x: Tensor, step: Optional[int] = None, code: Optional[Tensor] = None) -> Tensor:
        return self.split(x, step=step, code=code)[0]
