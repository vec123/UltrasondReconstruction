"""A shared flow from the template to the atlas, then a per-member flow from the atlas.

    steps 0 .. K_s - 1:   v = v_shared(x, t)          no code: the same for every member
    steps K_s .. K - 1:   v = v_member(x, t, z)       the member's code

Every member's path goes template -> atlas -> member, so the atlas psi(template) is one
learned shape that all members share (the hypertemplate model, Ma, Miller, Trouve &
Younes 2008, "Bayesian template estimation in computational anatomy": a fixed start
shape, the hypertemplate, is deformed by one shared flow into the
atlas, and the atlas by one flow per subject into each subject). When the member field
is zero at the zero code (an AtlasField, fields/composite/atlas.py), the zero code
gives the atlas exactly.

The shared velocity is returned as the priced part "shared" (fields/base.py
split_velocity), so the kinetic term charges it at its own price
(losses.kinetic.PartKineticLoss, field.shared.discount): at discount * kinetic_weight,
where discount 1/M (``once``) counts the one shared flow once for a cohort of M
members, as the hypertemplate model does, instead of once per member.

The two fields have their own blocks: the shared field K_s, the member field K - K_s.
Each is called with its own step index, so time-varying fields of either kind work.
"""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from src.models.lddmm.resnet_flow.fields.base import VelocityField, split_velocity


class SharedAtlasField(VelocityField):
    """``shared`` for the first ``shared_steps`` Euler steps, then ``network`` with the code."""

    def __init__(self, shared: VelocityField, network: VelocityField, shared_steps: int):
        """Initialize the field.

        Args:
            shared: the code-free field of the shared flow, with ``shared_steps`` blocks.
            network: the per-member field (normally an AtlasField), with K - shared_steps blocks.
            shared_steps: K_s, the number of Euler steps of the shared flow, >= 1.
        """
        super().__init__()
        if shared_steps < 1:
            raise ValueError(f"the shared flow needs at least one step, got {shared_steps}")
        self.shared = shared
        self.network = network
        self.shared_steps = int(shared_steps)

    def split(self, x: Tensor, step: Optional[int] = None,
              code: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        """(velocity, priced parts): the shared velocity as "shared", zero in the member steps.

        The integrators stack the parts over the steps, so every step returns the
        same names (integrators._trajectory).

        Raises:
            ValueError: no step (the two fields are told apart by it), or the member
                field has priced parts of its own (not supported here).
        """
        if step is None:
            raise ValueError("a SharedAtlasField needs the Euler step: it selects the shared "
                             "or the member field")
        if step < self.shared_steps:
            velocity = self.shared(x, step=step, code=None)
            return velocity, {"shared": velocity}
        velocity, parts = split_velocity(self.network, x, step=step - self.shared_steps, code=code)
        if parts:
            raise ValueError(f"the member field of a SharedAtlasField has priced parts "
                             f"{sorted(parts)}; only the shared part is supported")
        return velocity, {"shared": torch.zeros_like(velocity)}

    def forward(self, x: Tensor, step: Optional[int] = None, code: Optional[Tensor] = None) -> Tensor:
        return self.split(x, step=step, code=code)[0]
