"""Penalties for fitting a trained flow with part of it free (train.freeze_flow_except).

A flow restored through ``init`` and fitted to new targets with, say, its LayerNorm
affines free can follow the targets where its code alone cannot. These terms keep it
near the flow it was trained as:

  * ``parameter_anchor``      sum (p - p0)^2 over the free flow parameters whose name
                              holds ``pattern``; p0 their values at the first step.
  * ``field_change_kinetic``  1/2 int mean_i ||v(x_i(t)) - v0(x_i(t))||^2 dt along the
                              forward trajectory, v0 the field as it was at the first
                              step: Trajectory.kinetic_energy's reduction, on the change.

The trajectory covers every template point the step flowed, seen by a target or not,
so ``field_change_kinetic`` also charges a change where no target pulls - which is
what holds the regions no scan shows at what the trained flow makes of the code.

Both take their reference at the first call, i.e. after init has restored the
checkpoint; a resumed run takes it again from the resumed weights.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.models.lddmm.resnet_flow.losses.terms import LossContext


class ParameterAnchor(nn.Module):
    """``sum (p - p0)^2`` over the trainable flow parameters whose name holds ``pattern``."""

    def __init__(self, pattern: str = "norm"):
        super().__init__()
        self.pattern = pattern
        self.anchor = None

    def _free_parameters(self, flow):
        return [p for name, p in flow.named_parameters() if p.requires_grad and self.pattern in name]

    def forward(self, ctx: LossContext) -> Optional[Tensor]:
        params = self._free_parameters(ctx.flow)
        if not params:
            raise ValueError(f"parameter_anchor: no trainable flow parameter has {self.pattern!r} "
                             f"in its name (see train.freeze_flow_except)")
        if self.anchor is None:
            self.anchor = [p.detach().clone() for p in params]
            print(f"[parameter_anchor] {len(params)} tensors ({sum(p.numel() for p in params)} values) "
                  f"matching {self.pattern!r}, anchored at their values now")
        return sum(((p - p0) ** 2).sum() for p, p0 in zip(params, self.anchor))


class FieldChangeKinetic(nn.Module):
    """½ ∫ mean_i ||v(x_i(t)) - v0(x_i(t))||² dt, v0 a frozen copy of the field at the first step."""

    def __init__(self):
        super().__init__()
        self.reference = None

    def forward(self, ctx: LossContext) -> Optional[Tensor]:
        if ctx.fwd_traj is None:
            return None
        field = ctx.flow.field
        if self.reference is None:
            # A list keeps the copy out of this module's parameters and state_dict.
            reference = copy.deepcopy(field)
            for param in reference.parameters():
                param.requires_grad_(False)
            self.reference = [reference]
        reference = self.reference[0]
        traj = ctx.fwd_traj
        change = 0.0
        for k in range(traj.velocities.shape[0]):
            with torch.no_grad():
                v0 = reference(traj.points[k], step=k, code=ctx.code)
            change = change + (traj.velocities[k] - v0).pow(2).sum(-1).mean()
        return 0.5 * change * traj.dt
