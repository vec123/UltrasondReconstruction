"""How much of a flow each part of a ModalField did: the network, and each designed mode - the maths.

Two complementary answers, both on a trained flow, both kept step by step:

    decompose(flow, points)        the ACTUAL path, split exactly: every Euler step's
                                   velocity is v_net + sum_m h_m phi_m, so each part's
                                   share of the displacement, up to any step, is the
                                   sum of what it contributed so far - shares that add
                                   up to the total, to rounding.
    flow_with(flow, points, keep)  COUNTERFACTUAL flows: integrate with only some parts
                                   switched on (e.g. the network alone, or one mode
                                   alone) - what each would have done by itself. These
                                   do not add up to the total (the flow is nonlinear),
                                   but they are the shapes to look at.

Points are processed in chunks: a ModalField over an MLP is pointwise, so a
point's path does not depend on which chunk it is in.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from torch import Tensor

from src.resnet_lddmm.resnet_flow.fields.modes import ModalField

NETWORK = "network"


@dataclass
class PartsTrajectory:
    """A flow's path, step by step, with the velocity of each part that was switched on.

    ``points[k]`` is the shape after k Euler steps (k = 0 .. K) and
    ``velocities[part][k]`` that part's velocity at ``points[k]`` - the one that
    moved it to ``points[k + 1]``. All in the flow's frame.
    """

    points: Tensor                     # [K + 1, N, 3]
    velocities: Dict[str, Tensor]      # part -> [K, N, 3]
    dt: float

    @property
    def start(self) -> Tensor:
        return self.points[0]

    @property
    def end(self) -> Tensor:
        return self.points[-1]

    @property
    def total(self) -> Tensor:
        return self.end - self.start

    @property
    def steps(self) -> int:
        return len(self.points) - 1

    def displacement(self, part: str, upto: Optional[int] = None) -> Tensor:
        """[N, 3] how far ``part`` moved each point over the first ``upto`` steps (default: all)."""
        upto = self.steps if upto is None else upto
        velocities = self.velocities[part][:upto]
        return self.dt * velocities.sum(dim=0) if upto else torch.zeros_like(self.start)

    @property
    def parts(self) -> Dict[str, Tensor]:
        """Each part's share of the whole displacement, [N, 3]."""
        return {part: self.displacement(part) for part in self.velocities}


def modal_field(flow) -> ModalField:
    """``flow.field`` as a ModalField.

    Raises:
        ValueError: the flow was built without field.modes.
    """
    field = flow.field
    if not isinstance(field, ModalField):
        raise ValueError("this flow has no designed modes (field.modes was not configured)")
    return field


@torch.no_grad()
def _integrate(flow, points: Tensor, keep: List[str], code: Optional[Tensor], chunk: int) -> PartsTrajectory:
    """ForwardEuler with only the parts in ``keep`` switched on, every step and part recorded."""
    field = modal_field(flow)
    steps = flow.num_steps
    dt = 1.0 / steps
    modes = [(m, name) for m, name in enumerate(field.names) if name in keep]
    chunks_points, chunks_velocities = [], {part: [] for part in keep}
    for part_points in torch.split(points, chunk):
        q = part_points[None].to(field.amplitude.dtype)
        path = [q[0]]
        velocities = {part: [] for part in keep}
        for k in range(steps):
            v = torch.zeros_like(q)
            if NETWORK in keep:
                network = field.network(q, step=k, code=code)
                velocities[NETWORK].append(network[0])
                v = v + network
            if modes:
                basis = field.basis(q)                                    # [1, n, M, 3]
                for m, name in modes:
                    modal = field.amplitude[k, m].to(q.dtype) * basis[:, :, m]
                    velocities[name].append(modal[0])
                    v = v + modal
            q = q + dt * v
            path.append(q[0])
        chunks_points.append(torch.stack(path))
        for part in keep:
            chunks_velocities[part].append(torch.stack(velocities[part]))
    return PartsTrajectory(torch.cat(chunks_points, dim=1),
                           {part: torch.cat(v, dim=1) for part, v in chunks_velocities.items()}, dt)


def decompose(flow, points: Tensor, code: Optional[Tensor] = None, chunk: int = 4096) -> PartsTrajectory:
    """The trained flow of ``points`` [N, 3], every part switched on and recorded separately."""
    field = modal_field(flow)
    return _integrate(flow, points, [NETWORK] + list(field.names), code, chunk)


def flow_with(flow, points: Tensor, keep: Iterable[str], code: Optional[Tensor] = None,
              chunk: int = 4096) -> PartsTrajectory:
    """The flow of ``points`` with only the parts in ``keep`` ("network" and/or mode names) on.

    Raises:
        ValueError: ``keep`` names a part the field does not have.
    """
    field = modal_field(flow)
    keep = list(dict.fromkeys(keep))
    unknown = set(keep) - {NETWORK, *field.names}
    if unknown:
        raise ValueError(f"no part(s) {sorted(unknown)}; have {[NETWORK] + field.names}")
    return _integrate(flow, points, keep, code, chunk)
