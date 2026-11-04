"""Small helpers for working with plain ``[N, irreps.dim]`` tensors in PyTorch e3nn.

These replace the e3nn-jax ``IrrepsArray`` conveniences (``slice_by_irreps``,
``.norm()``) that do not exist in e3nn 0.6.0. Features are plain tensors laid out
according to their ``o3.Irreps``; the irreps are tracked by the caller.
"""

import torch
from e3nn import o3


def scalar_features(x, irreps):
    """Gather all invariant scalar (0e) channels of ``x`` -> ``[..., #0e]``."""
    irreps = o3.Irreps(irreps)
    cols = [x[..., s] for (mul, ir), s in zip(irreps, irreps.slices())
            if ir.l == 0 and ir.p == 1]
    if not cols:
        return x[..., :0]
    return torch.cat(cols, dim=-1)


def vector_features(x, irreps, target='1o'):
    """Gather every ``target`` irrep (e.g. ``1o``) as ``[..., total_mul, ir.dim]``."""
    irreps = o3.Irreps(irreps)
    target = o3.Irrep(target)
    blocks = []
    for (mul, ir), s in zip(irreps, irreps.slices()):
        if ir == target:
            blocks.append(x[..., s].reshape(*x.shape[:-1], mul, ir.dim))
    if not blocks:
        return x.new_zeros(*x.shape[:-1], 0, target.dim)
    return torch.cat(blocks, dim=-2)


def expand_per_irrep_gate(gate, irreps):
    """Broadcast a per-irrep gate ``[..., num_irreps]`` to per-channel ``[..., dim]``.

    Each irrep's single gate value is repeated across its ``ir.dim`` components, so
    a vector irrep is scaled by one shared scalar — which keeps the gate equivariant.
    """
    irreps = o3.Irreps(irreps)
    repeats = []
    for mul, ir in irreps:
        repeats += [ir.dim] * mul
    repeats = torch.tensor(repeats, device=gate.device)
    return gate.repeat_interleave(repeats, dim=-1)
