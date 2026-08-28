"""Unbalanced entropic optimal transport as a correspondence method.

Nearest-neighbour matching answers "which single target point is this model
vertex closest to", and every model vertex gets an answer whether or not it
deserves one. That is the wrong question for a partial scan: the ultrasound
covers part of the face, the model covers all of it, and there is no reason for
the two point sets to be in bijection - or for either to be fully explained by
the other.

Optimal transport asks instead how to move a unit of mass from one cloud to the
other at least total cost. The *unbalanced* variant additionally lets mass be
destroyed rather than transported, at a price set by ``reach``: a model vertex
with nothing plausible nearby simply keeps its mass and drops out of the
matching, and so does an unexplained target point. Both sides may be partial,
which is exactly the situation here.

The solution is a soft coupling - one model vertex may share mass across several
target points - so this emits weighted correspondences rather than a hard
pairing. The weights are the transport masses, which the data term and the
closed-form solver already honour, so nothing downstream needs to know that the
matching came from OT: minimising ``sum_e w_e ||x_i - y_j||^2`` over the emitted
edges *is* minimising the transport cost.

Implementation notes:

* The coupling is restricted to a k-nearest-neighbour support, taken from both
  directions and unioned. A dense 31027 x 11034 coupling is 3.4e8 entries; the
  entropic kernel decays as exp(-C/eps), so pairs beyond the nearest handful
  carry no meaningful mass and the sparse support costs nothing in accuracy.
* Scaling iterations run in the log domain, which is what keeps small ``blur``
  values stable.
* ``blur`` and ``reach`` are distances, in model units, comparable to
  ``dist_thresh``; internally they are squared, since the cost is squared
  Euclidean distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.nicp.cloud import PointCloud
from src.nicp.correspondence import (
    Correspondences,
    CorrespondenceStrategy,
    register_correspondence,
)
from src.nicp.neighbours import nearest
from src.nicp.shape import resolve_device


def _segment_logsumexp(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """log sum exp of ``values`` grouped by ``index``, over ``size`` groups.

    Empty groups come back as -inf, i.e. no mass, which is a valid state here:
    a target point outside the model's reach takes part in no edge.
    """
    peak = torch.full((size,), float("-inf"), dtype=values.dtype, device=values.device)
    peak = peak.scatter_reduce(0, index, values, reduce="amax", include_self=False)

    # only groups that actually occur in `index` are read back, so the -inf
    # entries never enter the exponent
    total = torch.zeros(size, dtype=values.dtype, device=values.device)
    total.index_add_(0, index, torch.exp(values - peak[index]))

    out = torch.full((size,), float("-inf"), dtype=values.dtype, device=values.device)
    occupied = torch.isfinite(peak)
    out[occupied] = peak[occupied] + torch.log(total[occupied])
    return out


def unbalanced_sinkhorn(
    cost: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    n_rows: int,
    n_cols: int,
    epsilon: float,
    rho: Optional[float],
    iterations: int = 50,
    tolerance: float = 1e-9,
):
    """Scaling iterations for entropic OT on a sparse support.

    Solves, over couplings supported on the given edges,

        min_P  <P, C> + eps KL(P | a x b) + rho KL(P1 | a) + rho KL(P^T 1 | b)

    with uniform marginals ``a``, ``b``. ``rho=None`` constrains the marginals
    exactly (balanced Sinkhorn); finite ``rho`` relaxes them, which is what lets
    mass be dropped instead of forced into a bad match.

    Returns the mass on each edge.
    """
    device, dtype = cost.device, cost.dtype
    log_a = torch.full((n_rows,), -np.log(n_rows), dtype=dtype, device=device)
    log_b = torch.full((n_cols,), -np.log(n_cols), dtype=dtype, device=device)

    # tau=1 constrains the marginals exactly; tau<1 shrinks each update towards
    # zero potential, i.e. towards letting the marginal fall short
    tau = 1.0 if rho is None else rho / (rho + epsilon)

    f = torch.zeros(n_rows, dtype=dtype, device=device)
    g = torch.zeros(n_cols, dtype=dtype, device=device)

    for _ in range(max(1, int(iterations))):
        previous = f

        values = log_a[row] + (f[row] - cost) / epsilon
        lse = _segment_logsumexp(values, col, n_cols)
        g = -tau * epsilon * torch.nan_to_num(lse, neginf=0.0)

        values = log_b[col] + (g[col] - cost) / epsilon
        lse = _segment_logsumexp(values, row, n_rows)
        f = -tau * epsilon * torch.nan_to_num(lse, neginf=0.0)

        if tolerance > 0 and float((f - previous).abs().max()) < tolerance:
            break

    log_mass = log_a[row] + log_b[col] + (f[row] + g[col] - cost) / epsilon
    return torch.exp(log_mass)


@register_correspondence("sinkhorn")
@dataclass(frozen=True)
class SinkhornMatching(CorrespondenceStrategy):
    """Soft, partial pairing from unbalanced optimal transport.

    blur       width of the entropic smoothing, as a distance. Small values
               approach a hard assignment and need more iterations; large ones
               spread each vertex's mass over a wider neighbourhood.
    reach      distance past which mass is dropped rather than transported. This
               is the knob that makes the matching partial in both directions;
               null gives balanced OT, which forces every unit of mass across
               however far it has to go.
    k          neighbours per point, in each direction, forming the support.
    max_dist   hard cutoff applied to the support before solving.
    min_weight edges below this fraction of the heaviest one are dropped, which
               is what turns a soft coupling into a manageable number of pairs.
    normalize  rescale the emitted weights to average 1, so that prior weights
               tuned against unweighted matching keep their meaning.
    """

    blur: float = 5.0e-4
    reach: Optional[float] = 2.6e-3
    k: int = 8
    max_dist: Optional[float] = None
    iterations: int = 50
    min_weight: float = 1.0e-3
    normalize: bool = True
    device: str = "auto"

    def __post_init__(self):
        if self.blur <= 0:
            raise ValueError(f"blur must be positive, got {self.blur}")
        if self.reach is not None and self.reach <= 0:
            raise ValueError(f"reach must be positive or null, got {self.reach}")
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if not 0.0 <= self.min_weight < 1.0:
            raise ValueError(f"min_weight must be in [0, 1), got {self.min_weight}")
        if self.max_dist is not None and self.max_dist <= 0:
            raise ValueError(f"max_dist must be positive or null, got {self.max_dist}")

    def _support(self, source: torch.Tensor, target: torch.Tensor):
        """Candidate edges: k nearest in each direction, unioned and deduplicated."""
        n_target = target.shape[0]

        _, forward = nearest(source, target, k=self.k)
        rows = torch.arange(source.shape[0], device=source.device).repeat_interleave(forward.shape[1])
        cols = forward.reshape(-1)

        _, backward = nearest(target, source, k=self.k)
        back_cols = torch.arange(n_target, device=source.device).repeat_interleave(backward.shape[1])
        back_rows = backward.reshape(-1)

        rows = torch.cat([rows, back_rows])
        cols = torch.cat([cols, back_cols])

        # a pair may be nearest in both directions; keep it once
        unique = torch.unique(rows.to(torch.int64) * n_target + cols.to(torch.int64))
        return unique // n_target, unique % n_target

    def estimate(self, source, target) -> Correspondences:
        src = PointCloud.as_cloud(source)
        tgt = PointCloud.as_cloud(target)

        device = resolve_device(self.device)
        x = torch.as_tensor(src.points, dtype=torch.float32, device=device)
        y = torch.as_tensor(tgt.points, dtype=torch.float32, device=device)

        row, col = self._support(x, y)
        cost = (x[row] - y[col]).pow(2).sum(-1)

        if self.max_dist is not None:
            keep = cost <= self.max_dist ** 2
            row, col, cost = row[keep], col[keep], cost[keep]

        if row.numel() == 0:
            return Correspondences.from_arrays([], [], [], n_source=len(src), n_target=len(tgt))

        mass = unbalanced_sinkhorn(
            cost, row, col, len(src), len(tgt),
            epsilon=self.blur ** 2,
            rho=None if self.reach is None else self.reach ** 2,
            iterations=self.iterations,
        )

        if self.min_weight > 0:
            keep = mass >= self.min_weight * mass.max()
            row, col, cost, mass = row[keep], col[keep], cost[keep], mass[keep]

        if self.normalize and mass.numel():
            mass = mass / mass.mean()

        return Correspondences.from_arrays(
            row.cpu().numpy(),
            col.cpu().numpy(),
            cost.sqrt().cpu().numpy(),
            weight=mass.cpu().numpy(),
            n_source=len(src),
            n_target=len(tgt),
        )
