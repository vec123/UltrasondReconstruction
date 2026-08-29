"""Nearest-neighbour search for the correspondence-free losses.

Chamfer and CPD both need "which reference point is closest", every iteration,
in both directions. Only the *indices* are needed without gradient - the
distance is then recomputed from the live tensors so it stays differentiable
(the index is piecewise constant, so it contributes no gradient of its own).

The search is a spatial query, not a linear-algebra problem, so it runs on a
k-d tree regardless of where the tensors live: 31027 x 11034 is 3.4e8 pairs, and
materialising that as a distance matrix costs 1.4 GB per direction per iteration
in float32 (2.7 GB in float64) for an answer a tree gives in O(N log M). Moving
the points to host memory to do it costs 0.4 MB - four orders of magnitude less
than the matrix it avoids.

The brute-force path is kept for small problems, where building a tree is not
worth it, and as a reference implementation to test the tree against. It casts
to float32 first: the result is an argmin, so double precision buys nothing, and
consumer GPUs run float64 at a thirty-second of their float32 rate.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

#: below this many pairs, a dense distance matrix beats building a tree
BRUTE_FORCE_PAIRS = 250_000


def build_tree(points: torch.Tensor) -> cKDTree:
    """A k-d tree over ``points``; build once and pass it back in when static."""
    data = points.detach().cpu().numpy()
    return cKDTree(np.ascontiguousarray(data, dtype=np.float64))


def _brute_force(query: torch.Tensor, reference: torch.Tensor, k: int, chunk: int):
    """Blocked dense search. Only for small inputs - see the module docstring."""
    q = query.float()
    r = reference.float()
    dists, idx = [], []
    for start in range(0, q.shape[0], chunk):
        block = torch.cdist(q[start:start + chunk], r)
        d, i = torch.topk(block, k=k, dim=1, largest=False, sorted=True)
        dists.append(d)
        idx.append(i)
    return torch.cat(dists).to(query.dtype), torch.cat(idx)


def nearest(
    query: torch.Tensor,
    reference: torch.Tensor,
    k: int = 1,
    tree: Optional[cKDTree] = None,
    backend: str = "auto",
    chunk: int = 4096,
    workers: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """k closest ``reference`` points per ``query`` point, distances ascending.

    Returns ``(distance, index)``, both ``(Q, k)`` and both detached: use the
    index to gather from the live tensor and recompute the distance there.

    ``tree`` reuses a prebuilt index over a reference cloud that has not moved
    (the fit target); pass ``None`` for one that has (the model, mid-fit).
    ``backend`` is ``auto`` (tree unless the problem is tiny), ``kdtree`` or
    ``brute``. ``workers=-1`` spreads the query over all cores.
    """
    k = max(1, min(int(k), reference.shape[0]))

    if backend not in ("auto", "kdtree", "brute"):
        raise ValueError(f"backend must be auto/kdtree/brute, got {backend!r}")
    if backend == "auto":
        small = query.shape[0] * reference.shape[0] <= BRUTE_FORCE_PAIRS
        backend = "brute" if (small and tree is None) else "kdtree"

    with torch.no_grad():
        if backend == "brute":
            return _brute_force(query, reference, k, chunk)

        if tree is None:
            tree = build_tree(reference)
        points = np.ascontiguousarray(query.detach().cpu().numpy(), dtype=np.float64)
        d, i = tree.query(points, k=k, workers=workers)

        d = np.asarray(d).reshape(query.shape[0], k)
        i = np.asarray(i).reshape(query.shape[0], k)
        return (
            torch.as_tensor(d, dtype=query.dtype, device=query.device),
            torch.as_tensor(i, dtype=torch.long, device=query.device),
        )


def squared_distance_to(points: torch.Tensor, reference: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """||points_q - reference[index_q]||^2, differentiable in both arguments."""
    return (points - reference[index]).pow(2).sum(-1)
