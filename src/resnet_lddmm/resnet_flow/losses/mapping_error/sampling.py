"""Subsampling and gathering helpers shared by the mapping errors."""

from typing import Optional, Tuple
import torch
from torch import Tensor


def _draw_indices(n_points: int, n_subsample: Optional[int], device) -> Optional[Tensor]:
    """Draw the vertex subset for ONE cloud, or None when subsampling does not apply.

    Args:
        n_points: N, the cloud's vertex count
        n_subsample: target count; None/<=0 disables, as does N <= n_subsample
        device: device to allocate the index tensor on

    Returns:
        [n_subsample] index tensor, or None when the cloud passes through whole
    """
    if n_subsample is None or n_subsample <= 0 or n_points <= n_subsample:
        return None
    return torch.randperm(n_points, device=device)[:n_subsample]


def _take_vertices(
    points: Tensor,
    weights: Optional[Tensor],
    idx: Optional[Tensor],
    label: str,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Apply one cloud's index draw to its points and its per-vertex weights.

    One draw per CLOUD, never one per tensor: drawing separately for weights leaves
    every shape valid while pairing each vertex with some other vertex's weight —
    correctly shaped and silently wrong.

    Args:
        points: [B, N, 3] cloud
        weights: [B, N] per-vertex weights (e.g. area), or None
        idx: index draw from _draw_indices, or None to pass through unchanged
        label: cloud name, used in the shape-mismatch message

    Returns:
        (points, weights) restricted to idx. weights is None whenever it was None.
    """
    if idx is None:
        return points, weights

    if weights is not None and weights.shape[1] != points.shape[1]:
        raise ValueError(
            f"{label}: weights has {weights.shape[1]} entries for {points.shape[1]} "
            f"points; they must describe the same vertices to be subsampled together."
        )

    return points[:, idx, :], (None if weights is None else weights[:, idx])


def _take_normals(normals: Optional[Tensor], idx: Optional[Tensor],
                  label: str) -> Optional[Tensor]:
    """One cloud's normals, restricted to the SAME draw its points used.

    Separate from _take_vertices only because normals are [B, N, 3] while weights
    are [B, N]; the invariant is identical and the two MUST be called with the
    same ``idx``. Gathering normals on a different draw than their points leaves
    every shape valid while pairing each point with some other point's normal --
    correctly shaped and silently wrong, exactly the failure _take_vertices
    warns about.

    No re-normalisation here: a subsample selects normals, it does not transform
    them, so a unit input stays unit. The data term unit-lengths defensively
    anyway, for inputs that were never unit to begin with.

    Args:
        normals: [B, N, 3] per-vertex normals, or None
        idx: index draw from _draw_indices, or None to pass through unchanged
        label: cloud name, used in the shape-mismatch message

    Returns:
        [B, K, 3] normals restricted to idx, or None whenever normals was None.

    Raises:
        ValueError: normals does not address the cloud idx was drawn for.
    """
    if normals is None or idx is None:
        return normals

    if normals.shape[1] <= int(idx.max()):
        raise ValueError(
            f"{label}: normals has {normals.shape[1]} entries but the subsample "
            f"draw indexes up to {int(idx.max())}; they must describe the same "
            f"vertices as the points to be subsampled together."
        )

    return normals[:, idx, :]


def _gather_field(field: Optional[Tensor], idx: Optional[Tensor]) -> Optional[Tensor]:
    """field, restricted to this step's subsample draw, as a [1, K] weight.

    Mirrors _take_vertices's own idx==None passthrough, so a field lines up
    with its cloud's points whether or not subsampling is active.
    """
    if field is None:
        return None
    gathered = field if idx is None else field[idx]
    return gathered.unsqueeze(0)


def _combine_weights(weights: Optional[Tensor], field_w: Optional[Tensor]) -> Optional[Tensor]:
    """A fixed cloud's tgt_w: its "area" weights and its point-weight field.

    Independent signals (see field_weights.py), so when both are present they
    COMBINE multiplicatively rather than one silently overriding the other.
    """
    if field_w is None:
        return weights
    return field_w if weights is None else weights * field_w


def _draw_pair_indices(template_points: Tensor, sample_points: Tensor,
                       n_subsample: Optional[int]) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Index draws for the template and sample clouds of one step.

    Equal vertex counts mean the clouds may be in correspondence (the samples are
    deformations of the template), so they SHARE a single draw. That keeps
    index-based consumers valid under subsampling — the l2 data term and the rigid
    alignment pose loss both compare row i to row i. Chamfer is permutation-invariant
    and is unaffected either way.

    Unequal counts cannot be corresponded, so they get independent draws.

    Args:
        template_points: [B, N, 3]
        sample_points: [B, M, 3]
        n_subsample: target vertex count; None/<=0 disables

    Returns:
        (template_idx, sample_idx), either of which may be None
    """
    template_idx = _draw_indices(template_points.shape[1], n_subsample, template_points.device)
    if sample_points.shape[1] == template_points.shape[1]:
        return template_idx, template_idx
    return template_idx, _draw_indices(sample_points.shape[1], n_subsample, sample_points.device)
