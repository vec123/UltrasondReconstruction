"""Per-node weights of the code L2 penalty on a direct position-aware grid.

On a direct grid (code.grid_direct) each code entry belongs to one grid node, so
the penalty can hold some regions at the mean (zero code) harder than others.
The weights are fixed before training, one per member and code entry, and the
auto-decoder multiplies its L2 penalty by them (AutoDecoderCodes.set_penalty_weights).
"""

from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from src.models.lddmm.resnet_flow.config import GridRegCfg


def trilinear_corners(x: Tensor, g: int, domain: Tuple[float, float]) -> Tuple[Tensor, Tensor]:
    """The 8 grid nodes around each point and their trilinear weights.

    Nodes are spread evenly over `domain` per axis, g per axis, node (i, j, k) at
    index i*g*g + j*g + k. Outside the domain the nearest edge cell is used and its
    weights extrapolate linearly (they can leave [0, 1]).

    Args:
        x: [..., 3] points in the normalised frame
        g: grid resolution, at least 2
        domain: (min, max) of the normalised frame, per axis

    Returns:
        (nodes [..., 8] long, weights [..., 8])
    """
    lo, hi = domain
    t = (x - lo) / (hi - lo) * (g - 1)
    cell = t.detach().floor().clamp(0, g - 2).long()
    frac = t - cell
    nodes, weights = [], []
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                nodes.append(((cell[..., 0] + di) * g + (cell[..., 1] + dj)) * g + (cell[..., 2] + dk))
                weights.append((frac[..., 0] if di else 1 - frac[..., 0])
                               * (frac[..., 1] if dj else 1 - frac[..., 1])
                               * (frac[..., 2] if dk else 1 - frac[..., 2]))
    return torch.stack(nodes, dim=-1), torch.stack(weights, dim=-1)


def node_support(points: Tensor, g: int, domain: Tuple[float, float]) -> Tensor:
    """[g³] summed trilinear weight of `points` ([N, 3]) at each node; points
    outside the domain count only with their in-range (non-negative) weights."""
    nodes, weights = trilinear_corners(points, g, domain)
    support = torch.zeros(g ** 3, dtype=points.dtype, device=points.device)
    return support.index_add_(0, nodes.reshape(-1), weights.clamp(min=0).reshape(-1))


def coverage_weights(members: Sequence[Tensor], g: int, domain: Tuple[float, float],
                     cfg: GridRegCfg) -> Tensor:
    """[M, g³] node weights from each member's own target points ([N, 3] each):
    high where they do not reach, low from full_support on, linear in between."""
    support = torch.stack([node_support(p, g, domain) for p in members])
    covered = (support / cfg.full_support).clamp(max=1.0)
    return cfg.high - (cfg.high - cfg.low) * covered


def region_weights(landmarks: np.ndarray, num_members: int, g: int,
                   domain: Tuple[float, float], cfg: GridRegCfg) -> Tensor:
    """[M, g³] node weights, the same for every member: low at the corners of the
    cells holding `landmarks` ([L, 3], normalised frame), high elsewhere."""
    nodes, _ = trilinear_corners(torch.as_tensor(landmarks, dtype=torch.get_default_dtype()), g, domain)
    weights = torch.full((g ** 3,), cfg.high)
    weights[nodes.reshape(-1).unique().cpu()] = cfg.low
    return weights.expand(num_members, -1).clone()


def distance_weights(template_points: Tensor, num_members: int, g: int,
                     domain: Tuple[float, float], cfg: GridRegCfg) -> Tensor:
    """[M, g³] node weights, the same for every member, from each node's distance to
    the nearest of `template_points` ([N, 3], normalised frame; the template's
    points, or its landmarks for face_distance) in grid cells: low up to cfg.near,
    high from cfg.far on, linear in between."""
    lo, hi = domain
    axis = torch.linspace(lo, hi, g, dtype=template_points.dtype, device=template_points.device)
    nodes = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    cells = torch.cdist(nodes, template_points).min(dim=1).values / ((hi - lo) / (g - 1))
    ramp = ((cells - cfg.near) / (cfg.far - cfg.near)).clamp(0.0, 1.0)
    return (cfg.low + (cfg.high - cfg.low) * ramp).expand(num_members, -1).clone()


def grid_penalty_weights(code_cfg, members: List[Tensor], template_points: Tensor,
                         template_landmarks, domain) -> Tensor:
    """[M, n_z] penalty weight per member and code entry of a direct grid.

    Args:
        code_cfg: CodeCfg with grid_direct set
        members: each member's target points [N, 3] in the normalised frame
        template_points: the template's points [N, 3] in the normalised frame (distance only)
        template_landmarks: callable labels -> [L, 3] normalised positions (region only)
        domain: (min, max) of the normalised frame

    Returns:
        [M, g³ * C], each node's weight repeated over its C channels
    """
    g, channels, reg = code_cfg.grid, code_cfg.grid_channels, code_cfg.grid_reg
    if reg.kind == "coverage":
        per_node = coverage_weights(members, g, domain, reg)
    elif reg.kind == "distance":
        per_node = distance_weights(template_points, len(members), g, domain, reg)
    elif reg.kind == "face_distance":
        landmarks = torch.as_tensor(template_landmarks(list(reg.labels)), dtype=template_points.dtype,
                                    device=template_points.device)
        per_node = distance_weights(landmarks, len(members), g, domain, reg)
    else:
        per_node = region_weights(template_landmarks(list(reg.labels)), len(members), g, domain, reg)
    return per_node.repeat_interleave(channels, dim=1)
