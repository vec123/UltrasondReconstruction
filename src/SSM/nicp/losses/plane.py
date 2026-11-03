"""Point-to-plane losses: plain, and the Chamfer variant that projects onto the target's tangent planes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import torch
from src.SSM.nicp.matching.neighbours import nearest
from src.shared.core.registry import RegistryError

from .base import Loss, LossContext, LossValue, _subsample, register_loss
from .chamfer import ChamferLoss


@register_loss("point_to_plane")
@dataclass(frozen=True)
class PointToPlaneLoss(Loss):
    """Matched distance measured along the target normal, not straight-line.

    Only the component of the residual perpendicular to the target surface is
    penalised, so the model is free to slide tangentially instead of being
    pulled at whichever point it happened to be paired with. On smooth regions
    that converges markedly faster than point-to-point.

    Like ``matched_l2`` this consumes a pairing rather than making one, so the
    rejection rules are the correspondence method's. Needs target normals.
    """

    weight: float = 1.0
    reduction: str = "mean"

    requires_correspondences = True
    needs_target_normals = True

    def __call__(self, pred: torch.Tensor, ctx: LossContext) -> LossValue:
        if ctx.target_normals is None:
            raise RegistryError("loss 'point_to_plane' needs target normals, none were supplied")

        corr = ctx.require_correspondences(self.name)
        device, dtype = pred.device, pred.dtype

        src = torch.as_tensor(corr.source_idx, dtype=torch.long, device=device)
        tgt = torch.as_tensor(corr.target_idx, dtype=torch.long, device=device)
        pair_weight = torch.as_tensor(corr.weight, dtype=dtype, device=device)

        delta = pred[src] - ctx.target[tgt]
        signed = (delta * ctx.target_normals[tgt]).sum(-1)
        weighted = pair_weight * signed.pow(2)

        value = weighted.sum() / pair_weight.sum() if self.reduction == "mean" else weighted.sum()
        total = self.weight * value
        return LossValue(total, {"point_to_plane": float(total.detach())})


def _mesh_edges(faces, device) -> torch.Tensor:
    """Unique undirected edges of a triangulation, as a ``(2, E)`` index tensor."""
    f = np.asarray(faces, dtype=np.int64)
    pairs = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    return torch.as_tensor(pairs.T, dtype=torch.long, device=device)


@register_loss("p2plane_chamfer")
class P2PlaneChamferLoss(ChamferLoss):
    """``chamfer``'s per-direction weighting, with the residual measured to the
    target's tangent plane rather than to its nearest sample.

    Point-to-point Chamfer floors at roughly half the target's local sample
    spacing - a model vertex can never score better than the nearest scan
    *vertex* - and it pins tangentially, so vertices stick to samples instead of
    sliding along the surface. Projecting the residual onto the target normal
    removes both: only off-surface displacement is penalised, tangential motion
    is free, and the achievable residual is set by the surface rather than by
    how densely it happens to be sampled. The standard ICP point-to-plane
    argument, applied to a term that makes its own pairs.

    BOTH directions project onto the TARGET's normal, never the model's::

        source_to_target    ((x_i - t*) . n_t*)^2     t* = nearest target to x_i
        target_to_source    ((x* - t_j) . n_tj)^2     x* = nearest model vtx to t_j

    so the only normals needed are the target's, which are static: resolved once
    per fit rather than rebuilt from the moving model every iteration.
    Orientation is irrelevant - the projection is squared, so n and -n score
    alike and a target with inconsistent winding needs no repair.

    ``plane_weight`` blends against plain point-to-point. Pure point-to-plane
    (1.0) leaves tangential motion completely unconstrained, which is degenerate
    wherever the target is locally flat: the model can slide within the plane at
    no cost and nothing pulls it back. The default keeps a point-to-point anchor
    for exactly that reason.
    """

    needs_target_normals = True

    def __init__(self, source_to_target=None, target_to_source=None,
                 plane_weight: float = 0.8, p: int = 2,
                 source_ratio: float = 0.0, target_ratio: float = 0.0,
                 stiffness: float = 0.0,
                 robust_kernel: Optional[str] = None,
                 robust_breakpoint: Optional[float] = None,
                 robust_slope: Optional[float] = None,
                 robust_tau: Optional[float] = None):
        super().__init__(source_to_target=source_to_target, target_to_source=target_to_source,
                         p=p, source_ratio=source_ratio, target_ratio=target_ratio,
                         robust_kernel=robust_kernel, robust_breakpoint=robust_breakpoint,
                         robust_slope=robust_slope, robust_tau=robust_tau)
        if not 0.0 <= plane_weight <= 1.0:
            raise ValueError(
                f"plane_weight must be in [0, 1], got {plane_weight}: it blends "
                f"point-to-point (0) against point-to-plane (1), it is not a scale"
            )
        self.plane_weight = plane_weight
        if stiffness < 0:
            raise ValueError(f"stiffness must be >= 0, got {stiffness}")
        self.stiffness = stiffness

    def __call__(self, pred: torch.Tensor, ctx: LossContext) -> LossValue:
        if ctx.target_normals is None:
            raise RegistryError(
                "loss 'p2plane_chamfer' needs the target's per-point normals, but none "
                "reached it. Failing rather than falling back to point-to-point on "
                "purpose: a silent fallback is indistinguishable from the term working."
            )

        source, source_idx = _subsample(pred, self.source_ratio, ctx.generator)
        target, target_idx = _subsample(ctx.target, self.target_ratio, ctx.generator)

        normals = ctx.target_normals
        if target_idx is not None:
            normals = normals[target_idx]      # gathered on the SAME draw as target
        # Unit-length defensively: a normal array that was scaled, interpolated
        # or averaged is no longer unit, and a non-unit n silently rescales the
        # projection per point - a weighting nobody configured.
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        total = torch.zeros((), dtype=pred.dtype, device=pred.device)
        terms: Dict[str, float] = {}
        st, ts = self.source_to_target, self.target_to_source
        a = self.plane_weight

        if st.active:
            tree = ctx.target_tree if target_idx is None else None
            _, idx = nearest(source.detach(), target, k=1, tree=tree)
            nn = idx[:, 0]
            offset = source - target[nn]
            d_point = offset.pow(2).sum(-1)
            d_plane = (offset * normals[nn]).sum(-1).pow(2)
            weight = self._field(st, "source_to_target", ctx.space.n_vertices, ctx, None)
            if weight is not None and source_idx is not None:
                weight = weight[source_idx]
            value = self._weighted_mean(self._penalty((1.0 - a) * d_point + a * d_plane), weight)
            total = total + st.precision * value
            terms["p2plane_source_to_target"] = float(value.detach())

        if ts.active:
            _, idx = nearest(target, source.detach(), k=1)
            offset = source[idx[:, 0]] - target
            d_point = offset.pow(2).sum(-1)
            d_plane = (offset * normals).sum(-1).pow(2)     # the target's own normal
            weight = self._field(ts, "target_to_source", ctx.target.shape[0],
                                 ctx, ctx.target_cloud)
            if weight is not None and target_idx is not None:
                weight = weight[target_idx]
            value = self._weighted_mean(self._penalty((1.0 - a) * d_point + a * d_plane), weight)
            total = total + ts.precision * value
            terms["p2plane_target_to_source"] = float(value.detach())

        terms["p2plane_chamfer"] = float(total.detach())
        terms["chamfer_n_source"] = float(source.shape[0])
        terms["chamfer_n_target"] = float(target.shape[0])

        if self.stiffness > 0.0:
            total = total + self.stiffness * self._stiffness_energy(pred, ctx, terms)
        return LossValue(total, terms)

    def _stiffness_energy(self, pred: torch.Tensor, ctx: LossContext, terms) -> torch.Tensor:
        """Mean squared *strain* of the displacement field over the mesh edges.

        ``E_stiff`` of the classical NICP energy, in the only form that means
        anything here. Amberg's term penalises differences between neighbouring
        per-vertex affine transforms; this model has no per-vertex transforms to
        compare, only the coefficients, so what is penalised instead is the
        variation of the displacement ``d = x(alpha) - xbar`` across each edge::

            E_stiff = mean over edges (i,j) of  ||d_i - d_j||^2 / ||xbar_i - xbar_j||^2

        A pure translation of the whole surface costs nothing, as it must - the
        term charges for *differential* deformation, i.e. for stretching and
        bending the template rather than for moving it.

        The division by the rest edge length is what makes the weight portable.
        Without it the energy carries units of length squared, and since the two
        domains here differ by a factor of ~5000 in scale (see the Units note in
        src/SSM/kernels/stationary.py) one weight could not serve both. Divided,
        each edge contributes a squared relative stretch, which is dimensionless
        and means the same thing on either model.

        Edge weighting is uniform rather than cotangent-based: the mesh is fixed
        and roughly uniform, so the refinement would buy accuracy the rest of
        this term does not have. The edges are built once per fit and cached on
        the context, since the triangulation never moves.
        """
        faces = ctx.space.faces
        if faces is None:
            raise RegistryError(
                "loss 'p2plane_chamfer' was given stiffness > 0, but the shape space "
                "carries no triangulation - it has no edges to be stiff across. The "
                "pipeline only attaches the model's faces when nicp.use_model_faces is "
                "true (it is false by default, which treats the model as a point cloud); "
                "set that, or set stiffness: 0."
            )
        edges = ctx.cache.get("mesh_edges")
        rest = ctx.cache.get("mesh_rest_length_sq")
        if edges is None:
            edges = _mesh_edges(faces, pred.device)
            mean_xyz = ctx.space.mean.view(-1, 3)
            rest = (mean_xyz[edges[0]] - mean_xyz[edges[1]]).pow(2).sum(-1).clamp_min(1e-30)
            ctx.cache["mesh_edges"] = edges
            ctx.cache["mesh_rest_length_sq"] = rest

        displacement = pred - ctx.space.mean.view(-1, 3)
        delta = displacement[edges[0]] - displacement[edges[1]]
        value = (delta.pow(2).sum(-1) / rest).mean()
        terms["p2plane_stiffness"] = float(value.detach())
        terms["p2plane_total"] = float(
            terms["p2plane_chamfer"] + self.stiffness * value.detach()
        )
        return value

    def describe(self) -> str:
        stiff = f"stiffness={self.stiffness:g}, " if self.stiffness else ""
        return super().describe().replace(
            "ChamferLoss(",
            f"P2PlaneChamferLoss(plane_weight={self.plane_weight:g}, {stiff}")
