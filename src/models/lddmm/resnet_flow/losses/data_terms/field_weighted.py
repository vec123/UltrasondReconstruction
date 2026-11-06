"""Chamfer data terms with a per-vertex precision field on each direction."""

import copy
from typing import Optional
import torch
from torch import Tensor

from .distance import LossDistanceHandle
from .two_sided import TwoSidedDataTerm


class FieldWeightedCDData(TwoSidedDataTerm):
    """Chamfer with an independent (weight, base_sigma) per direction (ChamferSideCfg),
    each direction optionally modulated by a per-vertex precision field.

    Generalises CDData/WeightedCDData's "one shared source_to_target_weight/
    target_to_source_weight, one shared external loss.sigma, at most one implicit
    field (tgt_w only, never pred_w)" into one independent (weight, base_sigma)
    PER direction, with BOTH directions able to carry a field -- source_to_target
    via pred_w, target_to_source via tgt_w. See ChamferSideCfg for the config
    shape (data_name: field_weighted_chamfer takes {source_to_target: {...},
    target_to_source: {...}} in data_kwargs, not flat *_weight floats).

    This class owns its precision scaling ENTIRELY: unlike every other data term
    here, it does NOT depend on the external loss.sigma-derived composer weight
    -- runner.build sets that external weight to 1.0 whenever this data term is
    selected, specifically so the two sigmas (this one, per direction; the
    external one, global) never silently compound into an undocumented product.

    field_name/field_source themselves are NOT read here -- see ChamferSideCfg's
    docstring for why. They are kept as plain attributes (set below) purely so
    runner._add_field_weighted_chamfer can read them back off the already-built,
    already-validated data term instead of re-parsing the raw config dict.

    Each direction also has its OWN LossDistanceHandle. A side whose
    ChamferSideCfg carries its own loss_distance_handle block uses it; a side
    without one inherits the shared block, exactly as before per-direction
    settings existed. That is what
    lets source_to_target ignore far pairs -- the part of a full template a
    partial scan never saw -- while target_to_source keeps pulling on every
    observed point, however far.
    """

    def __init__(self, source_to_target: Optional[dict] = None,
                 target_to_source: Optional[dict] = None,
                 loss_distance_handle: Optional[dict] = None):
        """Initialize from two ChamferSideCfg-shaped dicts, one per direction.

        Args:
            source_to_target: dict matching ChamferSideCfg's fields (weight,
                base_sigma, field_name, field_source); {} / None uses its
                defaults (weight=1.0, base_sigma=0.1, no field).
            target_to_source: same, for the other direction.
            loss_distance_handle: the SHARED block, inherited by every direction
                that does not set its own (see _direction_handle); see
                TwoSidedDataTerm for its keys.
        """
        # Local import: config.py is deliberately torch-free at import time
        # (see DTYPES's docstring there), so importing it eagerly at module
        # scope here would pull torch into every caller of config.py that
        # never touches this data term.
        from src.models.lddmm.resnet_flow.config import ChamferSideCfg

        st = ChamferSideCfg.from_dict(source_to_target or {})
        ts = ChamferSideCfg.from_dict(target_to_source or {})
        # The shared loss_distance_handle is NOT handed to the base: it would log
        # it as the one setting for both directions, which a side's own may
        # override. Handles are built and logged per direction here instead.
        super().__init__(source_to_target_weight=st.weight, target_to_source_weight=ts.weight)
        self.loss_distance_handle = LossDistanceHandle.from_dict(loss_distance_handle)
        self.source_to_target_loss_distance_handle = self._direction_handle("source_to_target", st)
        self.target_to_source_loss_distance_handle = self._direction_handle("target_to_source", ts)
        self.source_to_target_sigma = st.base_sigma
        self.target_to_source_sigma = ts.base_sigma
        self.source_to_target_field_name = st.field_name
        self.source_to_target_field_source = st.field_source
        self.source_to_target_field_transform = st.field_transform
        self.source_to_target_field_scale = st.field_scale
        self.target_to_source_field_name = ts.field_name
        self.target_to_source_field_source = ts.field_source
        self.target_to_source_field_transform = ts.field_transform
        self.target_to_source_field_scale = ts.field_scale

        if st.inherit_source_field:
            raise ValueError(
                "inherit_source_field is a target_to_source setting: it gives each "
                "TARGET point the source field at its matched source point. The "
                "source_to_target side already reads the source field directly."
            )
        if ts.inherit_source_field and st.field_name is None:
            raise ValueError(
                "target_to_source.inherit_source_field needs a field to inherit: set "
                "source_to_target.field_name (e.g. landmark_weight)."
            )
        self.target_to_source_inherit_source_field = ts.inherit_source_field
        if ts.inherit_source_field:
            print(f"[data_term] target_to_source: each target point weighted by "
                  f"{st.field_name!r} at its nearest deformed source point")

        if ts.inherit_target_field:
            raise ValueError(
                "inherit_target_field is a source_to_target setting: it gives each "
                "SOURCE point the target field at its matched target point. The "
                "target_to_source side already reads the target field directly."
            )
        if st.inherit_target_field is True and ts.field_name is None:
            raise ValueError(
                "source_to_target.inherit_target_field: true inherits the target_to_source "
                "field, but there is none: set target_to_source.field_name, or name the "
                "field to inherit (e.g. inherit_target_field: edge_weight)."
            )
        self.source_to_target_inherit_target_field = bool(st.inherit_target_field)
        # A named field reaches forward() as tgt_match_w (runner resolves it per member).
        self.source_to_target_inherit_field_name = (st.inherit_target_field
                                                    if isinstance(st.inherit_target_field, str) else None)
        if st.inherit_target_field:
            inherited = self.source_to_target_inherit_field_name or ts.field_name
            print(f"[data_term] source_to_target: each source point weighted by "
                  f"{inherited!r} at its nearest target point")

    def _source_to_target_weight(self, pred_w: Optional[Tensor], tgt_w: Optional[Tensor],
                                 nearest_target: Tensor,
                                 tgt_match_w: Optional[Tensor] = None) -> Optional[Tensor]:
        """The per-source-point weight source_to_target's mean runs under.

        ``pred_w`` unchanged unless inherit_target_field is set; then each source
        point's matched target point's ``tgt_match_w`` (a named field) or ``tgt_w``
        (``true``), times ``pred_w`` when that is set too. The mirror of
        _target_to_source_weight: the match is an argmin, so the weight is a
        constant for this step.

        Args:
            pred_w: [1 or B, N] source weights, or None
            tgt_w: [1 or B, M] target_to_source field at the target points, or None
            nearest_target: [B, N] index of each source point's nearest target point
            tgt_match_w: [1 or B, M] the named inherited field at the target points, or None

        Returns:
            [B, N] (or pred_w's shape) weights, or None for a plain mean.

        Raises:
            ValueError: inheriting, but the field to inherit did not reach the data term.
        """
        if not self.source_to_target_inherit_target_field:
            return pred_w
        source = tgt_match_w if self.source_to_target_inherit_field_name else tgt_w
        if source is None:
            raise ValueError(
                f"source_to_target.inherit_target_field is set, but the target field "
                f"{self.source_to_target_inherit_field_name or 'of target_to_source'!s} did "
                f"not reach the data term (a named field is resolved in cohort mode only, "
                f"runner._add_cohort_field_weighted_chamfer)."
            )
        batch = nearest_target.shape[0]
        inherited = source.detach().expand(batch, -1).gather(1, nearest_target)
        return inherited if pred_w is None else inherited * pred_w

    def _target_to_source_weight(self, pred_w: Optional[Tensor], tgt_w: Optional[Tensor],
                                 nearest_source: Tensor) -> Optional[Tensor]:
        """The per-target-point weight target_to_source's mean runs under.

        ``tgt_w`` unchanged unless inherit_source_field is set; then each target
        point's matched source point's ``pred_w``, times ``tgt_w`` when that is
        set too. The match is an argmin, so no gradient flows through it -- the
        weight is a constant for this step, exactly like a configured field.

        Args:
            pred_w: [1 or B, N] source field at the flowed points, or None
            tgt_w: [1 or B, M] target weights, or None
            nearest_source: [B, M] index of each target point's nearest pred point

        Returns:
            [B, M] (or tgt_w's shape) weights, or None for a plain mean.

        Raises:
            ValueError: inheriting, but no source field reached the data term.
        """
        if not self.target_to_source_inherit_source_field:
            return tgt_w
        if pred_w is None:
            raise ValueError(
                "target_to_source.inherit_source_field is set, but no source field "
                "reached the data term (pred_w is None). The source_to_target field "
                "is resolved in runner._add_field_weighted_chamfer, pair mode only."
            )
        batch = nearest_source.shape[0]
        inherited = pred_w.detach().expand(batch, -1).gather(1, nearest_source)
        return inherited if tgt_w is None else inherited * tgt_w

    def _direction_handle(self, direction: str, side) -> LossDistanceHandle:
        """One direction's LossDistanceHandle, logged as one line.

        Its own when ``side`` (a ChamferSideCfg) carries a loss_distance_handle
        block -- built from that block alone, never mixed with the shared one --
        otherwise the shared ``self.loss_distance_handle`` itself.

        Args:
            direction: "source_to_target" or "target_to_source", for the log line.
            side: that direction's ChamferSideCfg.

        Returns:
            The handle this direction's residuals go through in forward.
        """
        if side.loss_distance_handle is None:
            handle, origin = self.loss_distance_handle, "shared"
        else:
            handle, origin = LossDistanceHandle.from_dict(side.loss_distance_handle), "own"
        unused = "" if side.weight != 0.0 else " [weight 0: never computed]"
        print(f"[data_term] {direction} ({origin}): {handle.describe()}{unused}")
        return handle

    # The per-side settings forward() reads. field_name/field_source are not
    # among them: fields reach forward() as pred_w/tgt_w, attached to the points.
    _SIDE_ATTRS = (("source_to_target_weight", "target_to_source_weight"),
                   ("source_to_target_sigma", "target_to_source_sigma"),
                   ("source_to_target_loss_distance_handle",
                    "target_to_source_loss_distance_handle"))

    def mirrored(self) -> "FieldWeightedCDData":
        """This term with its two sides exchanged, for the backward half of
        ``direction: bidirectional``.

        BidirectionalMappingError also scores D(phi^-1(target), source). Called
        as-is, that would put each side's settings on the OTHER claim, since they
        follow the argument position (see TwoSidedDataTerm):

            first argument -> second    "the target is explained by the source"
            second -> first argument    "the source lies on the target"

        i.e. target_to_source's and source_to_target's claims respectively --
        with the plain call, a source_to_target handle truncated for a partial
        target would switch off the wrong half, and the untruncated one would pull
        the whole source onto the patches the scan saw. Swapping the settings
        keeps each attached to its claim, whichever shape is being flowed.

        The point-weight fields need no swap: the mapping error hands each cloud's
        own field in alongside its points.

        A shallow copy -- it shares the LossDistanceHandle objects, which hold
        configuration only, and this term has no parameters.

        Returns:
            A new instance of the same class; ``self`` is unchanged.
        """
        mirror = copy.copy(self)
        for st, ts in self._SIDE_ATTRS:
            setattr(mirror, st, getattr(self, ts))
            setattr(mirror, ts, getattr(self, st))
        return mirror

    @staticmethod
    def _weighted_mean(values: Tensor, weight: Optional[Tensor]) -> Tensor:
        """Mean of values, optionally under a per-point precision weight.

        Normalises the weight to mean 1 first -- the same convention
        WeightedCDData already uses for tgt_w -- so turning a field ON changes
        WHERE the loss concentrates, not its overall magnitude: a config that
        adds a field should not also need every weight/sigma retuned just to
        stay in the same ballpark as before.
        """
        if weight is None:
            return values.mean()
        w = weight / (weight.mean(dim=-1, keepdim=True) + 1e-8)
        return (values * w).mean()

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
        per_point: bool = False,
        tgt_match_w: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the field-weighted, per-direction-sigma Chamfer distance.

        Args:
            pred: [B, N, 3] predicted (flowed source) positions
            target: [B, M, 3] target positions
            pred_w: [B, N] per-source-point precision multiplier (source_to_target
                direction), or None for a constant field. Supplied by the
                mapping error once a field is resolved -- see
                field_weights.resolve_point_weight_field.
            tgt_w: [B, M] per-target-point precision multiplier (target_to_source
                direction), or None.
            normals: ignored (kept for protocol compatibility)
            per_point: if True, return RAW (unweighted, no loss_distance_handle) per-point
                target->source distances [B, M] -- same contract as CDData's,
                for adaptive-subsampling callers that need ranks, not the loss.

        Returns:
            Scalar loss if per_point=False, else per-point distances [B, M].
        """
        dist_sq = torch.cdist(pred, target, p=2).pow(2)

        if per_point:
            return dist_sq.min(dim=1)[0]

        term1 = None
        if self.scores_source_to_target:
            d_min, idx = dist_sq.min(dim=2)                       # [B, N]
            d1 = self.source_to_target_loss_distance_handle(d_min)
            w1 = self._source_to_target_weight(pred_w, tgt_w, idx, tgt_match_w)
            term1 = self._weighted_mean(d1, w1) / (2.0 * self.source_to_target_sigma ** 2)

        term2 = None
        if self.scores_target_to_source:
            d_min, idx = dist_sq.min(dim=1)                       # [B, M]
            d2 = self.target_to_source_loss_distance_handle(d_min)
            w2 = self._target_to_source_weight(pred_w, tgt_w, idx)
            term2 = self._weighted_mean(d2, w2) / (2.0 * self.target_to_source_sigma ** 2)

        return self._combine(term1, term2)


class FieldWeightedP2PlaneData(FieldWeightedCDData):
    """FieldWeightedCDData's per-direction (weight, base_sigma, field) scaling, with
    the residual measured to the target's TANGENT PLANE rather than to its nearest
    SAMPLE (data_name: field_weighted_point2plane_chamfer).

    Point-to-point chamfer floors at roughly half the target's local sample
    spacing -- a source point can never score better than the nearest target
    VERTEX -- and it pins tangentially, so points stick to samples instead of
    sliding along the surface. Projecting the residual onto the target normal
    removes both: only OFF-SURFACE displacement is penalised, tangential motion
    is free, and the achievable residual is set by the surface rather than by how
    densely it happens to be sampled. The standard ICP point-to-plane argument.

    BOTH directions project onto the TARGET's normal, never the deformed source's:

        source_to_target    ((phi(p) - t*) . n_t*)^2    t* = nearest target to p
        target_to_source    ((p* - t)      . n_t )^2    p* = nearest source to t

    so the only normals this term ever needs are the target's, which are STATIC --
    resolved once at load time, never recomputed. The symmetric alternative (the
    deformed source's own normal for the second direction) would need per-vertex
    normals rebuilt differentiably from faces at every step, to measure the same
    surface the target normal already describes.

    Orientation does not matter: the projection is SQUARED, so n and -n score
    identically. A target with inconsistent winding needs no repair first.

    A ZERO-length normal (a vertex whose incident faces are degenerate or cancel)
    survives normalisation as zero, so its plane distance is zero and that point
    falls back to the point-to-point half of ``plane_weight``. Graceful, and rare
    -- the point_fields kind normals prints the count, so a target where it is
    NOT rare is visible before training rather than after.

    ``plane_weight`` blends against plain point-to-point. Pure point-to-plane
    (1.0) leaves tangential motion completely unconstrained, which is degenerate
    wherever the target is locally flat -- the source can slide within the plane
    at no cost, and nothing pulls it back. The default keeps a point-to-point
    anchor for exactly that reason.

    NOTE ON SCALE: the projected residual is always <= the point-to-point one (a
    vector's component along a unit normal never exceeds its length), so at a
    FIXED base_sigma this term is WEAKER than field_weighted_chamfer. The two are
    not calibrated against each other -- expect to lower base_sigma when switching,
    rather than assuming the old value still means the same thing.
    """

    def __init__(self, source_to_target: Optional[dict] = None,
                 target_to_source: Optional[dict] = None,
                 plane_weight: float = 0.8,
                 loss_distance_handle: Optional[dict] = None):
        """Initialize from two ChamferSideCfg-shaped dicts, plus the blend.

        Args:
            source_to_target, target_to_source: see FieldWeightedCDData.
            plane_weight: 0.0 = plain point-to-point (this term becomes
                FieldWeightedCDData), 1.0 = pure point-to-plane. In between,
                the per-point penalty is the convex blend of the two.
            loss_distance_handle: the shared block; see FieldWeightedCDData.
                Each direction's LossDistanceHandle is applied to the BLENDED
                penalty, so its breakpoint keeps meaning "a distance, squared".
        """
        super().__init__(source_to_target=source_to_target,
                         target_to_source=target_to_source,
                         loss_distance_handle=loss_distance_handle)
        if not 0.0 <= plane_weight <= 1.0:
            raise ValueError(
                f"plane_weight must be in [0, 1], got {plane_weight}: it is a blend "
                f"between point-to-point (0) and point-to-plane (1), not a scale."
            )
        self.plane_weight = plane_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
        per_point: bool = False,
        tgt_match_w: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the field-weighted, per-direction-sigma point-to-plane distance.

        Args:
            pred: [B, N, 3] predicted (flowed source) positions
            target: [B, M, 3] target positions
            pred_w: [B, N] per-source-point precision multiplier, or None
            tgt_w: [B, M] per-target-point precision multiplier, or None
            normals: [B, M, 3] or [1, M, 3] TARGET normals, addressing the same
                M vertices as ``target`` and in the same order. Required -- see
                the raise below for why this is not silently optional.
            per_point: if True, return RAW (unweighted, no loss_distance_handle, and
                point-to-POINT) per-point target->source distances [B, M]. Same
                contract as CDData's, so adaptive-subsampling callers that rank
                by it are unaffected by the plane projection.

        Returns:
            Scalar loss if per_point=False, else per-point distances [B, M].

        Raises:
            ValueError: normals is None, or does not address target's vertices.
        """
        if normals is None:
            raise ValueError(
                "data_name='field_weighted_point2plane_chamfer' needs the TARGET's "
                "per-vertex normals, but none reached the data term. Bake a 'normal' "
                "point array onto the target .vtp with "
                "configs/surface/target/surface_normals.yaml -- io.load_shape picks that array "
                "up automatically and joint_normalize carries it through. Failing "
                "loudly rather than falling back to point-to-point on purpose: a "
                "silent fallback is indistinguishable from the term working."
            )
        if normals.shape[1] != target.shape[1]:
            raise ValueError(
                f"normals describe {normals.shape[1]} points but target has "
                f"{target.shape[1]}; they must be the same vertices in the same "
                f"order, i.e. gathered on the SAME subsample draw."
            )

        dist_sq = torch.cdist(pred, target, p=2).pow(2)

        if per_point:
            return dist_sq.min(dim=1)[0]

        # Unit-length defensively: a normal array that was scaled, interpolated or
        # averaged is no longer unit, and a non-unit n silently rescales the
        # projection per point -- a weighting nobody configured.
        n = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        if n.shape[0] != target.shape[0]:
            n = n.expand(target.shape[0], -1, -1)
        a = self.plane_weight

        term1 = None
        if self.scores_source_to_target:
            d_point, idx = dist_sq.min(dim=2)                     # [B, N]
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, 3)      # [B, N, 3]
            offset = pred - target.gather(1, gather_idx)          # [B, N, 3]
            d_plane = (offset * n.gather(1, gather_idx)).sum(-1).pow(2)
            d1 = self.source_to_target_loss_distance_handle((1.0 - a) * d_point + a * d_plane)
            w1 = self._source_to_target_weight(pred_w, tgt_w, idx, tgt_match_w)
            term1 = self._weighted_mean(d1, w1) / (2.0 * self.source_to_target_sigma ** 2)

        term2 = None
        if self.scores_target_to_source:
            d_point, idx = dist_sq.min(dim=1)                     # [B, M]
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, 3)      # [B, M, 3]
            offset = pred.gather(1, gather_idx) - target          # [B, M, 3]
            d_plane = (offset * n).sum(-1).pow(2)                 # target's own normal
            d2 = self.target_to_source_loss_distance_handle((1.0 - a) * d_point + a * d_plane)
            w2 = self._target_to_source_weight(pred_w, tgt_w, idx)
            term2 = self._weighted_mean(d2, w2) / (2.0 * self.target_to_source_sigma ** 2)

        return self._combine(term1, term2)
