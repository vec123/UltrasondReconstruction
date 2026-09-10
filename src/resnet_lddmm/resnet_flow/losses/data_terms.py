"""Loss terms for shape registration: data fitting and flow regularization."""

import abc
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from src.resnet_lddmm.learning.losses.losses import chamfer_loss
from src.resnet_lddmm.resnet_flow.trajectory import Trajectory
from src.resnet_lddmm.resnet_flow.fields.base import VelocityField


class RegistrationLoss(nn.Module, abc.ABC):
    """Abstract base for all registration loss terms.

    Subclasses: DataTerm (data fitting), FlowTerm (velocity field regularization).
    """

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Compute loss. Signature varies by subclass."""


class DataTerm(RegistrationLoss):
    """Measures mismatch between predicted and target point clouds.

    Maps [B,N,3] predicted points and [B,M,3] target points to a scalar loss.
    Subclasses decide how to combine these with optional weights and normals.
    """

    @abc.abstractmethod
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute data term loss.

        Args:
            pred: [B, N, 3] predicted point positions
            target: [B, M, 3] target point positions
            pred_w: [B, N] per-point weights for pred (unused by basic terms)
            tgt_w: [B, M] per-point weights for target (unused by basic terms)
            normals: [B, M, 3] target surface normals (unused by basic terms)

        Returns:
            Scalar loss tensor
        """


ROBUST_KINDS = ("quadratic", "linear", "truncated", "exponential")


def _robust_penalty(dist_sq: Tensor, kind: str, breakpoint: float,
                     slope: Optional[float] = None, tau: Optional[float] = None,
                     eps: float = 1e-12) -> Tensor:
    """Per-point robust penalty on squared distance: quadratic below ``breakpoint``,
    one of four selectable shapes above it.

    Every kind agrees EXACTLY with plain squared distance for ``d <= breakpoint``
    (returns ``dist_sq`` itself there, not a reconstruction via sqrt -- so the
    interior regime is bit-for-bit the pre-existing Chamfer behaviour). They differ
    only in the TAIL, ``d > breakpoint``, which is where "further away has less
    importance" is actually implemented -- each kind trades a different amount of
    the far point's influence (the gradient dρ/dd) away:

    - "quadratic": ``d**2`` everywhere, unmodified -- the no-op / baseline case, so
      a run can A/B a robust kernel against plain Chamfer via one config string
      instead of a separate flag.
    - "linear": ``breakpoint**2 + slope*(d - breakpoint)``. Influence SATURATES at
      the constant ``slope`` instead of growing with ``d``. ``slope=None`` defaults
      to ``2*breakpoint``, matching ``d(d**2)/dd`` at the breakpoint -- the classic
      C1-continuous Huber. ``slope=0`` flattens the tail to a constant plateau at
      ``breakpoint**2``: every point beyond the breakpoint still counts for that
      fixed amount, but none of them can be pulled on any harder than another.
    - "truncated": ``0`` beyond the breakpoint -- not a plateau, a hard drop from
      ``breakpoint**2`` down to exactly zero. Points past the breakpoint are fully
      excluded, gradient and all, matching trimmed-ICP-style outlier rejection.
      This is a genuine discontinuity in the penalty's VALUE at the breakpoint,
      deliberately -- it is what "excluded" means, not a smoothed approximation
      of it.
    - "exponential": ``breakpoint**2 + 2*breakpoint*tau*(1 - exp(-(d-breakpoint)/tau))``.
      C1-continuous at the breakpoint (value AND slope match the quadratic side),
      and approaches the finite asymptote ``breakpoint**2 + 2*breakpoint*tau`` as
      ``d -> inf`` with influence decaying like ``exp(-(d-breakpoint)/tau)`` --
      a genuinely redescending (Welsch-style) kernel: far points are not just
      capped, their pull vanishes. ``tau=None`` defaults to ``breakpoint``.

    A redescending kernel ("exponential") can zero out the gradient from a
    correspondence that is far only because the flow has not reached the right
    pose/alignment YET, not because it is a genuine outlier -- removing exactly the
    long-range signal that would have pulled it there. "linear" and "truncated"
    keep pulling (just less steeply, or not into a plateau), so they carry that
    risk less sharply; prefer them unless "exponential" is specifically wanted.

    Args:
        dist_sq: per-point squared distances, any shape.
        kind: one of ROBUST_KINDS.
        breakpoint: the distance (NOT squared) at which behaviour changes from the
            plain quadratic regime, in the same normalised length units as
            ``loss.sigma``.
        slope: "linear" only; see above.
        tau: "exponential" only; see above.
        eps: floor under the sqrt so ``d`` stays differentiable at ``dist_sq == 0``.

    Returns:
        Per-point penalty, same shape as ``dist_sq``.

    Raises:
        ValueError: unknown ``kind``.
    """
    if kind == "quadratic":
        return dist_sq
    if kind not in ROBUST_KINDS:
        raise ValueError(f"unknown robust_kernel {kind!r}; expected one of {ROBUST_KINDS}")

    c = breakpoint
    d = dist_sq.clamp_min(eps).sqrt()
    beyond = d > c

    if kind == "linear":
        s = 2.0 * c if slope is None else slope
        tail = c ** 2 + s * (d - c)
    elif kind == "truncated":
        tail = torch.zeros_like(dist_sq)
    else:  # "exponential"
        t = c if tau is None else tau
        tail = c ** 2 + 2.0 * c * t * (1.0 - torch.exp(-(d - c) / t))

    return torch.where(beyond, tail, dist_sq)


class TwoSidedDataTerm(DataTerm):
    """Chamfer-family base: ``w_st * D(source->target) + w_ts * D(target->source)``.

    "source" and "target" are the config's own two shapes (``source:`` and
    ``target:`` in the YAML). The ``pred`` argument of ``forward`` IS the source:
    the source flowed forward, phi(source), which is the only form of it the data
    term ever sees.

    Both weights default to 1.0 -- the plain symmetric Chamfer every subclass
    computed before these knobs existed. Setting one to 0.0 drops that direction
    ENTIRELY: it is never computed, so it costs nothing and builds no graph.

    Zeroing ``source_to_target_weight`` is what a PARTIAL target needs. That
    direction asserts "every source point lies on the target surface", which is
    FALSE when the target is a partial scan -- most of the source has no
    counterpart in it -- and the cheapest way to satisfy it is to collapse the
    whole source into the retained patches. Dropping it leaves only the half that
    is true: target->source, "every OBSERVED point is explained by the source".
    The unobserved region is then shaped by the REGULARIZER alone, so an
    asymmetric data term is only half a fix -- kinetic/isometry become the model
    of everything the scan does not see.

    Under ``direction: bidirectional`` these names stop being literal. One
    instance is called twice with the roles swapped -- as (flowed source, target)
    and as (flowed target, source), see BidirectionalMappingError -- so the
    weights follow the ARGUMENT POSITION, not the shape, and asymmetry drops the
    first argument's direction in both calls. That is not "one-sided
    registration"; asymmetry is meant for ``direction: forward``.

    No renormalisation by the weight sum, deliberately: (1, 1) has to stay
    bit-for-bit what it was. So zeroing a direction also roughly HALVES the data
    term's magnitude while the kinetic and isometry weights do not change --
    compensate with ``loss.sigma``, which sets the data weight to 1/(2 sigma^2).
    """

    def __init__(self, source_to_target_weight: float = 1.0,
                 target_to_source_weight: float = 1.0,
                 robust_kernel: Optional[str] = None,
                 robust_breakpoint: Optional[float] = None,
                 robust_slope: Optional[float] = None,
                 robust_tau: Optional[float] = None):
        """Initialize the direction weights and optional robust kernel.

        Args:
            source_to_target_weight: weight on D(phi(source) -> target); 0.0 disables it
            target_to_source_weight: weight on D(target -> phi(source)); 0.0 disables it
            robust_kernel: one of ROBUST_KINDS, or None (default) to keep plain
                squared-distance Chamfer -- bit-for-bit the pre-existing behaviour.
                See _robust_penalty for what each kind does.
            robust_breakpoint: the distance (not squared) where behaviour changes
                from the quadratic near-zero regime, in the same normalised length
                units as loss.sigma. Required whenever robust_kernel is set to
                anything other than "quadratic"/None.
            robust_slope: "linear" kernel only; None defaults to the C1-continuous
                Huber slope (2 * robust_breakpoint). 0 flattens the tail to a
                constant plateau.
            robust_tau: "exponential" kernel only; None defaults to robust_breakpoint.
        """
        super().__init__()
        if source_to_target_weight < 0 or target_to_source_weight < 0:
            raise ValueError(
                "direction weights must be >= 0, got source_to_target_weight="
                f"{source_to_target_weight}, target_to_source_weight="
                f"{target_to_source_weight}"
            )
        if source_to_target_weight == 0 and target_to_source_weight == 0:
            raise ValueError(
                "both direction weights are 0 -- the data term would be a constant "
                "zero with no gradient, and training would silently optimize the "
                "regularizers alone. Set at least one > 0."
            )
        self.source_to_target_weight = float(source_to_target_weight)
        self.target_to_source_weight = float(target_to_source_weight)

        if robust_kernel is not None and robust_kernel not in ROBUST_KINDS:
            raise ValueError(f"unknown robust_kernel {robust_kernel!r}; expected "
                              f"one of {ROBUST_KINDS} or None")
        if robust_kernel not in (None, "quadratic") and robust_breakpoint is None:
            raise ValueError(
                f"robust_kernel={robust_kernel!r} needs a robust_breakpoint -- the "
                f"distance at which its behaviour changes from the plain quadratic "
                f"regime. Pass one, or drop robust_kernel to disable it."
            )
        self.robust_kernel = robust_kernel
        self.robust_breakpoint = robust_breakpoint
        self.robust_slope = robust_slope
        self.robust_tau = robust_tau
        if robust_kernel not in (None, "quadratic"):
            extra = {"linear": f"slope={robust_slope!r}",
                     "exponential": f"tau={robust_tau!r}"}.get(robust_kernel, "")
            print(f"[data_term] robust_kernel={robust_kernel!r} "
                  f"breakpoint={robust_breakpoint:g} {extra} -- quadratic below the "
                  f"breakpoint, {robust_kernel!r} above it")

    def _robustify(self, dist_sq: Tensor) -> Tensor:
        """Apply the configured robust kernel to per-point squared distances.

        No-op when robust_kernel is None/"quadratic" -- returns dist_sq unchanged,
        so the default is bit-for-bit identical to plain Chamfer. Meant to be
        called on the ALREADY-REDUCED nearest-neighbour distance (post ``.min()``),
        never on the full pairwise matrix: every kind here is monotonic in ``d``
        below the point where it starts flattening/dropping, so applying it after
        argmin gives the identical nearest-neighbour assignment as before, for a
        fraction of the cost.
        """
        if self.robust_kernel is None or self.robust_kernel == "quadratic":
            return dist_sq
        return _robust_penalty(dist_sq, self.robust_kernel, self.robust_breakpoint,
                                slope=self.robust_slope, tau=self.robust_tau)

    def regime_fraction(self, dist_sq: Tensor) -> float:
        """Fraction of points currently beyond the robust breakpoint.

        The direct way to SEE where the configured breakpoint sits relative to the
        current fit, alongside the printed description at construction time: near
        0 means almost nothing is being robustified yet (breakpoint generous, or
        the fit is already tight); near 1 means most points are in the tail, which
        means either the fit has not converged yet or the breakpoint is too tight
        for the current scale. Returns 0.0 (not an error) when no robust kernel is
        configured, since then nothing is ever "in the tail".

        Args:
            dist_sq: per-point squared distances (e.g. the same tensor _robustify
                would be called on).

        Returns:
            A python float in [0, 1].
        """
        if self.robust_kernel is None or self.robust_kernel == "quadratic":
            return 0.0
        d = dist_sq.clamp_min(0).sqrt()
        return (d > self.robust_breakpoint).float().mean().item()

    @property
    def scores_source_to_target(self) -> bool:
        """Whether source->target is worth computing at all."""
        return self.source_to_target_weight != 0.0

    @property
    def scores_target_to_source(self) -> bool:
        """Whether target->source is worth computing at all."""
        return self.target_to_source_weight != 0.0

    def _combine(self, source_to_target: Optional[Tensor],
                 target_to_source: Optional[Tensor]) -> Tensor:
        """Weighted sum of the two directions; ``None`` is one the caller skipped.

        A direction with a non-zero weight MUST arrive computed. Passing None
        there is a subclass forgetting to guard on the matching ``scores_*``
        property, which would silently drop half the objective -- assert instead.
        """
        total = None
        if self.scores_source_to_target:
            assert source_to_target is not None, \
                "source->target is weighted but was not computed"
            total = self.source_to_target_weight * source_to_target
        if self.scores_target_to_source:
            assert target_to_source is not None, \
                "target->source is weighted but was not computed"
            weighted = self.target_to_source_weight * target_to_source
            total = weighted if total is None else total + weighted
        return total


class CDData(TwoSidedDataTerm):
    """Chamfer distance on unpadded point clouds.

    Wraps the reused chamfer_loss with an all-ones mask, since our inputs
    have no padding. Differentiable, and symmetric at the default weights --
    see TwoSidedDataTerm for the per-direction weights that break that symmetry.

    Supports both scalar loss and per-point losses (for adaptive subsampling in T28).
    """

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
        per_point: bool = False,
    ) -> Tensor:
        """Compute Chamfer distance with all-ones mask.

        Args:
            pred: [B, N, 3] predicted positions
            target: [B, M, 3] target positions
            pred_w, tgt_w, normals: ignored (kept for protocol compatibility)
            per_point: if True, return per-point losses [B, M]; if False, return scalar

        Returns:
            Scalar Chamfer loss if per_point=False, else per-point distances [B, M]
        """
        # Create all-ones mask for unpadded target cloud
        mask = torch.ones(
            target.shape[:2], dtype=torch.bool, device=target.device
        )

        # Compute pairwise distances [B, N, M]
        dist_sq = torch.cdist(pred, target, p=2).pow(2)
        dist_sq = dist_sq.masked_fill(~mask.unsqueeze(1), 1e6)

        if per_point:
            # Return per-point losses from target to pred (term2 of Chamfer).
            # Deliberately UNWEIGHTED by the direction weights: this feeds adaptive
            # subsampling, which needs raw distances to RANK target points by. Under
            # target_to_source_weight=0 the weighted version would be all zeros and
            # rank nothing, which is not what a sampler is asking for.
            per_point_losses = dist_sq.min(dim=1)[0]  # [B, M]
            return per_point_losses

        # Standard scalar loss
        term1 = self._robustify(dist_sq.min(dim=2)[0]).mean() if self.scores_source_to_target else None

        term2 = None
        if self.scores_target_to_source:
            dist_t_to_p = self._robustify(dist_sq.min(dim=1)[0])
            term2 = (dist_t_to_p * mask).sum() / (mask.sum() + 1e-8)

        return self._combine(term1, term2)


class L2Data(DataTerm):
    """Mean squared pointwise error for corresponded point clouds.

    Assumes points are in correspondence by index (e.g., template points
    matched to target points at the same position). Simpler than Chamfer
    but requires pre-established correspondence.

    Supports both scalar loss and per-point losses (for adaptive subsampling in T28).
    """

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
        per_point: bool = False,
    ) -> Tensor:
        """Compute mean squared error between matched clouds.

        Args:
            pred: [B, N, 3] predicted positions
            target: [B, N, 3] target positions (same N as pred)
            pred_w, tgt_w, normals: ignored (kept for protocol compatibility)
            per_point: if True, return per-point losses [B, N]; if False, return scalar

        Returns:
            Scalar MSE loss if per_point=False, else per-point MSE [B, N]
        """
        per_point_losses = (pred - target).pow(2).sum(-1)  # [B, N]

        if per_point:
            return per_point_losses

        return per_point_losses.mean()


class NullData(DataTerm):
    """No-op data term: always zero, contributes no gradient.

    TwoSidedDataTerm.__init__ refuses source_to_target_weight=target_to_source_
    weight=0 specifically to catch that combination happening BY ACCIDENT (see
    its docstring: "training would silently optimize the regularizers alone").
    ``data_name: none`` is the explicit, intentional version of exactly that --
    for a run where a non-chamfer term (e.g. loss.landmark_weight) is meant to be
    the ONLY source of positional supervision, with kinetic/isometry doing the
    rest as regularizers.
    """

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
    ) -> Tensor:
        """Zero, connected to ``pred`` so autograd sees a valid (zero) gradient."""
        return pred.sum() * 0.0


class WeightedCDData(TwoSidedDataTerm):
    """Weighted Chamfer distance using per-point target areas (T33).

    Similar to CDData but weights the target→pred term by target point weights
    (typically surface areas). Reduces to standard Chamfer with uniform weights.

    Two independent weightings meet here, and they are not the same thing:
    ``tgt_w`` is PER TARGET POINT (which observations count for more), while the
    inherited direction weights are PER DIRECTION (which half of Chamfer is
    scored at all). See TwoSidedDataTerm.
    """

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
        per_point: bool = False,
    ) -> Tensor:
        """Compute weighted Chamfer distance.

        Args:
            pred: [B, N, 3] predicted positions
            target: [B, M, 3] target positions
            pred_w: [B, N] weights for pred (ignored)
            tgt_w: [B, M] per-point target weights (surface areas); uniform if None
            normals: ignored (kept for protocol compatibility)
            per_point: if True, return per-point losses [B, M]; if False, return scalar

        Returns:
            Scalar weighted Chamfer loss if per_point=False, else per-point losses [B, M]
        """
        B, N, D = pred.shape
        B_tgt, M, D = target.shape

        # Compute pairwise distances [B, N, M]
        dist_sq = torch.cdist(pred, target, p=2).pow(2)

        if per_point:
            # Return per-point losses from target to pred, weighted by tgt_w
            per_point_losses = dist_sq.min(dim=1)[0]  # [B, M]
            if tgt_w is not None:
                per_point_losses = per_point_losses * tgt_w
            return per_point_losses

        # Term 1: pred → target (unweighted, average nearest target distance)
        term1 = self._robustify(dist_sq.min(dim=2)[0]).mean() if self.scores_source_to_target else None

        # Term 2: target → pred (weighted by tgt_w if provided)
        term2 = None
        if self.scores_target_to_source:
            tgt_to_pred = self._robustify(dist_sq.min(dim=1)[0])  # [B, M]
            if tgt_w is not None:
                # Normalize weights to avoid changing overall scale
                w_normalized = tgt_w / (tgt_w.mean(dim=1, keepdim=True) + 1e-8)
                term2 = (tgt_to_pred * w_normalized).mean()
            else:
                term2 = tgt_to_pred.mean()

        return self._combine(term1, term2)


class PCDData(TwoSidedDataTerm):
    """Point Cloud Distance: symmetric Chamfer-like metric without explicit masking (T33).

    Similar to CDData but computed more directly without mask operations.
    Equivalent to CDData on unpadded point clouds. Per-direction weights are
    inherited from TwoSidedDataTerm; "symmetric" describes the defaults.
    """

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
        per_point: bool = False,
    ) -> Tensor:
        """Compute point cloud distance (symmetric).

        Args:
            pred: [B, N, 3] predicted point positions
            target: [B, M, 3] target point positions
            pred_w, tgt_w, normals: ignored (kept for protocol compatibility)
            per_point: if True, return per-point losses [B, M]; if False, return scalar

        Returns:
            Scalar PCD loss if per_point=False, else per-point distances [B, M]
        """
        # Compute pairwise distances [B, N, M]
        dist_sq = torch.cdist(pred, target, p=2).pow(2)

        if per_point:
            # Return per-point losses from target to pred
            per_point_losses = dist_sq.min(dim=1)[0]  # [B, M]
            return per_point_losses

        # Symmetric at the default weights: pred→target + target→pred
        term1 = self._robustify(dist_sq.min(dim=2)[0]).mean() if self.scores_source_to_target else None
        term2 = self._robustify(dist_sq.min(dim=1)[0]).mean() if self.scores_target_to_source else None
        return self._combine(term1, term2)


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
    """

    def __init__(self, source_to_target: Optional[dict] = None,
                 target_to_source: Optional[dict] = None,
                 robust_kernel: Optional[str] = None,
                 robust_breakpoint: Optional[float] = None,
                 robust_slope: Optional[float] = None,
                 robust_tau: Optional[float] = None):
        """Initialize from two ChamferSideCfg-shaped dicts, one per direction.

        Args:
            source_to_target: dict matching ChamferSideCfg's fields (weight,
                base_sigma, field_name, field_source); {} / None uses its
                defaults (weight=1.0, base_sigma=0.1, no field).
            target_to_source: same, for the other direction.
            robust_kernel, robust_breakpoint, robust_slope, robust_tau: shared
                across both directions; see TwoSidedDataTerm / ROBUST_KINDS.
        """
        # Local import: config.py is deliberately torch-free at import time
        # (see DTYPES's docstring there), so importing it eagerly at module
        # scope here would pull torch into every caller of config.py that
        # never touches this data term.
        from src.resnet_lddmm.resnet_flow.config import ChamferSideCfg

        st = ChamferSideCfg.from_dict(source_to_target or {})
        ts = ChamferSideCfg.from_dict(target_to_source or {})
        super().__init__(source_to_target_weight=st.weight, target_to_source_weight=ts.weight,
                          robust_kernel=robust_kernel, robust_breakpoint=robust_breakpoint,
                          robust_slope=robust_slope, robust_tau=robust_tau)
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
    ) -> Tensor:
        """Compute the field-weighted, per-direction-sigma Chamfer distance.

        Args:
            pred: [B, N, 3] predicted (flowed source) positions
            target: [B, M, 3] target positions
            pred_w: [B, N] per-source-point precision multiplier (source_to_target
                direction), or None for a constant field. Supplied by
                UnidirectionalMappingError once a field is resolved -- see
                field_weights.resolve_point_weight_field.
            tgt_w: [B, M] per-target-point precision multiplier (target_to_source
                direction), or None.
            normals: ignored (kept for protocol compatibility)
            per_point: if True, return RAW (unweighted, unrobustified) per-point
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
            d1 = self._robustify(dist_sq.min(dim=2)[0])
            term1 = self._weighted_mean(d1, pred_w) / (2.0 * self.source_to_target_sigma ** 2)

        term2 = None
        if self.scores_target_to_source:
            d2 = self._robustify(dist_sq.min(dim=1)[0])
            term2 = self._weighted_mean(d2, tgt_w) / (2.0 * self.target_to_source_sigma ** 2)

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
    -- write_surface_normals.py prints the count so a target where it is NOT rare
    is visible before training rather than after.

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
                 robust_kernel: Optional[str] = None,
                 robust_breakpoint: Optional[float] = None,
                 robust_slope: Optional[float] = None,
                 robust_tau: Optional[float] = None):
        """Initialize from two ChamferSideCfg-shaped dicts, plus the blend.

        Args:
            source_to_target, target_to_source: see FieldWeightedCDData.
            plane_weight: 0.0 = plain point-to-point (this term becomes
                FieldWeightedCDData), 1.0 = pure point-to-plane. In between,
                the per-point penalty is the convex blend of the two.
            robust_kernel, robust_breakpoint, robust_slope, robust_tau: see
                TwoSidedDataTerm. Applied to the BLENDED penalty, so a
                robust_breakpoint keeps meaning "a distance, squared".
        """
        super().__init__(source_to_target=source_to_target,
                         target_to_source=target_to_source,
                         robust_kernel=robust_kernel, robust_breakpoint=robust_breakpoint,
                         robust_slope=robust_slope, robust_tau=robust_tau)
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
            per_point: if True, return RAW (unweighted, unrobustified, and
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
                "scripts/write_surface_normals.py -- io.load_shape picks that array "
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
            d1 = self._robustify((1.0 - a) * d_point + a * d_plane)
            term1 = self._weighted_mean(d1, pred_w) / (2.0 * self.source_to_target_sigma ** 2)

        term2 = None
        if self.scores_target_to_source:
            d_point, idx = dist_sq.min(dim=1)                     # [B, M]
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, 3)      # [B, M, 3]
            offset = pred.gather(1, gather_idx) - target          # [B, M, 3]
            d_plane = (offset * n).sum(-1).pow(2)                 # target's own normal
            d2 = self._robustify((1.0 - a) * d_point + a * d_plane)
            term2 = self._weighted_mean(d2, tgt_w) / (2.0 * self.target_to_source_sigma ** 2)

        return self._combine(term1, term2)


class NCDData(TwoSidedDataTerm):
    """Normal-aware Chamfer distance: penalizes surfaces with wrong orientation (T33).

    Combines point-to-point distance with normal alignment penalty.
    Encourages normals to be oriented away from the matching pred points.

    The normal penalty lives INSIDE the source->target direction -- it scores
    where each SOURCE point sits relative to its nearest target's surface -- so
    ``source_to_target_weight=0`` drops the orientation penalty along with the
    distance. A one-sided term that still reads normals is a point-to-plane
    target->source term, which is a different loss, not a setting of this one.
    """

    def __init__(self, normal_weight: float = 1.0,
                 source_to_target_weight: float = 1.0,
                 target_to_source_weight: float = 1.0,
                 robust_kernel: Optional[str] = None,
                 robust_breakpoint: Optional[float] = None,
                 robust_slope: Optional[float] = None,
                 robust_tau: Optional[float] = None):
        """Initialize NCD loss.

        Args:
            normal_weight: balance between geometric and normal penalties (default 1.0)
            source_to_target_weight: weight on D(phi(source) -> target); 0.0 disables it
            target_to_source_weight: weight on D(target -> phi(source)); 0.0 disables it
            robust_kernel, robust_breakpoint, robust_slope, robust_tau: see
                TwoSidedDataTerm. Applied to the GEOMETRIC distance only -- the
                normal-alignment penalty is a separate, unrelated quantity and is
                never robustified.
        """
        super().__init__(source_to_target_weight=source_to_target_weight,
                         target_to_source_weight=target_to_source_weight,
                         robust_kernel=robust_kernel, robust_breakpoint=robust_breakpoint,
                         robust_slope=robust_slope, robust_tau=robust_tau)
        self.normal_weight = normal_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
        per_point: bool = False,
    ) -> Tensor:
        """Compute normal-aware Chamfer distance.

        Args:
            pred: [B, N, 3] predicted point positions
            target: [B, M, 3] target point positions
            pred_w, tgt_w: ignored
            normals: [B, M, 3] target surface normals; if None, reduces to CDData
            per_point: if True, return per-point losses [B, M]; if False, return scalar

        Returns:
            Scalar NCD loss if per_point=False, else per-point losses [B, M]
        """
        B, N, D = pred.shape
        B_tgt, M, D = target.shape

        # Compute pairwise distances [B, N, M]
        dist_vec = pred.unsqueeze(2) - target.unsqueeze(1)  # [B, N, M, 3]
        dist_sq = (dist_vec ** 2).sum(dim=3)  # [B, N, M]

        # If no normals, fall back to standard Chamfer
        if normals is None:
            if per_point:
                return dist_sq.min(dim=1)[0]
            term1 = self._robustify(dist_sq.min(dim=2)[0]).mean() if self.scores_source_to_target else None
            term2 = self._robustify(dist_sq.min(dim=1)[0]).mean() if self.scores_target_to_source else None
            return self._combine(term1, term2)

        if per_point:
            # Per-point losses from target to pred (geometric only). Returned BEFORE
            # the normal penalty, which this path never used and which costs two
            # [B, N, 3] gathers to build.
            return dist_sq.min(dim=1)[0]  # [B, M]

        # Term 1: pred → target (point distance + normal penalty)
        term1 = None
        if self.scores_source_to_target:
            # Compute normal alignment penalty: penalizes normals pointing towards pred
            # For each pred point, find nearest target and its normal
            pred_to_tgt_indices = dist_sq.min(dim=2)[1]  # [B, N]
            pred_to_tgt_dists = dist_sq.min(dim=2)[0]  # [B, N]

            # Vectorized normal penalty: gather directions and normals, compute dot products in parallel
            # direction: [B, N, 3] - vector from target to pred
            b_idx = torch.arange(B, device=pred.device).unsqueeze(1)  # [B, 1]
            n_idx = torch.arange(N, device=pred.device).unsqueeze(0)  # [1, N]
            direction = dist_vec[b_idx, n_idx, pred_to_tgt_indices]  # [B, N, 3]

            # normal: [B, N, 3] - normals at nearest target points
            normal = normals[b_idx, pred_to_tgt_indices]  # [B, N, 3]

            # Normalize both
            dir_norm = torch.norm(direction, dim=-1, keepdim=True) + 1e-8  # [B, N, 1]
            normal_norm = torch.norm(normal, dim=-1, keepdim=True) + 1e-8  # [B, N, 1]
            direction_normalized = direction / dir_norm  # [B, N, 3]
            normal_normalized = normal / normal_norm  # [B, N, 3]

            # Compute dot products and penalty
            dot_products = (direction_normalized * normal_normalized).sum(dim=-1)  # [B, N]
            normal_penalty = torch.clamp(dot_products, min=0.0).mean()  # scalar

            # Geometric distance is robustified; the normal-alignment penalty is a
            # different quantity (a bounded [0,1] dot-product term already) and is
            # never subject to the distance kernel.
            term1 = self._robustify(pred_to_tgt_dists).mean() + self.normal_weight * normal_penalty

        # Term 2: target → pred (point distance only)
        term2 = self._robustify(dist_sq.min(dim=1)[0]).mean() if self.scores_target_to_source else None

        return self._combine(term1, term2)


class SinkhornData(DataTerm):
    """Sinkhorn optimal transport distance via geomloss (T34).

    Solves the optimal transport problem using Sinkhorn's algorithm.
    Recommended for articulated shapes (e.g., hands) where Chamfer can
    produce incorrect finger-to-finger matchings.

    Point weights are normalised to probability measures before being handed to
    geomloss, which solves BALANCED transport and therefore needs both sides to
    carry equal mass; see _as_measure.

    Lazy-loads geomloss: auto-decoder configs work even without it installed.
    """

    def __init__(self, p: float = 2, blur: float = 0.01, backend: str = "auto", **kwargs):
        """Initialize Sinkhorn loss.

        Args:
            p: Distance norm (default 2 for L2)
            blur: Sinkhorn blur radius (higher = smoother gradient, default 0.01)
            backend: geomloss backend ("auto", "keops", "torch", default "auto")
            **kwargs: ignored (compatibility with other data terms)
        """
        super().__init__()
        self.p = p
        self.blur = blur
        self.backend = backend
        self.loss_fn = None  # Lazy initialized

    def _get_loss_fn(self):
        """Lazy-load geomloss on first call."""
        if self.loss_fn is None:
            try:
                import geomloss
            except ImportError:
                raise ImportError(
                    "geomloss required for Sinkhorn distance. "
                    "Install with: pip install geomloss"
                )
            self.loss_fn = geomloss.SamplesLoss(
                "sinkhorn", p=self.p, blur=self.blur, backend=self.backend
            )
        return self.loss_fn

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute Sinkhorn optimal transport distance.

        Args:
            pred: [B, N, 3] predicted point positions
            target: [B, M, 3] target point positions
            pred_w: [B, N] weights for pred (optional, passed to geomloss)
            tgt_w: [B, M] weights for target (optional, passed to geomloss)
            normals: ignored (kept for protocol compatibility)

        Returns:
            Scalar Sinkhorn loss
        """
        loss_fn = self._get_loss_fn()

        B, N, D = pred.shape
        B_tgt, M, D = target.shape
        assert B == B_tgt, f"Batch size mismatch: {B} vs {B_tgt}"

        loss_total = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
        for b in range(B):
            p_b = pred[b]  # [N, 3]
            t_b = target[b]  # [M, 3]

            loss_b = loss_fn(
                self._as_measure(None if pred_w is None else pred_w[b], N, p_b),
                p_b,
                self._as_measure(None if tgt_w is None else tgt_w[b], M, t_b),
                t_b,
            )
            loss_total = loss_total + loss_b

        return loss_total / B

    @staticmethod
    def _as_measure(weights, count, like):
        """Per-point weights as a probability measure: non-negative, summing to one.

        geomloss solves BALANCED optimal transport, so both arguments must carry the
        same total mass. Passing raw ``ones(N)`` gives mass N rather than 1, which
        rescales the loss by that mass -- and is outright ill-posed once N != M, as
        happens whenever the two clouds are subsampled to different sizes.
        Normalising also makes the value independent of the point count, so a run is
        comparable across subsample_M settings.

        Args:
            weights: [K] per-point weights, or None for uniform
            count: K, the number of points
            like: tensor supplying device and dtype

        Returns:
            [K] non-negative weights summing to 1
        """
        uniform = torch.full((count,), 1.0 / count, device=like.device, dtype=like.dtype)
        if weights is None:
            return uniform

        w = weights.to(dtype=like.dtype).clamp_min(0)
        total = w.sum()
        # Degenerate input (all zero, or negative before clamping) would divide by ~0
        # and hand geomloss a non-finite measure; fall back rather than propagate it.
        if not torch.isfinite(total) or total <= 0:
            return uniform
        return w / total


class EMDData(DataTerm):
    """Earth Mover's Distance via Sinkhorn's algorithm (legacy alias for SinkhornData).

    Deprecated: use SinkhornData instead. Kept for backward compatibility.
    """

    def __init__(self, p: float = 2, blur: float = 0.01, backend: str = "auto"):
        """Initialize EMD loss (delegates to SinkhornData)."""
        super().__init__()
        self._sinkhorn = SinkhornData(p=p, blur=blur, backend=backend)

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute Earth Mover's Distance via Sinkhorn."""
        return self._sinkhorn(pred, target, pred_w=pred_w, tgt_w=tgt_w, normals=normals)


class FlowTerm(RegistrationLoss):
    """Regularizes velocity field properties during integration.

    Subclasses: IsometryLoss (shape-preservation), and future terms.
    """

    @abc.abstractmethod
    def forward(self, trajectory: Trajectory, field: VelocityField) -> Tensor:
        """Compute regularization loss on trajectory and field."""


class IsometryLoss(FlowTerm):
    """Penalizes non-isometric deformation, per Euler step, along the trajectory.

    The quantity being judged is the jacobian of the STEP MAP the integrator
    actually applies, ``F = I + dt J``, where ``J = dv/dx``. Not J itself: the
    deformation is ``x + dt v(x)``, so an isometric flow is one where F is a
    rotation, which for small dt means the SYMMETRIC part of J vanishes (the
    Killing-field condition). Penalizing J directly would have the objective
    backwards -- it is minimized by a large velocity gradient, and maximal at the
    identity deformation.

    Every penalty is divided by dt^2, which makes it a RATE and therefore
    independent of ``field.num_steps``: the same weight means the same thing when
    K changes.

    Supported loss types (rate form, up to O(dt) corrections):
    - "strain":     ||sym(J)||_F^2 -- the strain-rate tensor; the cheapest, and
                    the linearization of the one below (which it equals times 4,
                    as dt -> 0). EXACTLY zero for any rigid flow, which makes it
                    the default choice.
    - "orthogonal": ||F^T F - I||_F^2 / dt^2 -- the exact condition on the
                    discrete step map, but see the floor below.
    - "det":        ((det F - 1)/dt)^2 -- volume only; permits shear. Same floor.

    The floor: an explicit Euler step of a genuine ROTATION is not exactly
    isometric. For a rotation generator W, F^T F = I - dt^2 W^2 and det F =
    1 + dt^2 |w|^2, so "orthogonal" and "det" both charge a rigid flow for the
    integrator's error -- dt^2 ||W^T W||_F^2 and (dt |w|^2)^2 respectively. That
    is not always negligible: at dt = 0.1 and |w| = 3.7 it measures 3.92 and 1.96,
    which is MORE than those same terms report for a 50%-per-unit-time uniform
    expansion (3.15 and 2.48). Under those two settings a fast rigid rotation
    looks worse than a real distortion. "strain" has no such term.

    NO singular-value decomposition anywhere, deliberately. The natural way to
    write "distance to the closest rotation" is ||F - UV^T||_F, but the SVD
    backward carries 1/(s_i^2 - s_j^2) terms that are infinite when singular
    values repeat -- which is not a rare edge case here but the TARGET state: a
    rigid F has s = (1,1,1), and a zero-initialized field gives s = (0,0,0). That
    formulation returns a NaN gradient exactly where the loss is minimal, which
    then propagates into the field weights and surfaces one step later as
    "linalg.svd: input contained non-finite values". The forms above are
    polynomial in F, so they are smooth everywhere including at the optimum.
    """

    def __init__(self, loss_type: str = "strain", sample_points: int = 64,
                 detach_trajectory: bool = True):
        super().__init__()
        if loss_type not in ("strain", "det", "orthogonal"):
            raise ValueError(f"loss_type must be 'strain', 'det', or 'orthogonal', got {loss_type}")
        self.loss_type = loss_type
        self.sample_points = sample_points
        self.detach_trajectory = detach_trajectory

    def forward(self, trajectory: Trajectory, field: VelocityField,
                code: Optional[Tensor] = None) -> Tensor:
        """Compute isometry penalty using finite differences for jacobian.

        Samples a subset of points to reduce computational cost.

        ``code`` must be the SAME per-shape code the trajectory was integrated
        with. A conditioned field's first layer takes [3 + n_z] inputs, so
        omitting it is a shape error rather than a milder unconditioned
        approximation.

        ``detach_trajectory`` decides how far the gradient reaches. True (default)
        detaches both the trajectory and the code, leaving this a pure FIELD
        regularizer. False lets it reach whatever produced the trajectory -- which
        under ``pose_application="pre"`` includes the POSE, and is the only way the
        "least non-rigid flow" criterion can actually train it. Detaching here is
        not a free choice under that wiring: it silently severs the coupling while
        the loss VALUE still appears to respond to the pose.
        """
        points = trajectory.points.detach() if self.detach_trajectory else trajectory.points
        K = points.shape[0] - 1
        N = points.shape[2]
        if code is not None and self.detach_trajectory:
            code = code.detach()

        # Sample points for efficiency
        n_sample = min(self.sample_points, N)
        indices = torch.randperm(N, device=points.device)[:n_sample]

        all_losses = []

        # For each step
        for k in range(K):
            # [B, n_sample, 3]. The batch dim is KEPT: conditioning broadcasts a
            # per-shape code along it and cannot recover it once flattened away.
            x_sampled = points[k][:, indices, :]

            # Create a wrapper for this step
            def field_fn(x_):
                return field(x_, code=code, step=k)

            # Compute loss for all sampled points at once (vectorized)
            all_losses.append(self._step_penalty(field_fn, x_sampled, trajectory.dt).reshape(-1))

        # Compute final loss as mean of all point losses
        return torch.cat(all_losses).mean()

    def _step_penalty(self, field_fn, x: Tensor, dt: float) -> Tensor:
        """Per-point isometry penalty for ONE Euler step.

        Args:
            x: [B, M, 3] points
            dt: integrator step size, so ``F = I + dt J`` is the step map's jacobian
        Returns:
            [B, M] loss values
        """
        jac = self._jacobians(field_fn, x)                     # [B, M, 3, 3]
        eye = torch.eye(3, device=x.device, dtype=x.dtype)

        # Every statistic below is invariant to transposing F, which is what lets
        # ``_jacobians`` return the transposed layout it finds convenient:
        # ||F^T F - I||_F = ||F F^T - I||_F (same eigenvalues), det F = det F^T,
        # and sym(J) is unchanged by construction.
        if self.loss_type == "strain":
            # ||sym(J)||_F^2 -- no matmul, no dt: already a rate.
            sym = 0.5 * (jac + jac.transpose(-1, -2))
            return (sym ** 2).sum(dim=(-2, -1))

        F = eye + dt * jac
        if self.loss_type == "det":
            return ((torch.det(F) - 1.0) / dt) ** 2

        # "orthogonal": the right Cauchy-Green deviation C - I, which vanishes iff
        # F is orthogonal. One 3x3 matmul per point -- cheaper than an SVD, and
        # differentiable at the optimum, which the SVD form was not.
        C = F.transpose(-1, -2) @ F
        return ((C - eye) ** 2).sum(dim=(-2, -1)) / (dt ** 2)

    @staticmethod
    def _jacobians(field_fn, x: Tensor) -> Tensor:
        """Finite-difference Jacobians of the velocity field at every point.

        Forward differences: 4 field evaluations per point (one base, three
        perturbed), issued as a SINGLE batched call. At these point counts the
        field is launch-bound rather than FLOP-bound -- the [B, M, 3] base call
        costs about as much as the [B, 3M, 3] perturbed one -- so fusing the two
        into [B, 4M, 3] is worth close to a factor of 2 on the whole term.

        The step is sqrt(machine eps), the textbook optimum for a forward
        difference -- ~1.5e-8 in float64, ~3.4e-4 in float32 -- which assumes
        points of order 1, as the normalized frame gives.

        Args:
            x: [B, M, 3] points, the batch dim being the SHAPE index
        Returns:
            [B, M, 3, 3] jacobians. Rows hold the jacobian COLUMNS (dv/dx_j), i.e.
            the block is J^T; every statistic taken from it is transpose-invariant
            (see _step_penalty), so the layout does not matter.
        """
        eps = torch.finfo(x.dtype).eps ** 0.5
        B, M, _ = x.shape

        eye = torch.eye(3, device=x.device, dtype=x.dtype)     # [3, 3]
        # [B, M, 4, 3]: the base point followed by its three perturbations.
        stacked = torch.cat([x.unsqueeze(-2), x.unsqueeze(-2) + eps * eye], dim=-2)

        # Flattened to [B, 4M, 3], NOT [B*4M, 3]: the leading dim has to stay the
        # shape index or a conditioned field broadcasts the wrong code.
        v = field_fn(stacked.reshape(B, 4 * M, 3)).reshape(B, M, 4, 3)
        return (v[..., 1:, :] - v[..., :1, :]) / eps


