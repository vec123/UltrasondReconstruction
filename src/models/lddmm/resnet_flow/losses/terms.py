"""Config-selected loss terms.

A term is any module taking a :class:`LossContext` and returning a scalar Tensor,
or ``None`` when it does not apply to the current step or mode. ``None`` is already
a first-class skip in ``LossComposer.compute``, so mode dependence (bidirectional
only, encoder only, ...) costs zero branches in the registration steppers.

Terms are selected by ``loss.terms`` in the YAML and resolved through the
``loss_term`` Registry category, so adding one is a new class plus a registry line
-- never an edit to the steppers.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class LossContext:
    """One training step's inputs, as seen by every configured term.

    Every field defaults to None so that adding a field here never breaks a term
    that does not read it.

    ``template_points`` is the load-bearing one: it is the tensor that was actually
    fed to the flow, AFTER subsampling and the cohort broadcast. Any term that
    instead re-derived its own subset would lose index correspondence with
    ``fwd_traj`` -- silently, since the shapes would still line up.
    """

    flow: Any = None                                   # NeuralODEFlow | IdentityFlowWrapper
    code: Optional[Tensor] = None                      # [B, n_z] or None
    template_points: Optional[Tensor] = None           # [B, N, 3] exactly as flowed
    sample_points: Optional[Tensor] = None             # [B, M, 3] exactly as compared against
    fwd_traj: Any = None                               # Trajectory
    bwd_traj: Any = None                               # Trajectory, bidirectional only
    pred: Optional[Tensor] = None                      # posed endpoint the data term saw
    encoder_pose: Tuple[Optional[Tensor], Optional[Tensor]] = (None, None)
    augmentation_pose: Tuple[Optional[Tensor], Optional[Tensor]] = (None, None)  # ground truth
    data_term: Any = None                              # configured DataTerm, reusable
    code_source: Any = None
    template: Any = None                               # unsubsampled batch (faces, weights)
    sample: Any = None                                 # augmented batch
    # [n] draw into the FULL template that produced template_points, or None when
    # the whole template was flowed. What a term holding a per-template-vertex
    # tensor gathers with, to stay row-for-row with fwd_traj.
    template_idx: Optional[Tensor] = None

    @property
    def field(self):
        """The velocity field, through whichever flow wrapper is in play."""
        return None if self.flow is None else self.flow.field


def apply_pose(points: Tensor, rotation: Optional[Tensor],
               translation: Optional[Tensor]) -> Tensor:
    """Apply a group element to batched points: ``x·R + t``.

    Deliberately mirrors ``mapping_error._apply_encoder_pose`` including its
    right-multiplication convention. A term that applied ``R·x`` instead would
    disagree with the data term about which element the encoder predicted, and the
    two objectives would pull against each other.

    Args:
        points: [B, N, 3]
        rotation: [B, 3, 3] or None
        translation: [B, 3] or None

    Returns:
        [B, N, 3] transformed points (the input tensor when both are None)
    """
    if rotation is not None:
        points = torch.einsum('bij,bjk->bik', points, rotation)
    if translation is not None:
        points = points + translation.unsqueeze(1)
    return points


# Relative singular-value gap below which the SVD backward stops being trustworthy.
_PROCRUSTES_GAP_FLOOR = 1e-3
_PROCRUSTES_WARN_LIMIT = 5
_procrustes_warnings = 0


def _raise_on_nonfinite_grad(grad: Tensor) -> Tensor:
    """Backward hook: turn a NaN/Inf gradient into a message that names the cause.

    Without this the non-finite value propagates until the composer's finite-loss guard
    trips, reporting only "non-finite loss" with no hint that a degenerate Procrustes
    SVD produced it.
    """
    if not torch.isfinite(grad).all():
        raise FloatingPointError(
            "non-finite gradient through procrustes_rotation: the SVD backward carries "
            "1/(s_i - s_j) terms and has blown up. The deformed shape is (near) "
            "rotationally degenerate, so the optimal rotation is undefined rather than "
            "merely hard to compute. Lower flow_rotation_penalty_weight, or check "
            "whether the deformation has collapsed toward a sphere/plane/line."
        )
    return grad


def _check_procrustes_conditioning(singular_values: Tensor) -> None:
    """Warn loudly when the SVD backward is about to become ill-conditioned.

    ``torch.linalg.svd``'s gradient contains ``1/(s_i - s_j)`` terms, so it diverges as
    two singular values approach each other. Geometrically that is the shape becoming
    rotationally symmetric about an axis (a spheroid), planar, or spherical -- cases
    where the optimal rotation is genuinely undefined. The FORWARD value stays finite
    and perfectly plausible throughout, which is exactly why this needs saying out loud
    rather than being left to surface as a mystery NaN later.

    Args:
        singular_values: [B, 3] descending singular values of the cross-covariance
    """
    global _procrustes_warnings
    if _procrustes_warnings >= _PROCRUSTES_WARN_LIMIT:
        return

    scale = singular_values[..., 0].clamp_min(1e-12)
    gaps = torch.stack([
        (singular_values[..., 0] - singular_values[..., 1]) / scale,
        (singular_values[..., 1] - singular_values[..., 2]) / scale,
    ], dim=-1)                                            # [B, 2] relative gaps
    worst, index = gaps.min().detach(), gaps.min(dim=-1).values.argmin()

    if worst < _PROCRUSTES_GAP_FLOOR:
        _procrustes_warnings += 1
        tail = (" (further warnings suppressed)"
                if _procrustes_warnings == _PROCRUSTES_WARN_LIMIT else "")
        print(
            f"[PROCRUSTES_ILL_CONDITIONED] relative singular-value gap "
            f"{worst.item():.3e} < {_PROCRUSTES_GAP_FLOOR:.0e}; singular values "
            f"{[round(v, 6) for v in singular_values[index].tolist()]}. The rotation is "
            f"near-degenerate, so its gradient is unreliable even though the value looks "
            f"fine. Check for a deformation collapsing toward a sphere, plane or line."
            f"{tail}"
        )


def procrustes_rotation(X: Tensor, Y: Tensor) -> Tensor:
    """Optimal rotation R minimising ``||X @ R - Y||`` over SO(3), in closed form.

    Orthogonal Procrustes / Kabsch, in the RIGHT-multiplication convention used by
    :func:`apply_pose`. Both clouds are mean-centred first, so the answer is the
    rotation alone and never absorbs a translation.

    Requires X and Y to be in vertex correspondence, row for row.

    Args:
        X: [B, N, 3] source cloud
        Y: [B, N, 3] target cloud, corresponded row-for-row with X

    Returns:
        [B, 3, 3] proper rotations (det = +1; reflections excluded)
    """
    Xc = X - X.mean(dim=1, keepdim=True)
    Yc = Y - Y.mean(dim=1, keepdim=True)
    H = Xc.transpose(1, 2) @ Yc                                   # [B, 3, 3]
    U, singular_values, Vh = torch.linalg.svd(H)
    _check_procrustes_conditioning(singular_values)
    V = Vh.transpose(1, 2)
    # det = -1 would be a reflection, not a rotation; flip the least-significant
    # singular direction to stay inside SO(3).
    det = torch.det(U @ V.transpose(1, 2))
    ones = torch.ones_like(det)
    D = torch.diag_embed(torch.stack([ones, ones, det], dim=-1))
    rotation = U @ D @ V.transpose(1, 2)
    if rotation.requires_grad:
        rotation.register_hook(_raise_on_nonfinite_grad)
    return rotation


class FlowRotationPenalty(nn.Module):
    """||R_flow - I||_F^2, where R_flow is the net rigid rotation the deformation applies.

    Fixes the pose/deformation GAUGE. The data term only ever sees the product
    ``phi(T) @ R_hat``, so for any rotation A the pair ``(phi(T)@A, A^-1 @ R_hat)``
    predicts exactly the same points -- a 3-parameter family of equally optimal
    answers, of which training picks one arbitrarily. The symptom is
    ``pose_supervision_loss`` pinned at ``||A^-1 - I||_F^2`` (~6 for a random A, since
    R_aug cancels out of that expression) while the data term falls happily.

    This is the only term that can see the split, because it looks at ``phi(T)``
    alone rather than at the product. Driving R_flow to identity selects A = I, which
    means the deformation performs no net rotation and R_hat carries the whole pose --
    the factorisation implied by "the template defines the canonical frame".

    Correspondence, which Procrustes needs, holds STRUCTURALLY here: the integrator is
    elementwise, so ``phi(T)[i]`` is by construction the image of ``T[i]``. (Between
    the deformed template and a SAMPLE it would not hold, which is a different and
    much less safe use of the same algorithm.)

    Translation is not penalised: Procrustes mean-centres both clouds, and a net
    translation of the flow is genuinely needed to match shapes whose centroids
    differ -- it is not gauge, because R_hat cannot absorb it.

    Weighting: the data term is EXACTLY flat along the gauge, so any weight > 0
    selects A = I with nothing opposing it; the weight only sets how fast. Keep it
    small so it stays a tie-breaker. Genuine articulated motion, whose net Procrustes
    rotation is not pose, sits on a CURVED direction and is biased by a factor
    ``D'' / (D'' + 4w)`` -- negligible while ``w << D''/4``. The empirical check is to
    raise the weight 10x: if the data term does not move, the penalty is acting only
    on flat directions and cannot be distorting geometry.
    """

    def forward(self, ctx: LossContext) -> Optional[Tensor]:
        """Compute ||R_flow - I||_F^2 for the current deformation.

        Args:
            ctx: the step's LossContext

        Returns:
            Scalar tensor, or None when there is no trajectory to measure
        """
        if ctx.fwd_traj is None or ctx.template_points is None:
            return None

        rotation = procrustes_rotation(ctx.template_points, ctx.fwd_traj.end)
        identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
        return (rotation - identity).pow(2).sum(dim=(-2, -1)).mean()


class PoseSupervisionLoss(nn.Module):
    """Supervise the predicted pose against the group element the augmenter drew.

    ``SO3Augmentation`` samples the rotation that produces the encoder's input, so
    that rotation IS the answer the pose head should give: the pipeline compares
    ``φ(T)·R̂`` against ``S·R_aug``, which agrees exactly when ``R̂ = R_aug``.

    This is the only pose signal here that is genuinely supervised. It needs no
    vertex correspondence, cannot be satisfied by collapsing anything, and does not
    depend on the flow being any good — which is what lets the pose head and the
    field train at the same time instead of waiting on each other. Chamfer, by
    contrast, reaches the pose head through a nearest-neighbour assignment that
    flips as the rotation turns, giving the bumpy landscape that traps R̂.

    Chordal distance ``‖R̂ − R_aug‖_F²`` rather than the geodesic angle: monotone in
    that angle, but smooth everywhere, whereas ``arccos`` blows up at 0 and π.

    Returns None whenever there is nothing to compare — no predicted rotation, or an
    augmentation that draws no element (``kind: none``), so the term simply vanishes
    from the breakdown instead of erroring.

    Note this supervision exists only because the pose is SYNTHETIC. It teaches the
    encoder to undo the augmenter's rotations; it is not available for real pose
    variation in held-out data.
    """

    def __init__(self, translation: bool = False):
        """Initialize pose supervision.

        Args:
            translation: also supervise the predicted translation against the drawn
                one. Off by default: with SO(3) augmentation there is no translation
                to recover, and the encoder's translation currently arrives with
                requires_grad=False, so supervising it would contribute no gradient.
        """
        super().__init__()
        self.translation = translation

    def forward(self, ctx: LossContext) -> Optional[Tensor]:
        """Compute ‖R̂ − R_aug‖_F² (plus the translation term if enabled).

        Args:
            ctx: the step's LossContext

        Returns:
            Scalar tensor, or None when either pose is unavailable
        """
        pred_rotation, pred_translation = ctx.encoder_pose
        true_rotation, true_translation = ctx.augmentation_pose

        terms = []
        if pred_rotation is not None and true_rotation is not None:
            # The target generated the input; it is data, never a variable.
            terms.append(
                (pred_rotation - true_rotation.detach()).pow(2).sum(dim=(-2, -1)).mean()
            )

        if self.translation and pred_translation is not None and true_translation is not None:
            terms.append(
                (pred_translation - true_translation.detach()).pow(2).sum(dim=-1).mean()
            )

        if not terms:
            return None
        return sum(terms)


class EquivariantDeformationLoss(nn.Module):
    """L2 between ``φ(g·T)`` and ``g·φ(T)``: the deformation must commute with the pose.

    The deformation of a posed template should equal the posed deformation of the
    template. Both sides descend from the SAME ``ctx.template_points`` and the
    integrator is elementwise, so the two clouds are in exact pointwise
    correspondence -- which is why this is a plain L2 and not a Chamfer distance.

    Costs one extra forward integration per step, on the subsampled template, so
    ``loss.subsample_M`` sets its price directly.

    Expected values, useful as sanity checks: exactly 0 under
    ``freeze_flow_at_identity`` (identity commutes with everything), ~0 for an e3nn
    equivariant field (where the property holds by construction), and strictly
    positive for an MLP field, where it is a genuine soft constraint.
    """

    def __init__(self, field_only: bool = True, translation: bool = True):
        """Initialize the equivariance term.

        Args:
            field_only: route the gradient to the VELOCITY FIELD alone. Both the
                group element g and the latent code z are treated as constants, so
                nothing reaches the encoder — neither its pose head via g nor its
                latent path via z. This costs one extra integration: the forward
                trajectory the data term computed carries encoder gradient through
                z, so it cannot be reused for the right-hand side and is recomputed
                against the detached code. The loss VALUE is identical either way;
                only gradient routing and cost change.

                False keeps the cheap path (reuse that trajectory, detach nothing),
                letting the constraint reach the field, the latent, and the pose
                head. Beware: the term is trivially zero at g = identity, so it can
                then be minimised by collapsing the predicted rotation rather than
                by making the flow equivariant.
            translation: include the predicted translation in g. Rotation-only
                (False) isolates the SO(3) part of the constraint.
        """
        super().__init__()
        self.field_only = field_only
        self.translation = translation

    def forward(self, ctx: LossContext) -> Optional[Tensor]:
        """Compute ‖φ(g·T) − g·φ(T)‖², or None when there is no pose to test.

        Args:
            ctx: the step's LossContext

        Returns:
            Scalar tensor, or None when no group element is available (no encoder
            pose predicted, so nothing to be equivariant to)
        """
        rotation, translation = ctx.encoder_pose
        if not self.translation:
            translation = None
        if rotation is None and translation is None:
            return None

        code = ctx.code
        if self.field_only:
            rotation = None if rotation is None else rotation.detach()
            translation = None if translation is None else translation.detach()
            code = None if code is None else code.detach()
            # ctx.fwd_traj was built with the LIVE code, so reusing it would leak
            # gradient into the encoder. Recompute against the detached code: same
            # numbers, but a graph that stops at the field.
            deformed = ctx.flow(ctx.template_points, code).end
        else:
            deformed = ctx.fwd_traj.end

        # Flow the POSED template. Same tensor the forward trajectory started from,
        # so lhs[i] and rhs[i] describe the same vertex.
        posed_template = apply_pose(ctx.template_points, rotation, translation)
        lhs = ctx.flow(posed_template, code).end              # φ(g·T)

        rhs = apply_pose(deformed, rotation, translation)     # g·φ(T)

        return (lhs - rhs).pow(2).sum(-1).mean()


class LandmarkGuidanceLoss(nn.Module):
    """Anisotropic weighted L2 between the deformed template's landmarks and annotated targets.

    ``template_points``/``target_points``/``target_precision`` are resolved once,
    at build time, by ``landmarks.build_landmark_term_kwargs`` -- see that module
    for how the template side (argmax of each ``heatmap_<label>`` field) and the
    target side (mean + per-axis precision across annotation files) are derived.
    This module only runs the trained flow on that small, fixed point set and
    compares, exactly like ``EquivariantDeformationLoss`` runs the flow on a
    point set of its own rather than reusing ``ctx.template_points`` -- the
    landmarks are not a subset of the (sub-sampled) surface cloud, so they need
    their own forward integration to stay in exact correspondence.

    ``target_precision`` carries ``1 / (2 * std**2)`` PER AXIS already (see
    ``build_landmark_term_kwargs``), so this is the same "drop the Gaussian NLL's
    normalising constants, keep the precision weighting" convention as
    ``data weight = 1/(2*sigma**2)`` and ``IsometryLoss`` elsewhere in this file --
    an axis every annotator placed identically is enforced tightly, an axis they
    disagreed on is enforced loosely.

    No learnable parameters: the three tensors are registered as buffers, not
    ``nn.Parameter``s, so this term only ever pulls the FLOW (and, through it,
    the code/pose) toward the landmarks, never the other way around. Consistent
    with every other ``loss_term`` here carrying no state of its own -- none of
    them appear in the stepper's ``state_dict``/``load_state_dict``.
    """

    def __init__(self, template_points: Tensor, target_points: Tensor,
                 target_precision: Tensor, labels: Optional[list] = None):
        """Initialize with the resolved landmark tensors.

        Args:
            template_points: [1, L, 3] template landmark positions (normalised
                frame), one per label, in the SAME order as ``target_points``.
            target_points: [1, L, 3] target landmark positions (normalised
                frame, i.e. already passed through the run's FrameTransform).
            target_precision: [1, L, 3] per-axis precision, ``1/(2*std**2)``.
            labels: the L label names, same order as the L axis. Not used by
                ``forward``; kept for logging/debugging.
        """
        super().__init__()
        self.register_buffer("template_points", template_points)
        self.register_buffer("target_points", target_points)
        self.register_buffer("target_precision", target_precision)
        self.labels = labels

    def forward(self, ctx: LossContext) -> Optional[Tensor]:
        """Compute the precision-weighted squared error on the deformed landmarks.

        Args:
            ctx: the step's LossContext

        Returns:
            Scalar tensor, or None when there is no flow to deform the
            landmarks with.
        """
        if ctx.flow is None:
            return None

        points = self.template_points
        if ctx.code is not None and ctx.code.shape[0] > 1:
            points = points.expand(ctx.code.shape[0], -1, -1)

        deformed = ctx.flow(points, ctx.code).end                    # [B, L, 3]
        target = self.target_points.expand_as(deformed)
        precision = self.target_precision.expand_as(deformed)
        return (precision * (deformed - target).pow(2)).sum(-1).mean()
