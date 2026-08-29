"""Mapping error strategies: unidirectional (forward-only) and bidirectional."""

from typing import Tuple, Optional
import torch
from torch import Tensor
from src.transforms.group_transforms import SE3_transform
import logging
from pathlib import Path
from src.vtk.create import create_polydata
from src.vtk.io import save_vtp

logger = logging.getLogger(__name__)


def apply_pose(points: Tensor, rotation: Optional[Tensor],
               translation: Optional[Tensor]) -> Tensor:
    """Apply an SE(3) pose to a batch of points: ``points @ R + t``.

    The ONE definition of the pipeline's pose convention -- ``points @ R``, not
    ``R @ points``. Everything that moves geometry by a pose goes through here
    (both mapping errors, and apply's client), because a handedness that differs
    between training and inference is invisible in the shapes' extents and only
    shows up as a mirrored result. See EncoderCodes.get_pose for why this
    convention, and not its transpose, is the one the encoder is built against.

    einsum rather than SE3_transform: it preserves gradients through the
    rotation, which is the only path the pose head has to train on.

    Args:
        points: [B, N, 3] batched points
        rotation: [B, 3, 3] or None
        translation: [B, 3] or None

    Returns:
        [B, N, 3] transformed points (the input tensor itself when both are None)
    """
    if rotation is not None:
        # 'bij,bjk->bik': for each batch b, [N,3] @ [3,3] = [N,3]
        points = torch.einsum('bij,bjk->bik', points, rotation)
    if translation is not None:
        points = points + translation.unsqueeze(1)  # [B,1,3] + [B,N,3]
    return points


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


class UnidirectionalMappingError:
    """Forward-only mapping error: D(φ(T), S) and kinetic energy of forward trajectory.

    Used for per-pair registration. Computes data term and kinetic
    energy over the forward trajectory only. Flow deforms template to match sample.

    Supports optional random subsampling for computational efficiency on high-density shapes:
    both template and sample are subsampled to subsample_n points before data-term computation,
    but flow is only computed on subsampled template. Tracks full and subsampled vertex counts.
    Optionally computes and stores full trajectory when save_full=True.
    """

    pose_call_counter = 0  # Counter for pose transformation debugging

    def __init__(self, subsample_n: Optional[int] = None, save_full: bool = False,
                 pose_application: str = "post"):
        """Initialize mapping error.

        Args:
            subsample_n: if not None, randomly subsample source to this many points before flow.
                        Target shape is kept at full resolution.
            save_full: if True, also compute and store trajectory on full source points
            pose_application: WHERE the encoder pose enters, relative to the flow.

                "post" (default, the original behaviour)
                    ``pred = phi(T) @ R``. The flow starts from the canonical
                    template, so the trajectory does not depend on R at all --
                    kinetic energy and the isometry penalty are then EXACTLY
                    constant in R (verified: identical to 6 decimals over a full
                    360-degree sweep, and autograd reports no path). Only the data
                    term can move the pose.

                "pre"
                    ``pred = phi(T @ R)``. The flow starts from the POSED template,
                    so the trajectory -- and therefore kinetic energy and isometry
                    -- become functions of R. This is what makes "the correct pose
                    is the one needing the least non-rigid flow" an objective the
                    pose can actually be trained on: kinetic charges the rigid part
                    of any compensation the flow attempts, isometry charges the
                    non-rigid part.

                    Caveat: the direct coupling relies on the field NOT being
                    equivariant. For an exactly equivariant field
                    phi(T @ R) = phi(T) @ R, so at fixed phi the flow cost is
                    rotation-invariant again and only the (much weaker) envelope
                    effect survives. Use with the MLP fields.
        """
        if pose_application not in ("post", "pre"):
            raise ValueError(
                f"pose_application must be 'post' or 'pre', got {pose_application!r}"
            )
        self.subsample_n = subsample_n
        self.save_full = save_full
        self.pose_application = pose_application
        self.last_full_vertices = None  # Track full vertex count for logging
        self.last_subsample_vertices = None
        self.last_fwd_traj = None  # Store trajectory to avoid recomputation
        self.last_fwd_traj_full = None  # Store full trajectory if save_full=True
        # Inputs configured loss terms read back (see losses/terms.py). The points
        # are the ones actually flowed; the pose is the EFFECTIVE one, after the
        # requires_grad filter below, so a term can never apply an element the data
        # term dropped.
        self.last_template_points = None
        self.last_sample_points = None
        self.last_effective_pose = (None, None)
        # Full-mesh per-vertex precision fields (see field_weights.py), gathered
        # at the SAME subsample indices as template_points/sample_points every
        # call and handed to the data term as pred_w/tgt_w. None by default --
        # only set via set_point_weight_fields, by runner._add_field_weighted_chamfer,
        # and only when data_name == "field_weighted_chamfer".
        self.template_field = None
        self.sample_field = None

    def set_point_weight_fields(self, template_field: Optional[Tensor] = None,
                                sample_field: Optional[Tensor] = None) -> None:
        """Attach per-vertex precision fields for pred_w/tgt_w, resolved once at build time.

        Args:
            template_field: [N_full] non-negative weight, one per TEMPLATE vertex
                (template.points.shape[1], the FULL mesh, before any subsampling),
                or None for "no field" (pred_w stays None every call).
            sample_field: [M_full] non-negative weight, one per SAMPLE/target
                vertex, or None.
        """
        self.template_field = template_field
        self.sample_field = sample_field

    @staticmethod
    def _gather_field(field: Optional[Tensor], idx: Optional[Tensor]) -> Optional[Tensor]:
        """field, restricted to this step's subsample draw, as a [1, K] weight.

        Mirrors _take_vertices's own idx==None passthrough, so a field lines up
        with template_points/sample_points whether or not subsampling is active.
        """
        if field is None:
            return None
        gathered = field if idx is None else field[idx]
        return gathered.unsqueeze(0)

    def __call__(self, flow, data_term, template, sample, code, encoder_pose=None) -> Tuple[Tensor, Tensor]:
        """Compute unidirectional mapping error.

        Args:
            flow: NeuralODEFlow instance
            data_term: DataTerm instance
            template: batch with .points [B,N,3] (canonical reference, where flow starts)
            sample: batch with .points [B,M,3] (augmented input, encoder input)
            code: [B, n_z] or None
            encoder_pose: Optional (rotation [B,3,3], translation [B,3]) tuple, or None

        Returns:
            (data_loss, kinetic_energy) tuple of scalars
        """
        # Track full template vertices
        self.last_full_vertices = template.points.shape[1]

        # One paired draw for both clouds (shared when they are corresponded), so
        # index-based terms stay valid. template.weights ("area") is still never
        # threaded through here -- the unidirectional data term compares against
        # the sample, never the template -- but self.template_field (a DIFFERENT
        # concept, set via set_point_weight_fields) IS gathered at this same draw,
        # below, so it lines up with template_points row for row.
        template_idx, sample_idx = _draw_pair_indices(
            template.points, sample.points, self.subsample_n
        )
        template_points, _ = _take_vertices(template.points, None, template_idx, "template")
        self.last_subsample_vertices = template_points.shape[1]
        pred_w = self._gather_field(self.template_field, template_idx)

        # Broadcast template to match code batch size (cohort: template [1,N,3], code [B,n_z])
        if template_points.shape[0] == 1 and code is not None and code.shape[0] > 1:
            template_points = template_points.expand(code.shape[0], -1, -1)

        rotation, translation = None, None
        has_pose = encoder_pose is not None and (encoder_pose[0] is not None
                                                 or encoder_pose[1] is not None)
        if has_pose:
            rotation, translation = encoder_pose
            # For SO(3)-only training, skip translation (doesn't have requires_grad)
            # to avoid breaking gradient flow. Translation will be enabled for SE(3) later.
            # NOTE: reading model semantics off an autograd flag is fragile -- whether
            # the pose is SO(3) or SE(3) is the pose head's to declare, not the loss's
            # to infer. Left as-is here; changing it is an encoder-side change.
            if translation is not None and not translation.requires_grad:
                translation = None

            if rotation is not None and not rotation.requires_grad:
                print(f"[GRADIENT_ERROR] rotation.requires_grad=False! This breaks gradient flow to encoder.")

        self.last_effective_pose = (rotation, translation)

        # "pre": the pose moves the template BEFORE the flow, which is what puts the
        # trajectory -- and so kinetic energy and isometry -- downstream of R.
        if has_pose and self.pose_application == "pre":
            template_points = self._apply_encoder_pose(template_points, rotation, translation)

        # Flow on (possibly subsampled) template points
        fwd = flow(template_points, code)
        self.last_fwd_traj = fwd  # Store for reuse (avoid double-computation in pair.py)
        # The points actually flowed, posed or not -- configured terms must reuse
        # these or their reference frame disagrees with the trajectory's.
        self.last_template_points = template_points
        pred = fwd.end

        # "post": the flow output is rotated into the augmented frame for comparison.
        if has_pose and self.pose_application == "post":
            pred = self._apply_encoder_pose(pred, rotation, translation)

        # Optionally compute full trajectory for export if save_full=True
        if self.save_full and self.subsample_n is not None and self.subsample_n > 0:
            template_full = template.points
            if template_full.shape[0] == 1 and code is not None and code.shape[0] > 1:
                template_full = template_full.expand(code.shape[0], -1, -1)
            self.last_fwd_traj_full = flow(template_full, code)
        else:
            self.last_fwd_traj_full = None

        # Points and weights share one draw, so tgt_w[i] stays the weight OF
        # sample_points[i].
        sample_points, sample_weights = _take_vertices(
            sample.points, sample.weights, sample_idx, "sample"
        )
        self.last_sample_points = sample_points  # what configured terms must reuse

        # sample.weights ("area") and self.sample_field (a point-weight field, see
        # field_weights.py) are independent signals; when both are present they
        # COMBINE multiplicatively rather than one silently overriding the other.
        sample_field_w = self._gather_field(self.sample_field, sample_idx)
        if sample_field_w is not None:
            tgt_w = sample_field_w if sample_weights is None else sample_weights * sample_field_w
        else:
            tgt_w = sample_weights

        data = data_term(pred, sample_points, pred_w=pred_w, tgt_w=tgt_w)
        kinetic = fwd.kinetic_energy()
        return data, kinetic

    @staticmethod
    def _log_pose_transform(label: str, points_before: Tensor, rotation: Optional[Tensor],
                            translation: Optional[Tensor], points_after: Tensor):
        """Log detailed information about pose transformation.

        Args:
            label: context label (e.g., "UniDirectional", "BiDirectional fwd", "BiDirectional bwd")
            points_before: [B, N, 3] before transformation
            rotation: [B, 3, 3] or None
            translation: [B, 3] or None
            points_after: [B, N, 3] after transformation
        """
        B, N, D = points_before.shape
        logger.info(f"\n{'='*70}")
        logger.info(f"[ENCODER_POSE] {label}")
        logger.info(f"{'='*70}")

        # Log input shape
        logger.info(f"Shape: batch={B}, points={N}, dims={D}")

        # Log rotation
        if rotation is not None:
            logger.info(f"\nRotation matrix [B,3,3]:")
            for b in range(min(B, 2)):  # Log first 2 batch elements
                logger.info(f"  Batch {b}:\n{rotation[b]}")
                # Check orthogonality
                R = rotation[b]
                det = torch.det(R)
                ortho_error = (R @ R.T - torch.eye(3, device=R.device)).norm()
                logger.info(f"    det(R) = {det.item():.6f}, orthogonality_error = {ortho_error.item():.6e}")
        else:
            logger.info("Rotation: None")

        # Log translation
        if translation is not None:
            logger.info(f"\nTranslation vector [B,3]:")
            for b in range(min(B, 2)):
                logger.info(f"  Batch {b}: {translation[b]}")
        else:
            logger.info("Translation: None")

        # Log point cloud statistics before
        logger.info(f"\nBefore transformation:")
        points_before_reshaped = points_before.view(-1, 3)
        logger.info(f"  Mean: {points_before_reshaped.mean(0)}")
        logger.info(f"  Std:  {points_before_reshaped.std(0)}")
        logger.info(f"  Min:  {points_before_reshaped.min(0)[0]}")
        logger.info(f"  Max:  {points_before_reshaped.max(0)[0]}")
        if N > 0:
            logger.info(f"  First 3 points (batch 0):\n{points_before[0, :3]}")

        # Log point cloud statistics after
        logger.info(f"\nAfter transformation:")
        points_after_reshaped = points_after.view(-1, 3)
        logger.info(f"  Mean: {points_after_reshaped.mean(0)}")
        logger.info(f"  Std:  {points_after_reshaped.std(0)}")
        logger.info(f"  Min:  {points_after_reshaped.min(0)[0]}")
        logger.info(f"  Max:  {points_after_reshaped.max(0)[0]}")
        if N > 0:
            logger.info(f"  First 3 points (batch 0):\n{points_after[0, :3]}")

        # Log change
        point_change = (points_after - points_before).view(-1, 3)
        logger.info(f"\nPoint change (after - before):")
        logger.info(f"  Mean: {point_change.mean(0)}")
        logger.info(f"  Std:  {point_change.std(0)}")
        logger.info(f"  Max:  {point_change.abs().max(0)[0]}")

        # Save VTP files for visualization
        try:
            call_counter = getattr(UnidirectionalMappingError, 'pose_call_counter', 0) if B > 0 else getattr(BidirectionalMappingError, 'pose_call_counter', 0)
            if B > 0:
                UnidirectionalMappingError.pose_call_counter = call_counter + 1
                call_idx = UnidirectionalMappingError.pose_call_counter
            else:
                BidirectionalMappingError.pose_call_counter = call_counter + 1
                call_idx = BidirectionalMappingError.pose_call_counter
            output_dir = Path("outputs/encoder_pose_debug") / f"call_{call_idx:06d}" / label.lower().replace(" ", "_")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save before and after for first 2 batch elements
            for b in range(min(B, 2)):
                batch_label = f"batch_{b}" if B > 1 else ""
                before_path = output_dir / f"before_{batch_label}.vtp"
                after_path = output_dir / f"after_{batch_label}.vtp"

                before_poly = create_polydata(points_before[b:b+1])
                after_poly = create_polydata(points_after[b:b+1])

                save_vtp(before_poly, str(before_path))
                save_vtp(after_poly, str(after_path))

            logger.info(f"VTP files saved to: {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to save VTP files: {e}")

        logger.info(f"{'='*70}\n")

    @staticmethod
    def _apply_encoder_pose(points: Tensor, rotation: Optional[Tensor],
                            translation: Optional[Tensor]) -> Tensor:
        """apply_pose, plus this class's diagnostic logging.

        Args:
            points: [B, N, 3] batched points
            rotation: [B, 3, 3] or None
            translation: [B, 3] or None

        Returns:
            [B, N, 3] transformed points
        """
        if rotation is None and translation is None:
            return points

        points_before = points.detach().clone() if logger.isEnabledFor(logging.INFO) else None
        points = apply_pose(points, rotation, translation)

        # Log if enabled
        if logger.isEnabledFor(logging.INFO) and points_before is not None:
            UnidirectionalMappingError._log_pose_transform(
                "UniDirectional", points_before, rotation, translation, points
            )

        return points


class BidirectionalMappingError:
    """Bidirectional mapping error: D(φ(T), S) + D(φ⁻¹(S), T).

    Used for cohort registration. Computes data term and kinetic
    energy summed over both forward and backward trajectories (AD-SVFD Eq. loss function).
    Forward: deform template to match sample. Backward: inverse flow from sample.

    Supports optional random subsampling for computational efficiency on high-density shapes:
    both template and sample are subsampled to subsample_n points before data-term computation,
    but flows are computed on subsampled points. Tracks full and subsampled vertex counts.
    Optionally computes and stores full trajectories when save_full=True.
    """

    pose_call_counter = 0  # Counter for pose transformation debugging

    def __init__(self, subsample_n: Optional[int] = None, save_full: bool = False):
        """Initialize mapping error.

        Args:
            subsample_n: if not None, randomly subsample source and target to this many points each call
                        before their respective flows.
            save_full: if True, also compute and store trajectories on full source/target points
        """
        self.subsample_n = subsample_n
        self.save_full = save_full
        self.last_full_source_vertices = None
        self.last_full_target_vertices = None
        self.last_subsample_vertices = None
        self.last_fwd_traj = None  # Store trajectory to avoid recomputation
        self.last_bwd_traj = None
        self.last_fwd_traj_full = None  # Store full trajectories if save_full=True
        self.last_bwd_traj_full = None
        # Inputs configured loss terms read back (see losses/terms.py)
        self.last_template_points = None
        self.last_sample_points = None
        self.last_effective_pose = (None, None)

    def __call__(self, flow, data_term, template, sample, code, encoder_pose=None) -> Tuple[Tensor, Tensor]:
        """Compute bidirectional mapping error.

        Args:
            flow: NeuralODEFlow instance (must support .inverse())
            data_term: DataTerm instance
            template: batch with .points [B,N,3] (canonical reference)
            sample: batch with .points [B,M,3] (augmented input)
            code: [B, n_z] or None
            encoder_pose: Optional (rotation [B,3,3], translation [B,3]) tuple, or None

        Returns:
            (data_loss, kinetic_energy) tuple where:
            - data_loss = D(φ(T), S) + D(φ⁻¹(S), T)
            - kinetic_energy = KE(forward) + KE(backward)
        """
        # Track full vertex counts
        self.last_full_source_vertices = template.points.shape[1]
        self.last_full_target_vertices = sample.points.shape[1]

        # Subsample template and sample before flow if configured. Each cloud gets
        # ONE draw, shared by its points and its weights: both appear as tgt_w below,
        # where entry i must be the weight of point i of the same cloud.
        template_idx, sample_idx = _draw_pair_indices(
            template.points, sample.points, self.subsample_n
        )
        template_points, template_weights = _take_vertices(
            template.points, template.weights, template_idx, "template"
        )
        sample_points, sample_weights = _take_vertices(
            sample.points, sample.weights, sample_idx, "sample"
        )
        self.last_sample_points = sample_points  # what configured terms must reuse

        if self.subsample_n is not None and self.subsample_n > 0:
            self.last_subsample_vertices = self.subsample_n
        else:
            self.last_subsample_vertices = None

        # Broadcast template/sample to match code batch size (cohort: one is [1,N,3], code is [B,n_z])
        if template_points.shape[0] == 1 and code is not None and code.shape[0] > 1:
            template_points = template_points.expand(code.shape[0], -1, -1)
        if sample_points.shape[0] == 1 and code is not None and code.shape[0] > 1:
            sample_points = sample_points.expand(code.shape[0], -1, -1)

        # Flow on (possibly subsampled) template and sample
        fwd = flow(template_points, code)
        bwd = flow.inverse(sample_points, code)
        self.last_fwd_traj = fwd  # Store for reuse (avoid double-computation in cohort.py)
        self.last_bwd_traj = bwd
        self.last_template_points = template_points  # what configured terms must reuse
        self.last_effective_pose = (None, None)

        # Optionally compute full trajectories for export if save_full=True
        if self.save_full and self.subsample_n is not None and self.subsample_n > 0:
            template_full = template.points
            sample_full = sample.points
            if template_full.shape[0] == 1 and code is not None and code.shape[0] > 1:
                template_full = template_full.expand(code.shape[0], -1, -1)
            if sample_full.shape[0] == 1 and code is not None and code.shape[0] > 1:
                sample_full = sample_full.expand(code.shape[0], -1, -1)
            self.last_fwd_traj_full = flow(template_full, code)
            self.last_bwd_traj_full = flow.inverse(sample_full, code)
        else:
            self.last_fwd_traj_full = None
            self.last_bwd_traj_full = None

        # Apply encoder pose if provided (transforms endpoints to augmented frame for comparison)
        fwd_end = fwd.end
        bwd_end = bwd.end
        if encoder_pose is not None and (encoder_pose[0] is not None or encoder_pose[1] is not None):
            rotation, translation = encoder_pose
            # For SO(3)-only training, skip translation to avoid breaking gradient flow
            if translation is not None and not translation.requires_grad:
                translation = None
            self.last_effective_pose = (rotation, translation)
            fwd_end_before = fwd_end.detach().clone() if logger.isEnabledFor(logging.INFO) else None
            bwd_end_before = bwd_end.detach().clone() if logger.isEnabledFor(logging.INFO) else None
            fwd_end = self._apply_encoder_pose(fwd_end, rotation, translation)
            bwd_end = self._apply_encoder_pose(bwd_end, rotation, translation)
            if logger.isEnabledFor(logging.INFO):
                BidirectionalMappingError._log_pose_transform("BiDirectional fwd", fwd_end_before, rotation, translation, fwd_end)
                BidirectionalMappingError._log_pose_transform("BiDirectional bwd", bwd_end_before, rotation, translation, bwd_end)

        # Data term: forward distance + backward distance (both on subsampled points)
        data = (
            data_term(fwd_end, sample_points, tgt_w=sample_weights)
            + data_term(bwd_end, template_points, tgt_w=template_weights)
        )

        # Kinetic energy: sum over both trajectories
        kinetic = fwd.kinetic_energy() + bwd.kinetic_energy()

        return data, kinetic

    @staticmethod
    def _log_pose_transform(label: str, points_before: Tensor, rotation: Optional[Tensor],
                            translation: Optional[Tensor], points_after: Tensor):
        """Log detailed information about pose transformation.

        Args:
            label: context label (e.g., "UniDirectional", "BiDirectional fwd", "BiDirectional bwd")
            points_before: [B, N, 3] before transformation
            rotation: [B, 3, 3] or None
            translation: [B, 3] or None
            points_after: [B, N, 3] after transformation
        """
        B, N, D = points_before.shape
        logger.info(f"\n{'='*70}")
        logger.info(f"[ENCODER_POSE] {label}")
        logger.info(f"{'='*70}")

        # Log input shape
        logger.info(f"Shape: batch={B}, points={N}, dims={D}")

        # Log rotation
        if rotation is not None:
            logger.info(f"\nRotation matrix [B,3,3]:")
            for b in range(min(B, 2)):  # Log first 2 batch elements
                logger.info(f"  Batch {b}:\n{rotation[b]}")
                # Check orthogonality
                R = rotation[b]
                det = torch.det(R)
                ortho_error = (R @ R.T - torch.eye(3, device=R.device)).norm()
                logger.info(f"    det(R) = {det.item():.6f}, orthogonality_error = {ortho_error.item():.6e}")
        else:
            logger.info("Rotation: None")

        # Log translation
        if translation is not None:
            logger.info(f"\nTranslation vector [B,3]:")
            for b in range(min(B, 2)):
                logger.info(f"  Batch {b}: {translation[b]}")
        else:
            logger.info("Translation: None")

        # Log point cloud statistics before
        logger.info(f"\nBefore transformation:")
        points_before_reshaped = points_before.view(-1, 3)
        logger.info(f"  Mean: {points_before_reshaped.mean(0)}")
        logger.info(f"  Std:  {points_before_reshaped.std(0)}")
        logger.info(f"  Min:  {points_before_reshaped.min(0)[0]}")
        logger.info(f"  Max:  {points_before_reshaped.max(0)[0]}")
        if N > 0:
            logger.info(f"  First 3 points (batch 0):\n{points_before[0, :3]}")

        # Log point cloud statistics after
        logger.info(f"\nAfter transformation:")
        points_after_reshaped = points_after.view(-1, 3)
        logger.info(f"  Mean: {points_after_reshaped.mean(0)}")
        logger.info(f"  Std:  {points_after_reshaped.std(0)}")
        logger.info(f"  Min:  {points_after_reshaped.min(0)[0]}")
        logger.info(f"  Max:  {points_after_reshaped.max(0)[0]}")
        if N > 0:
            logger.info(f"  First 3 points (batch 0):\n{points_after[0, :3]}")

        # Log change
        point_change = (points_after - points_before).view(-1, 3)
        logger.info(f"\nPoint change (after - before):")
        logger.info(f"  Mean: {point_change.mean(0)}")
        logger.info(f"  Std:  {point_change.std(0)}")
        logger.info(f"  Max:  {point_change.abs().max(0)[0]}")

        # Save VTP files for visualization
        try:
            call_counter = getattr(UnidirectionalMappingError, 'pose_call_counter', 0) if B > 0 else getattr(BidirectionalMappingError, 'pose_call_counter', 0)
            if B > 0:
                UnidirectionalMappingError.pose_call_counter = call_counter + 1
                call_idx = UnidirectionalMappingError.pose_call_counter
            else:
                BidirectionalMappingError.pose_call_counter = call_counter + 1
                call_idx = BidirectionalMappingError.pose_call_counter
            output_dir = Path("outputs/encoder_pose_debug") / f"call_{call_idx:06d}" / label.lower().replace(" ", "_")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save before and after for first 2 batch elements
            for b in range(min(B, 2)):
                batch_label = f"batch_{b}" if B > 1 else ""
                before_path = output_dir / f"before_{batch_label}.vtp"
                after_path = output_dir / f"after_{batch_label}.vtp"

                before_poly = create_polydata(points_before[b:b+1])
                after_poly = create_polydata(points_after[b:b+1])

                save_vtp(before_poly, str(before_path))
                save_vtp(after_poly, str(after_path))

            logger.info(f"VTP files saved to: {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to save VTP files: {e}")

        logger.info(f"{'='*70}\n")

    @staticmethod
    def _apply_encoder_pose(points: Tensor, rotation: Optional[Tensor],
                            translation: Optional[Tensor]) -> Tensor:
        """apply_pose, kept as a method for call-site symmetry with the forward case.

        Args:
            points: [B, N, 3] batched points
            rotation: [B, 3, 3] or None
            translation: [B, 3] or None

        Returns:
            [B, N, 3] transformed points
        """
        return apply_pose(points, rotation, translation)
