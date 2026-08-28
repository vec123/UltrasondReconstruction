"""Encoder pose diagnostics callback for monitoring learned SE(3) transformations."""

import os
import torch
import json
import numpy as np
from dataclasses import dataclass

from src.learning.callbacks.base import Callback
from src.resnet_lddmm.io import FrameTransform, Shape
from src.vtk.create import create_polydata
from src.vtk.io import save_vtp
from src.vtk.fields import add_point_field


@dataclass
class PoseMetrics:
    """Metrics comparing learned vs ground truth pose."""
    learned_rotation: np.ndarray  # [B, 3, 3]
    learned_translation: np.ndarray  # [B, 3]
    ground_truth_rotation: np.ndarray  # [B, 3, 3] or None
    ground_truth_translation: np.ndarray  # [B, 3] or None

    rotation_error_frobenius: list  # ||R_learned - R_gt||_F per sample
    rotation_error_angles: list  # angle difference in degrees per sample
    translation_error: list  # ||t_learned - t_gt|| per sample
    rotation_singular_values: list  # singular values of R for invertibility check
    det_rotation: list  # determinant (should be ≈1 for SO(3))


def extract_augmentation_pose_so3(augmentation_cfg, sample_points_original, sample_points_augmented):
    """Estimate ground truth pose by comparing original and augmented point clouds.

    For SO(3) augmentation (rotation only), estimates R by solving: points_aug ≈ R @ points_orig

    Args:
        augmentation_cfg: AugmentationCfg with kind='so3'
        sample_points_original: [B, N, 3] original point cloud
        sample_points_augmented: [B, N, 3] augmented point cloud

    Returns:
        (rotation [B,3,3], translation [B,3]) estimated from augmentation
    """
    if augmentation_cfg.kind != "so3":
        return None, None

    B, N, _ = sample_points_original.shape

    # Estimate rotation via SVD (Procrustes)
    # Solve: min ||R @ X - Y||_F where X=orig, Y=aug
    rotations = []
    for b in range(B):
        X = sample_points_original[b].T  # [3, N]
        Y = sample_points_augmented[b].T  # [3, N]

        H = X @ Y.T  # [3, 3] covariance
        U, _, Vt = torch.linalg.svd(H)
        R = (U @ Vt).T  # Transpose to get rotation matrix

        # Ensure proper rotation (det=1, not reflection)
        if torch.det(R) < 0:
            Vt[-1, :] *= -1
            R = (U @ Vt).T

        rotations.append(R)

    rotation = torch.stack(rotations)  # [B, 3, 3]
    translation = torch.zeros(B, 3, device=sample_points_original.device)  # No translation in SO(3)

    return rotation, translation


def rotation_error_frobenius(R_learned, R_gt):
    """Frobenius norm error between two rotation matrices."""
    if R_gt is None:
        return None
    return torch.norm(R_learned - R_gt, p='fro').item()


def rotation_error_angle(R_learned, R_gt):
    """Angular error in degrees between two rotations.

    angle = arccos((trace(R_learned^T @ R_gt) - 1) / 2)
    """
    if R_gt is None:
        return None

    # R_learned^T @ R_gt
    rel_rot = R_learned.T @ R_gt

    # Clamp to avoid numerical issues with arccos
    trace = torch.trace(rel_rot)
    trace = torch.clamp((trace - 1) / 2, -1, 1)
    angle_rad = torch.acos(trace)
    angle_deg = torch.rad2deg(angle_rad).item()

    return angle_deg


def translation_error(t_learned, t_gt):
    """L2 norm error between two translation vectors."""
    if t_gt is None or t_learned is None:
        return None
    return torch.norm(t_learned - t_gt).item()


class EncoderPoseLogger(Callback):
    """Monitor encoder pose extraction with ground truth comparison and numerical diagnostics.

    For each logging step:
    1. Extracts ground truth pose from augmentation (SO(3) only)
    2. Gets learned pose from encoder.get_pose()
    3. Computes trajectory before/after pose application
    4. Compares learned vs ground truth
    5. Checks numerical stability (singular values, determinant)
    6. Exports point clouds and trajectories to VTP
    7. Logs metrics to JSON
    """

    def __init__(self, every_n_steps=100, augmentation_cfg=None):
        super().__init__(every_n_steps)
        self.transform = None  # Set by runner after build()
        self.augmentation_cfg = augmentation_cfg

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Log encoder pose diagnostics if due."""
        if not self._due(step):
            return

        if pred is None or not hasattr(ctx.stepper, 'use_encoder_pose'):
            return

        if not ctx.stepper.use_encoder_pose:
            return

        try:
            self._log_pose_diagnostics(ctx, step, metrics, batch, pred)
        except Exception as e:
            print(f"[EncoderPoseLogger] Step {step}: {e}")
            import traceback
            traceback.print_exc()

    def _log_pose_diagnostics(self, ctx, step, metrics, batch, pred_traj):
        """Compute and log pose diagnostics."""
        stepper = ctx.stepper
        template, sample = batch

        # Get augmented sample (stored by stepper during forward pass)
        augmented_sample = None
        if hasattr(stepper, 'augmented_sample'):
            augmented_sample = stepper.augmented_sample
        elif hasattr(stepper, 'augmented_sample_batch'):
            augmented_sample = stepper.augmented_sample_batch

        if augmented_sample is None:
            return

        # Extract ground truth pose from augmentation (SO(3) only)
        if self.augmentation_cfg is None:
            gt_rotation, gt_translation = None, None
        else:
            gt_rotation, gt_translation = extract_augmentation_pose_so3(
                self.augmentation_cfg, sample.points, augmented_sample.points
            )

        # Get learned pose from encoder
        if not hasattr(stepper.code_source, 'get_pose'):
            return

        learned_rotation, learned_translation = stepper.code_source.get_pose()

        if learned_rotation is None:
            return

        # Constrain to SO(3) if augmentation is so3 (rotation-only)
        if stepper.augmentation_kind == 'so3':
            print(f"[EncoderPoseLogger] Step {step}: Constraining translation to None (SO3)")
            learned_translation = None

        B = learned_rotation.shape[0]

        # Compute pose error metrics
        pose_metrics = self._compute_pose_metrics(
            learned_rotation, learned_translation,
            gt_rotation, gt_translation
        )

        # Get flow trajectory (before pose application)
        if hasattr(stepper.mapping_error, 'last_fwd_traj'):
            fwd_traj_before_pose = stepper.mapping_error.last_fwd_traj
        else:
            fwd_traj_before_pose = None

        # Export diagnostics
        out_dir = f"{ctx.log_dir}/encoder_pose/step_{step}"
        os.makedirs(out_dir, exist_ok=True)

        self._export_pose_visualizations(
            out_dir, step, B,
            template, sample, augmented_sample,
            learned_rotation, learned_translation,
            gt_rotation, gt_translation,
            fwd_traj_before_pose, pred_traj
        )

        self._export_pose_metrics(out_dir, pose_metrics)

        # Log to metrics dict for training dashboard
        for b in range(B):
            if pose_metrics.rotation_error_angles[b] is not None:
                metrics[f"pose/rot_error_deg_shape{b}"] = pose_metrics.rotation_error_angles[b]
            if pose_metrics.det_rotation[b] is not None:
                metrics[f"pose/det_rot_shape{b}"] = pose_metrics.det_rotation[b]

        print(f"[EncoderPoseLogger] Step {step}: Exported pose diagnostics to {out_dir}")

    def _compute_pose_metrics(self, learned_rot, learned_trans, gt_rot, gt_trans):
        """Compute comprehensive pose error metrics."""
        B = learned_rot.shape[0]

        metrics = PoseMetrics(
            learned_rotation=learned_rot.detach().cpu().numpy(),
            learned_translation=learned_trans.detach().cpu().numpy() if learned_trans is not None else None,
            ground_truth_rotation=gt_rot.detach().cpu().numpy() if gt_rot is not None else None,
            ground_truth_translation=gt_trans.detach().cpu().numpy() if gt_trans is not None else None,
            rotation_error_frobenius=[],
            rotation_error_angles=[],
            translation_error=[],
            rotation_singular_values=[],
            det_rotation=[]
        )

        for b in range(B):
            R_learned = learned_rot[b]
            t_learned = learned_trans[b] if learned_trans is not None else None

            R_gt = gt_rot[b] if gt_rot is not None else None
            t_gt = gt_trans[b] if gt_trans is not None else None

            # Frobenius error
            f_error = rotation_error_frobenius(R_learned, R_gt)
            metrics.rotation_error_frobenius.append(f_error)

            # Angular error
            a_error = rotation_error_angle(R_learned, R_gt)
            metrics.rotation_error_angles.append(a_error)

            # Translation error
            t_error = translation_error(t_learned, t_gt)
            metrics.translation_error.append(t_error)

            # Singular values (invertibility check)
            try:
                _, s, _ = torch.linalg.svd(R_learned)
                metrics.rotation_singular_values.append(s.detach().cpu().numpy())
            except:
                metrics.rotation_singular_values.append(None)

            # Determinant (should be 1 for SO(3))
            try:
                det = torch.det(R_learned).item()
                metrics.det_rotation.append(det)
            except:
                metrics.det_rotation.append(None)

        return metrics

    def _export_pose_visualizations(self, out_dir, step, B,
                                     template, sample, augmented_sample,
                                     learned_rot, learned_trans,
                                     gt_rot, gt_trans,
                                     fwd_traj_before, fwd_traj_after):
        """Export point clouds and trajectories with pose applied/unapplied."""

        # Use default transform if not set
        if self.transform is None:
            transform = FrameTransform(center=torch.zeros(3), scale=torch.tensor(1.0))
        else:
            transform = self.transform

        os.makedirs(out_dir, exist_ok=True)

        # Export template (canonical, unchanging)
        template_points = template.points[0, :, :]
        template_world = transform.invert(template_points)
        template_poly = create_polydata(template_world, faces=None)
        save_vtp(template_poly, f"{out_dir}/template.vtp", binary=True)

        # Export sample at different stages
        sample_original = sample.points[0, :, :]
        sample_augmented = augmented_sample.points[0, :, :]

        # Unnormalized in world space
        sample_orig_world = transform.invert(sample_original)
        sample_aug_world = transform.invert(sample_augmented)

        sample_orig_poly = create_polydata(sample_orig_world, faces=None)
        sample_aug_poly = create_polydata(sample_aug_world, faces=None)

        save_vtp(sample_orig_poly, f"{out_dir}/sample_original.vtp", binary=True)
        save_vtp(sample_aug_poly, f"{out_dir}/sample_augmented.vtp", binary=True)

        # Export trajectory endpoints
        if fwd_traj_before is not None and fwd_traj_after is not None:
            # Before pose application
            pred_before = fwd_traj_before.end[0, :, :]  # [N, 3]
            pred_before_world = transform.invert(pred_before)
            pred_before_poly = create_polydata(pred_before_world, faces=None)
            save_vtp(pred_before_poly, f"{out_dir}/trajectory_end_before_pose.vtp", binary=True)

            # After pose application
            pred_after = fwd_traj_after.end[0, :, :]
            pred_after_world = transform.invert(pred_after)
            pred_after_poly = create_polydata(pred_after_world, faces=None)
            save_vtp(pred_after_poly, f"{out_dir}/trajectory_end_after_pose.vtp", binary=True)

    def _export_pose_metrics(self, out_dir, metrics):
        """Export pose metrics to JSON."""
        B = len(metrics.rotation_error_angles)

        data = {
            "num_samples": B,
            "rotation_error_frobenius": metrics.rotation_error_frobenius,
            "rotation_error_angles_deg": metrics.rotation_error_angles,
            "translation_error": metrics.translation_error,
            "det_rotation": metrics.det_rotation,
            "rotation_singular_values": [
                sv.tolist() if sv is not None else None
                for sv in metrics.rotation_singular_values
            ],
            "learned_rotation": [
                metrics.learned_rotation[b].tolist()
                for b in range(B)
            ],
            "ground_truth_rotation": [
                metrics.ground_truth_rotation[b].tolist() if metrics.ground_truth_rotation is not None else None
                for b in range(B)
            ],
        }

        with open(f"{out_dir}/pose_metrics.json", "w") as f:
            json.dump(data, f, indent=2)
