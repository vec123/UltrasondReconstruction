"""PoseLogger: the encoder's predicted pose against the augmentation's ground truth."""

import os
import json
import numpy as np

from src.resnet_lddmm.learning.callbacks.base import Callback


class PoseLogger(Callback):
    """Log encoder-predicted poses (rotation + translation) to JSON.

    Records the estimated rotation matrix and translation vector from the
    encoder at specified cadence. Useful for monitoring pose learning in
    encoder-based registration with use_encoder_pose=true.

    For SO(3) augmentation, translation is not logged since target has zero
    translation. For SE(3) augmentation, both rotation and translation logged.

    Output format:
    {
        "step": int,
        "augmentation": "so3" or "se3",
        "rotation": [[3x3 matrix as list of lists]],
        "translation": [3D vector as list] (only for SE(3)),
        "rotation_det": float (should be ~1 for proper rotation),
        "orthogonality_error": float (should be ~0)
    }
    """

    def __init__(self, every_n_steps=50, augmentation_kind="se3"):
        super().__init__(every_n_steps)
        self.log_dir = None
        self.augmentation_kind = augmentation_kind  # "so3" or "se3"

    def on_train_start(self, ctx):
        """Create pose_logs directory at start of training."""
        self.log_dir = os.path.join(ctx.log_dir, "pose_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[PoseLogger] Logging poses to {self.log_dir}")

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Log pose if due."""
        if not self._due(step) or self.log_dir is None:
            return

        stepper = ctx.stepper
        code_source = stepper.code_source if hasattr(stepper, 'code_source') else None

        # Check if encoder has pose
        if code_source is None or not hasattr(code_source, 'get_pose'):
            return

        rotation, translation = code_source.get_pose()

        # Skip if pose is None
        if rotation is None and translation is None:
            return

        try:
            pose_data = {
                'step': step,
                'augmentation': self.augmentation_kind,
                'rotation': None,
                'translation': None,
                'rotation_det': None,
                'rotation_frobenius_norm': None,
                'v1': None,
                'v2': None,
                'v1_norm': None,
                'v2_norm': None,
            }

            # Log rotation matrix if present
            if rotation is not None:
                R = rotation.detach().cpu().numpy()
                # Handle batch dimension: if [B, 3, 3], take first sample [0, :, :]
                if R.ndim == 3:
                    R = R[0]  # Take first batch element

                pose_data['rotation'] = R.tolist()
                det = np.linalg.det(R)
                pose_data['rotation_det'] = float(det) if np.isscalar(det) else float(det.item())

                # Frobenius norm of (R^T R - I) to check orthonormality
                orthogonality_error = np.linalg.norm(R.T @ R - np.eye(3))
                pose_data['orthogonality_error'] = float(orthogonality_error)

            # Log translation vector ONLY for SE(3) augmentation (SO(3) has zero translation)
            if translation is not None and self.augmentation_kind == "se3":
                t = translation.detach().cpu().numpy()
                # Handle batch dimension: if [B, 3], take first sample [0, :]
                if t.ndim == 2:
                    t = t[0]
                pose_data['translation'] = t.tolist()

            # Log the raw pose vectors (v1, v2) that build the rotation
            if code_source is not None and hasattr(code_source, '_last') and code_source._last is not None:
                enc_output = code_source._last
                if hasattr(enc_output, 'aux') and enc_output.aux and 'v1' in enc_output.aux:
                    v1 = enc_output.aux['v1'].detach().cpu().numpy()
                    v2 = enc_output.aux['v2'].detach().cpu().numpy()
                    # Handle batch dimension
                    if v1.ndim == 2:
                        v1 = v1[0]
                    if v2.ndim == 2:
                        v2 = v2[0]
                    pose_data['v1'] = v1.tolist()
                    pose_data['v2'] = v2.tolist()
                    pose_data['v1_norm'] = float(np.linalg.norm(v1))
                    pose_data['v2_norm'] = float(np.linalg.norm(v2))

            # Save to JSON
            pose_path = os.path.join(self.log_dir, f"step_{step:06d}.json")
            with open(pose_path, 'w') as f:
                json.dump(pose_data, f, indent=2)

            # Append to history CSV
            self._append_pose_history(step, pose_data)

        except Exception as e:
            print(f"[PoseLogger] Failed to log pose at step {step}: {e}")
            import traceback
            traceback.print_exc()

    def _append_pose_history(self, step, pose_data):
        """Append pose data to CSV history for easy plotting."""
        import csv

        csv_path = os.path.join(self.log_dir, "pose_history.csv")
        file_exists = os.path.exists(csv_path)

        # Build fieldnames based on augmentation type
        fieldnames = ['step']

        # Only include translation columns for SE(3) augmentation
        if self.augmentation_kind == "se3":
            fieldnames.extend(['translation_x', 'translation_y', 'translation_z'])

        fieldnames.extend(['rotation_det', 'orthogonality_error', 'v1_norm', 'v2_norm'])
        for i in range(3):
            for j in range(3):
                fieldnames.append(f'rotation_{i}{j}')

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            # Build row
            row = {
                'step': step,
                'rotation_det': pose_data['rotation_det'],
                'orthogonality_error': pose_data['orthogonality_error'],
                'v1_norm': pose_data['v1_norm'],
                'v2_norm': pose_data['v2_norm'],
            }

            # Add translation entries ONLY for SE(3)
            if self.augmentation_kind == "se3":
                row['translation_x'] = None
                row['translation_y'] = None
                row['translation_z'] = None
                if pose_data['translation'] and len(pose_data['translation']) >= 3:
                    row['translation_x'] = pose_data['translation'][0]
                    row['translation_y'] = pose_data['translation'][1]
                    row['translation_z'] = pose_data['translation'][2]

            # Add rotation matrix entries
            if pose_data['rotation']:
                R = pose_data['rotation']
                for i in range(3):
                    for j in range(3):
                        row[f'rotation_{i}{j}'] = R[i][j]

            writer.writerow(row)
