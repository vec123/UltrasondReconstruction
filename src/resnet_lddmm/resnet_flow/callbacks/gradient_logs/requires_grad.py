"""RequiresGradMonitor: which parameters are trainable, logged to requires_grad_logs/."""

import os

from src.resnet_lddmm.learning.callbacks.base import Callback


class RequiresGradMonitor(Callback):
    """Monitor requires_grad status of encoder pose tensors.

    Ensures rotation matrix has requires_grad=True so gradients flow back to encoder.
    Saves a detailed report of all tensor requires_grad status to log directory.
    """

    def __init__(self, every_n_steps=100):
        super().__init__(every_n_steps)
        self.log_dir = None

    def on_train_start(self, ctx):
        """Create log directory."""
        self.log_dir = os.path.join(ctx.log_dir, "requires_grad_logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Check and log requires_grad status if due."""
        if not self._due(step):
            return

        stepper = ctx.stepper
        code_source = stepper.code_source if hasattr(stepper, 'code_source') else None
        flow = stepper.flow if hasattr(stepper, 'flow') else None

        report = {}

        # Check encoder pose requires_grad
        if code_source is not None and hasattr(code_source, 'get_pose'):
            rotation, translation = code_source.get_pose()
            report['encoder_pose'] = {
                'rotation_requires_grad': rotation.requires_grad if rotation is not None else 'N/A',
                'rotation_grad_fn': str(rotation.grad_fn) if rotation is not None else 'N/A',
                'translation_requires_grad': translation.requires_grad if translation is not None else 'N/A',
            }

            # ERROR: rotation must require grad!
            if rotation is not None and not rotation.requires_grad:
                print(f"[REQUIRES_GRAD_ERROR] Step {step}: rotation.requires_grad=False!")
                print(f"  This breaks gradient flow to encoder!")
                report['encoder_pose']['ERROR'] = 'rotation.requires_grad=False'

        # Check all encoder parameters
        if code_source is not None and hasattr(code_source, 'encoder'):
            encoder_requires_grad = []
            for name, param in code_source.encoder.named_parameters():
                encoder_requires_grad.append({
                    'name': name,
                    'requires_grad': param.requires_grad,
                    'shape': str(param.shape),
                })
            report['encoder_params'] = encoder_requires_grad

        # Check flow parameters
        if flow is not None:
            flow_requires_grad = []
            for name, param in flow.named_parameters():
                flow_requires_grad.append({
                    'name': name,
                    'requires_grad': param.requires_grad,
                    'shape': str(param.shape),
                })
            report['flow_params'] = flow_requires_grad

        # Save report
        self._save_report(step, report)

    def _save_report(self, step, report):
        """Save requires_grad status report to file."""
        report_path = os.path.join(self.log_dir, f"step_{step:06d}.txt")

        with open(report_path, 'w') as f:
            f.write(f"Requires Grad Status - Step {step}\n")
            f.write("=" * 80 + "\n\n")

            # Encoder pose
            if 'encoder_pose' in report:
                f.write("[ENCODER POSE]\n")
                pose_info = report['encoder_pose']
                for key, value in pose_info.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            # Encoder parameters
            if 'encoder_params' in report:
                f.write("[ENCODER PARAMETERS]\n")
                encoder_params = report['encoder_params']
                requires_grad_count = sum(1 for p in encoder_params if p['requires_grad'])
                f.write(f"  Total: {len(encoder_params)}, with requires_grad: {requires_grad_count}\n\n")
                for p in encoder_params:
                    f.write(f"  {p['name']:<50} requires_grad={p['requires_grad']:<5} shape={p['shape']}\n")
                f.write("\n")

            # Flow parameters
            if 'flow_params' in report:
                f.write("[FLOW PARAMETERS]\n")
                flow_params = report['flow_params']
                requires_grad_count = sum(1 for p in flow_params if p['requires_grad'])
                f.write(f"  Total: {len(flow_params)}, with requires_grad: {requires_grad_count}\n\n")
                for p in flow_params:
                    f.write(f"  {p['name']:<50} requires_grad={p['requires_grad']:<5} shape={p['shape']}\n")
                f.write("\n")
