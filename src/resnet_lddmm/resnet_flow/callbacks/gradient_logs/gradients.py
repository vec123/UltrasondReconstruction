"""Gradient norms during training: GradientMonitor (in memory) and GradientLogger (grad_logs/)."""

import os
import torch

from src.resnet_lddmm.learning.callbacks.base import Callback


class GradientMonitor(Callback):
    """Monitor and log gradient statistics during training.

    Tracks gradient norms, means, and detects NaN/Inf for each parameter group:
    encoder, flow, and code source. Useful for debugging gradient flow issues.
    """

    def __init__(self, every_n_steps=50):
        super().__init__(every_n_steps)

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Log gradient statistics if due."""
        if not self._due(step):
            return

        stepper = ctx.stepper
        if not hasattr(stepper, '_log_gradient_stats'):
            return

        # Call the stepper's gradient logging method
        stepper._log_gradient_stats()

        # Also store metrics for later analysis
        self._record_gradient_metrics(stepper, metrics, step)

    def _record_gradient_metrics(self, stepper, metrics, step):
        """Store gradient metrics for monitoring."""
        # Encoder stats
        if hasattr(stepper.code_source, 'encoder'):
            encoder_grad_norm = self._compute_grad_norm(stepper.code_source.encoder.parameters())
            metrics[f'grad_norm/encoder'] = encoder_grad_norm

        # Flow stats
        flow_grad_norm = self._compute_grad_norm(stepper.flow.parameters())
        metrics[f'grad_norm/flow'] = flow_grad_norm

        # Code source stats
        code_grad_norm = self._compute_grad_norm(stepper.code_source.parameters())
        metrics[f'grad_norm/code_source'] = code_grad_norm

    @staticmethod
    def _compute_grad_norm(parameters):
        """Compute total gradient norm for parameter group."""
        total_norm = 0.0
        for param in parameters:
            if param.grad is not None:
                total_norm += param.grad.data.norm().item() ** 2
        return total_norm ** 0.5

class GradientLogger(Callback):
    """Log detailed gradient information to grad_logs folder for analysis.

    Tracks gradient flow, parameter updates, and per-layer statistics.
    Saves to CSV files for post-training analysis.
    """

    def __init__(self, every_n_steps=50):
        super().__init__(every_n_steps)
        self.log_dir = None

    def on_train_start(self, ctx):
        """Create grad_logs directory at start of training."""
        self.log_dir = os.path.join(ctx.log_dir, "grad_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[GradientLogger] Logging to {self.log_dir}")

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Log gradient stats if due."""
        if not self._due(step) or self.log_dir is None:
            return

        stepper = ctx.stepper
        grad_info = self._compute_gradient_info(stepper)
        self._write_log(step, grad_info)

    def _compute_gradient_info(self, stepper):
        """Compute detailed gradient information."""
        info = {
            'flow': self._analyze_param_group(stepper.flow.parameters()),
            'code_source': self._analyze_param_group(stepper.code_source.parameters()),
        }

        # For EncoderCodes, encoder params are identical to code_source params, so skip duplication
        # Only log encoder separately if it has non-encoder params (like embeddings in AutoDecoderCodes)
        if hasattr(stepper.code_source, 'encoder'):
            encoder_params = list(stepper.code_source.encoder.parameters())
            code_source_params = list(stepper.code_source.parameters())

            # Only add encoder if it's not identical to code_source (i.e., has additional non-encoder params)
            if len(encoder_params) < len(code_source_params):
                info['encoder'] = self._analyze_param_group(stepper.code_source.encoder.parameters())

        return info

    @staticmethod
    def _analyze_param_group(parameters):
        """Analyze gradients for a parameter group."""
        params = list(parameters)

        grad_norms = []
        param_norms = []
        has_grad_count = 0
        zero_grad_count = 0
        nan_count = 0
        inf_count = 0
        total_param_count = 0
        num_tensors = len(params)

        for param in params:
            total_param_count += param.numel()
            param_norms.append(param.data.norm().item())

            if param.grad is None:
                continue

            has_grad_count += 1
            grad = param.grad.data
            grad_norm = grad.norm().item()
            grad_norms.append(grad_norm)

            if (grad == 0).all():
                zero_grad_count += 1
            if torch.isnan(grad).any():
                nan_count += 1
            if torch.isinf(grad).any():
                inf_count += 1

        total_param_norm = sum(p**2 for p in param_norms) ** 0.5
        total_grad_norm = sum(g**2 for g in grad_norms) ** 0.5

        update_ratio = total_grad_norm / total_param_norm if total_param_norm > 0 else 0

        return {
            'total_params': total_param_count,
            'num_tensors': num_tensors,
            'has_grad': has_grad_count,
            'zero_grad': zero_grad_count,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'grad_norm': total_grad_norm,
            'param_norm': total_param_norm,
            'update_ratio': update_ratio,
            'avg_grad_norm': sum(grad_norms) / len(grad_norms) if grad_norms else 0,
        }

    def _write_log(self, step, grad_info):
        """Write gradient info to CSV file."""
        import csv

        csv_path = os.path.join(self.log_dir, "gradient_history.csv")

        # Check if file exists to write header
        file_exists = os.path.exists(csv_path)

        with open(csv_path, 'a', newline='') as f:
            fieldnames = ['step']

            # Add fields for each component
            for component in grad_info.keys():
                for key in grad_info[component].keys():
                    fieldnames.append(f'{component}_{key}')

            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            # Build row
            row = {'step': step}
            for component, stats in grad_info.items():
                for key, value in stats.items():
                    row[f'{component}_{key}'] = value

            writer.writerow(row)

        # Also write a text summary for this step
        self._write_text_summary(step, grad_info)

    def _write_text_summary(self, step, grad_info):
        """Write human-readable summary for this step."""
        summary_path = os.path.join(self.log_dir, f"step_{step:06d}.txt")

        with open(summary_path, 'w') as f:
            f.write(f"Gradient Analysis - Step {step}\n")
            f.write("=" * 60 + "\n\n")

            for component, stats in grad_info.items():
                f.write(f"[{component.upper()}]\n")
                f.write(f"  Total parameters: {stats['total_params']:,}\n")
                f.write(f"  Parameter tensors: {stats['num_tensors']}\n")
                f.write(f"  Tensors with gradients: {stats['has_grad']}/{stats['num_tensors']}\n")
                f.write(f"  Tensors with zero gradients: {stats['zero_grad']}\n")
                f.write(f"  NaN detected: {stats['nan_count']}\n")
                f.write(f"  Inf detected: {stats['inf_count']}\n")
                f.write(f"  Gradient norm: {stats['grad_norm']:.4e}\n")
                f.write(f"  Parameter norm: {stats['param_norm']:.4e}\n")
                f.write(f"  Update ratio (grad/param): {stats['update_ratio']:.4e}\n")
                f.write(f"  Avg gradient: {stats['avg_grad_norm']:.4e}\n")
                f.write("\n")
