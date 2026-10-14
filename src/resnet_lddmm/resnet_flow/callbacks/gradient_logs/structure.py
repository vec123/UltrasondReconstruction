"""NetworkStructureInspector: the trained networks' layers and parameter counts, written once."""

import os

from src.resnet_lddmm.learning.callbacks.base import Callback


class NetworkStructureInspector(Callback):
    """Inspect and log the actual network structure once at the start."""

    def __init__(self):
        super().__init__(every_n_steps=0)  # Only run once

    def on_train_start(self, ctx):
        """Print and save detailed network structure."""
        stepper = ctx.stepper
        log_dir = os.path.join(ctx.log_dir, "grad_logs")
        os.makedirs(log_dir, exist_ok=True)

        output = []
        output.append("="*80)
        output.append("NETWORK STRUCTURE INSPECTION")
        output.append("="*80)

        # Flow parameters
        output.append("\n[FLOW PARAMETERS]")
        flow_total = 0
        for name, param in stepper.flow.named_parameters():
            size = param.numel()
            flow_total += size
            output.append(f"  {name}: {param.shape} = {size:,} params")
        output.append(f"  TOTAL FLOW: {flow_total:,} params\n")

        # Code source parameters
        output.append("[CODE SOURCE PARAMETERS]")
        code_total = 0
        for name, param in stepper.code_source.named_parameters():
            size = param.numel()
            code_total += size
            output.append(f"  {name}: {param.shape} = {size:,} params")
        output.append(f"  TOTAL CODE SOURCE: {code_total:,} params\n")

        # Encoder parameters (if exists)
        if hasattr(stepper.code_source, 'encoder'):
            output.append("[ENCODER PARAMETERS]")
            encoder_total = 0
            for name, param in stepper.code_source.encoder.named_parameters():
                size = param.numel()
                encoder_total += size
                output.append(f"  {name}: {param.shape} = {size:,} params")
            output.append(f"  TOTAL ENCODER: {encoder_total:,} params\n")

        output.append("="*80 + "\n")

        # Save to file
        output_text = "\n".join(output)
        save_path = os.path.join(log_dir, "network_structure.txt")
        with open(save_path, 'w') as f:
            f.write(output_text)
