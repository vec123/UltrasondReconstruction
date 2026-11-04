"""A trained run's output folder, loaded for analysis.

A run folder holds config.json (the ExperimentCfg it was trained with) and
checkpoints/ (final.pt, step_<N>.pt). The inspection scripts read the flow, the
learned codes and the cohort from it, and decode codes into shapes.
"""

import os

import torch

from src.models.lddmm.resnet_flow import registrations  # noqa: F401  (registers the components build_flow needs)
from src.models.lddmm.resnet_flow.apply.client import load_flow
from src.models.lddmm.resnet_flow.apply.config import load_experiment_config
from src.models.lddmm.resnet_flow.callbacks.checkpoints import load_checkpoint, resolve_checkpoint
from src.models.lddmm.resnet_flow.runner import load_cohort


def load_run_config(run):
    """The ExperimentCfg a run was trained with (its config.json)."""
    return load_experiment_config(os.path.join(run, "config.json"))


def load_run(run, step=None, device="cpu"):
    """(train_cfg, flow on device in eval mode, codes [M, n_z], cohort_norm, template_norm, transform)
    of a cohort run, from final.pt (step None) or step_<step>.pt."""
    cfg = load_run_config(run)
    checkpoint = load_checkpoint(resolve_checkpoint(os.path.join(run, "checkpoints"), step))
    flow = load_flow(cfg, checkpoint).to(device).eval()
    _, cohort_norm, template_norm, transform = load_cohort(cfg)
    return cfg, flow, checkpoint["codes"]["codes.weight"].float(), cohort_norm, template_norm, transform


@torch.no_grad()
def decode(flow, template_norm, transform, codes, device, batch):
    """Yield (start index, world points [b, N, 3] on device) of phi_z(template) for each code."""
    template = template_norm.points.to(device)
    center, scale = transform.center.to(device), transform.scale.to(device)
    for i in range(0, len(codes), batch):
        z = codes[i:i + batch].to(device)
        q = flow(template.expand(len(z), -1, -1), z).points[-1]
        yield i, q * scale + center
