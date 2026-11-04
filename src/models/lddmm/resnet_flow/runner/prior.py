"""Residual-on-a-prior setup: reads the prior run's config, loads its field and codes, checks the frame."""

import os
import json
import torch
from src.models.lddmm.resnet_flow.callbacks.checkpoints import (
    checkpoint_step,
    load_checkpoint,
    resolve_checkpoint,
)
from src.models.lddmm.resnet_flow.config import ExperimentCfg, MODE_FIELD_KINDS

from .common import code_size


def load_prior_config(cfg):
    """cfg.field.prior.run -> that run's ExperimentCfg, checked against cfg.

    The prior field is rebuilt from the run's own config.json, so the architecture
    its weights load into is the one they were trained in.

    Raises:
        FileNotFoundError: the run has no config.json
        ValueError: the prior cannot be combined with cfg (see the messages)
    """
    run = cfg.field.prior["run"]
    path = os.path.join(run, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"field.prior.run: no config.json in {run} (is it a run's output_dir?)")
    with open(path) as f:
        data = json.load(f)
    data.pop("mode", None)      # provenance stamp of _dump_config, not an ExperimentCfg key
    prior_cfg = ExperimentCfg.from_dict(data)

    problems = []
    if prior_cfg.field.kind not in MODE_FIELD_KINDS:
        problems.append(f"its field kind {prior_cfg.field.kind!r} is not pointwise {MODE_FIELD_KINDS}")
    if prior_cfg.field.modes is not None or prior_cfg.field.prior is not None:
        problems.append("its field has modes or a prior of its own")
    if prior_cfg.freeze_flow_at_identity:
        problems.append("its flow is frozen at the identity")
    for key in ("centering", "dtype"):
        if getattr(prior_cfg, key) != getattr(cfg, key):
            problems.append(f"its {key} is {getattr(prior_cfg, key)!r}, this run's {getattr(cfg, key)!r}")
    if prior_cfg.field.num_steps != cfg.field.num_steps:
        problems.append(f"it integrates {prior_cfg.field.num_steps} steps, this run {cfg.field.num_steps}")
    prior_n_z = code_size(prior_cfg)
    if prior_n_z > code_size(cfg):
        problems.append(f"its field reads {prior_n_z} code entries, this run's code has "
                        f"{code_size(cfg)}: set code.n_z to {prior_n_z} + the residual's entries")
    if problems:
        raise ValueError(f"field.prior.run {run} cannot be the prior of this run: " + "; ".join(problems))
    return prior_cfg


def _load_prior_field(cfg, flow):
    """Load field.prior's weights from its run's checkpoint into the flow's PriorResidualField.

    Training only: apply restores the prior with the rest of the run's checkpoint.

    Raises:
        RuntimeError: from load_state_dict, if the checkpoint does not fit the
            architecture of the run's config.json
    """
    if cfg.field.prior is None:
        return
    run = cfg.field.prior["run"]
    ref, state = _prior_checkpoint(cfg)
    prior_state = {k[len("field."):]: v for k, v in state["flow"].items() if k.startswith("field.")}
    flow.field.prior.load_state_dict(prior_state, strict=True)
    field = flow.field
    count = sum(p.numel() for p in field.prior.parameters())
    trainable = sum(p.numel() for p in field.network.parameters())
    print(f"[prior] frozen field of {run} ({ref.path.name}, step {checkpoint_step(state)}): "
          f"{count} parameters, code entries 0..{field.prior_n_z - 1}; residual network: {trainable} "
          f"parameters, code entries {field.prior_n_z}..{code_size(cfg) - 1}; kinetic price of the "
          f"prior {cfg.field.prior['discount']:g} x kinetic_weight")


def _prior_checkpoint(cfg):
    """field.prior's checkpoint: (CheckpointRef, its state dict)."""
    ref = resolve_checkpoint(os.path.join(cfg.field.prior["run"], "checkpoints"), step=cfg.field.prior["step"])
    return ref, load_checkpoint(ref)


def _init_prior_codes(cfg, code_source):
    """Start every code's prior entries at the mean of the prior run's codes.

    The random init (std 0.01 per entry) puts a fresh code far outside the range the
    prior was trained on when its codes are small; the mean is the prior's centre.
    The residual's entries keep the random init. init.parts with codes overrides this.
    """
    if cfg.field.prior is None or cfg.code.kind != "auto_decoder":
        return
    _, state = _prior_checkpoint(cfg)
    prior_n_z = code_size(load_prior_config(cfg))
    if prior_n_z == 0 or "codes" not in state:
        return
    mean = state["codes"]["codes.weight"].mean(dim=0)
    table = code_source.codes.weight
    with torch.no_grad():
        table[:, :prior_n_z] = mean.to(table)
    print(f"[prior] code entries 0..{prior_n_z - 1} of {table.shape[0]} code(s) start at the mean "
          f"of the prior's {state['codes']['codes.weight'].shape[0]} codes (norm {mean.norm():.3g})")


def _check_prior_frame(cfg):
    """A prior field works in its run's frame: this run must normalise into the same one.

    Raises:
        NotImplementedError: a pair run (its frame comes from its own two shapes)
        ValueError: a cohort run without frame_source
    """
    if cfg.field.prior is None:
        return
    if cfg.train.mode != "cohort":
        raise NotImplementedError("field.prior needs train.mode='cohort': only a cohort run can "
                                  "take its frame from the prior's members (frame_source)")
    prior_cfg = load_prior_config(cfg)
    prior_frame = prior_cfg.frame_source or prior_cfg.source
    if cfg.frame_source is None:
        raise ValueError(f"field.prior: set frame_source to the prior's members ({prior_frame}), "
                         f"else this run's frame comes from its own shapes and the prior field "
                         f"sees them in the wrong place")
    if os.path.realpath(cfg.frame_source) != os.path.realpath(prior_frame):
        print(f"[prior] WARNING: frame_source {cfg.frame_source} is not the prior's frame "
              f"{prior_frame}; correct only if it holds the same bounding-box extremes")
