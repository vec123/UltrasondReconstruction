"""Small helpers shared by every part of the runner: seeding, dtype, normalization domain, code size, config dump."""

import os
import json
import random
import dataclasses
import numpy as np
import torch
from src.models.lddmm.resnet_flow.config import CENTERING_DOMAINS


def seed_everything(seed):
    """Mirrors seeding: returns loader rng or None if seed is None."""
    if seed is None:
        return None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    key = torch.Generator(device="cpu")
    key.manual_seed(seed)
    return key


def _normalization_domain(cfg):
    """cfg.centering -> the (min, max) domain joint_normalize places shapes in.

    getattr so a stub config without the field still builds, matching how the rest
    of this module reads optional attributes.
    """
    return CENTERING_DOMAINS[getattr(cfg, "centering", "box")]


def set_default_dtype(cfg):
    """cfg.dtype -> the process-wide default dtype.

    Call BEFORE anything is constructed: nn.Module parameters take the default
    dtype at __init__, so setting this later would leave a float32 model reading
    float64 data. Process-wide, by PyTorch's design -- there is no per-model
    equivalent.
    """
    torch.set_default_dtype(getattr(torch, getattr(cfg, "dtype", "float32")))


def code_size(cfg) -> int:
    """How many code entries cfg's field reads: 0 without a code."""
    return 0 if cfg.code.kind == "none" else int(cfg.code.n_z)


def _dump_config(cfg, mode):
    """Dump the FULL config to output dir for provenance.

    Every config class from ExperimentCfg down (FieldCfg, CodeCfg, EncoderConfig,
    GraphSpec, ...) is a plain dataclass, so dataclasses.asdict recurses through
    the whole tree -- this must capture every field, not a hand-picked subset,
    since a field left out here is a field a resumed/compared run can't recover.
    """
    config_path = os.path.join(cfg.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({"mode": mode, **dataclasses.asdict(cfg)}, f, indent=2)
