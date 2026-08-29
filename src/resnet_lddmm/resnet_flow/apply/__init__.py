"""Applying a trained flow to a shape.

Training and inference are separate entry points on purpose. A training run
subsamples the template (``loss.subsample_M``) to keep each gradient step cheap,
so its exports are a different random 1000-odd points every time -- never the
whole shape. Applying reloads the finished weights and runs the template through
in full, once.

Three layers, same shape as the training side:

  config.py  ApplyCfg -- what to load, what to deform, where to write. References
             the training config for everything about the MODEL.
  client.py  DeformationClient -- the trained flow as a callable object. No files,
             no config; usable from a notebook.
  runner.py  the composition root: resolves paths, code and frame, then exports.

Usage::

    python -m src.resnet_lddmm.apply.cli configs/apply/processed_ultrasound.yaml
"""

from src.resnet_lddmm.checkpoint import (
    CheckpointRef,
    load_checkpoint,
    resolve_checkpoint,
)
from src.resnet_lddmm.apply.client import Deformation, DeformationClient
from src.resnet_lddmm.apply.config import ApplyCfg, ExportCfg, load_apply_config
from src.resnet_lddmm.apply.runner import run, run_from_path

__all__ = [
    "ApplyCfg",
    "CheckpointRef",
    "Deformation",
    "DeformationClient",
    "ExportCfg",
    "load_apply_config",
    "load_checkpoint",
    "resolve_checkpoint",
    "run",
    "run_from_path",
]
