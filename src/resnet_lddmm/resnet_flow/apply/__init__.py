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

Kept empty on purpose, like every other package `__init__.py` in this import
chain (src/, resnet_lddmm/, resnet_flow/): config.py is torch-free by design
(a pipeline registry can bind it without paying for a training/inference
stack it may never reach - see scripts/pipeline/registry.py), and a
convenience re-export here would force every import of *any* name in this
package through client.py/runner.py's torch import regardless. Import the
submodule you need directly, e.g. `from
src.resnet_lddmm.resnet_flow.apply.config import ApplyCfg`.
"""
