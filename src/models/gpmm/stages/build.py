"""The ssm_build stage: build a GPMM from its YAML configuration and save its NPZ archive.

    python scripts/run.py configs/gpmm/build/baby/gaussian.yaml

The archive is content-addressed (ModelCacheSpec): it records the resolved
kernel, approximation, truncation and inference it was built with.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from src.models.gpmm.config import GpmmModelConfig
from src.models.gpmm.model.builder import BuildContext, ModelSource, build_model, resolve_kernel_spec
from src.models.gpmm.model.model_memory_io import ModelCacheSpec, save_model

Config = GpmmModelConfig


def build_and_save(config: GpmmModelConfig, *, verbose: bool = True) -> Path:
    """Builds the configured model, saves its archive and returns the archive's path."""
    raw_config = asdict(config)

    model = build_model(raw_config, verbose=verbose)

    domain_spec = raw_config.get("domain", {})
    source_path = domain_spec.get("source") or raw_config.get("source")
    if source_path is None:
        raise ValueError("GPMM config must define domain.source")
    source = ModelSource.load(
        source_path,
        normalize=config.normalize,
        atlas_options=raw_config.get("atlas"),
        verbose=verbose,
    )
    resolved_kernel = resolve_kernel_spec(
        raw_config.get("kernel", {"name": "pca"}),
        BuildContext(model.domain, source, verbose=verbose),
    )

    cache_spec = ModelCacheSpec(
        kernel_spec=resolved_kernel,
        approximation=config.approximation,
        truncation=config.truncation,
        inference=config.inference,
    )
    output = config.output
    if hasattr(output, "path"):
        output_path = output.path
    else:
        output_path = Path(output["dir"]) / output["filename"]
    save_model(output_path, model, cache_spec)
    print(f"Saved model to {output_path}")
    return output_path


def run(config: GpmmModelConfig) -> Path:
    return build_and_save(config)
