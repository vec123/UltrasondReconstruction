"""Build and save a GPMM from a typed YAML configuration.

Usage:
    python -m scripts.SSM.build_model src/config/SSM/model_pca.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from src.SSM.model.builder import BuildContext, ModelSource, build_model, resolve_kernel_spec
from src.SSM.model.model_memory_io import ModelCacheSpec, save_model
from src.config.SSM.client import ConfigClient
from src.config.SSM.schema import GpmmModelConfig


def build_and_save(config_path, *, verbose: bool = True):
    """Build the configured model and save its content-addressed NPZ archive."""
    config_path = Path(config_path)
    project_root = Path(__file__).resolve().parents[2]
    client = ConfigClient(project_root=project_root)
    config = client.load(config_path, GpmmModelConfig)
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="GPMM YAML configuration path")
    args = parser.parse_args(argv)
    build_and_save(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
