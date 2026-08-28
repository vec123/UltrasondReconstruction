"""CLI for ResNetLDDMM experiments (STEP T17).

Entry point for YAML config-based training runs.
Loads config, validates it, and orchestrates registration.
"""

import argparse
import sys

import yaml

from src.resnet_lddmm.config import ExperimentCfg, load_yaml_mapping
from src.resnet_lddmm.runner import run


def load_config(config_path: str) -> ExperimentCfg:
    """Load and parse YAML config file.

    Args:
        config_path: path to .yaml config file

    Returns:
        ExperimentCfg instance

    Raises:
        FileNotFoundError: if config file does not exist
        ValueError: if config is invalid (missing required keys, unknown keys)
        yaml.YAMLError: if YAML parsing fails
    """
    return ExperimentCfg.from_dict(load_yaml_mapping(config_path))


def main():
    """Parse CLI args and run experiment."""
    parser = argparse.ArgumentParser(
        description="ResNetLDDMM: neural ODE registration with learned velocity fields"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML config file",
    )

    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, yaml.YAMLError) as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        ctx = run(cfg)
        print(f"Training complete. Output saved to: {cfg.output_dir}")
    except Exception as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
