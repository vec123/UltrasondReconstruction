"""CLI for applying a trained flow: ``python -m src.resnet_lddmm.apply.cli <config>``.

Mirrors src.resnet_lddmm.cli, including how it reports failure: a bad config is a
usage error and prints one line, not a traceback.
"""

import argparse
import sys
from dataclasses import replace

import yaml

from src.resnet_lddmm.resnet_flow.apply.config import load_apply_config
from src.resnet_lddmm.resnet_flow.apply.runner import run


def main(argv=None):
    """Parse CLI args and apply the checkpoint the config names.

    Args:
        argv: argument list, or None for sys.argv

    Returns:
        Process exit code
    """
    parser = argparse.ArgumentParser(
        description="Apply a trained ResNetLDDMM flow to a shape at full resolution"
    )
    parser.add_argument("config", type=str, help="Path to apply YAML config file")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Override the config's checkpoint (a .pt file, or a checkpoints/ directory)",
    )
    parser.add_argument(
        "--template", type=str, default=None,
        help="Override the shape to deform (default: the training config's source)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override where results are written",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_apply_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except (ValueError, yaml.YAMLError) as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 1

    # CLI beats file, for the three fields worth overriding ad hoc: which weights,
    # what to deform, where it goes. Anything else is a config edit -- it has to be
    # recorded to be reproducible.
    overrides = {"checkpoint": args.checkpoint, "template": args.template,
                 "output_dir": args.output_dir}
    cfg = replace(cfg, **{k: v for k, v in overrides.items() if v is not None})

    try:
        _, manifest = run(cfg)
    except Exception as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        return 1

    print(f"Applied {manifest['checkpoint']} to {manifest['template']} "
          f"({manifest['num_points']} points, no subsampling).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
