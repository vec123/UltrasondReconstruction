#!/usr/bin/env python
"""Split a trained ResNet-LDDMM run's deformation into the network and each designed mode.

CLI for ``src/resnet_lddmm/resnet_flow/callbacks/deformation/modes`` (see export.py
there for the files written to ``<run>/mode_decomposition/``). Nothing is trained.

Usage:
    python scripts/LDDMM/decompose_modes.py outputs/resnet_lddmm_baby_mean/9992
    python scripts/LDDMM/decompose_modes.py <run> --checkpoint <run>/checkpoints/step_500.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.resnet_lddmm.resnet_flow.callbacks.deformation.modes import export_decomposition  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="the run's output_dir (holds config.json and checkpoints/)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="a .pt file (default: the run's final checkpoint)")
    parser.add_argument("--out", type=Path, default=None, help="default: <run>/mode_decomposition")
    parser.add_argument("--ends-only", action="store_true",
                        help="write the end shapes only, not every step of every flow (~150 MB)")
    args = parser.parse_args(argv)
    try:
        export_decomposition(args.run, args.checkpoint, args.out, trajectories=not args.ends_only)
    except ValueError as exc:
        raise SystemExit(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
