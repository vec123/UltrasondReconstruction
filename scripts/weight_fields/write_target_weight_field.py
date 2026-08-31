#!/usr/bin/env python
"""Bake a ``target_weight`` point field onto a target scan, from a YAML config.

Reads an EXPLICIT ``include`` list of landmark labels (resolved from the
``landmarks:`` CSVs' per-label mean position, in the scan's own world frame),
computes each vertex's GEODESIC distance (along the scan's own mesh, not
straight-line) to the nearest included landmark, and multiplies that
proximity by an existing reliability field (``consistency_score`` by
default): "consistent and close is high, consistent and far is lower,
inconsistent stays low regardless of proximity" -- floored at ``min_weight``
so nothing is ever weighted to exactly zero.

Purely additive: every existing point field on the file (consistency_score,
roughness, ...) is preserved untouched; this only adds one more array
alongside them.

Geodesic distance on a large scan means building a sparse mesh graph and
running Dijkstra from each included landmark -- this can take a while
(minutes, not seconds) on a scan with a million-plus vertices.

Usage:
    python scripts/write_target_weight_field.py configs/target_weight_field.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.weight_field.config import load_yaml_mapping
from src.resnet_lddmm.resnet_flow.io import load_shape, export_shape, FrameTransform
from src.weight_field.target_weight_field import (
    TargetWeightFieldCfg,
    compute_target_weight_field,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="path to a target-weight-field YAML config "
                                        "(see configs/target_weight_field.yaml)")
    args = parser.parse_args()

    cfg = TargetWeightFieldCfg.from_dict(load_yaml_mapping(args.config))

    shape = load_shape(cfg.vtp)
    weight = compute_target_weight_field(shape, cfg)

    # Additive: start from every field the file already carries, add one more.
    point_fields = dict(shape.point_fields or {})
    point_fields[cfg.output_field] = weight.numpy()

    output_path = cfg.output or cfg.vtp
    # shape.points is already in WORLD coordinates (load_shape never normalises),
    # so an identity transform writes them back unchanged -- export_shape's
    # invert() is what every other exporter in this codebase uses to leave the
    # normalised frame; here there is no normalised frame to leave.
    export_shape(shape.points[0], shape.faces, FrameTransform.identity(), output_path,
                label="TargetWithWeightField", point_fields=point_fields)

    def _describe(entry):
        base = f"{entry.label}(scale={entry.decay_scale}, max={entry.max_weight}"
        if entry.decay_axes:
            axes = ", ".join(f"{axis}=[{neg},{pos}]" for axis, (neg, pos) in entry.decay_axes.items())
            base += f", axes={{{axes}}}"
        return base + ")"

    details = ", ".join(_describe(entry) for entry in cfg.entries)
    print(f"[write_target_weight_field] {cfg.output_field!r} from {len(cfg.entries)} "
          f"landmark(s) ({details}) x {cfg.consistency_field!r}, "
          f"min_weight={cfg.min_weight}: {weight.shape[0]} vertices, "
          f"range [{weight.min().item():.4g}, {weight.max().item():.4g}] -> {output_path}")


if __name__ == "__main__":
    main()
