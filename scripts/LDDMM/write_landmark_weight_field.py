#!/usr/bin/env python
"""Bake a ``landmark_weight`` point field onto a template mesh, from a YAML config.

Reads an EXPLICIT ``include`` list of landmark labels (whatever is not listed
is simply excluded), resolves each one's position on the template the same
way ``LandmarkGuidanceLoss`` does (heatmap argmax -- see
``landmarks.resolve_template_points``), and for each vertex takes the
strongest of every included landmark's own
``max_weight * exp(-0.5 * (distance_to_it / its_decay_scale) ** 2)`` --
floored at ``min_weight`` so nothing is ever weighted exactly to zero. Each
``include`` entry may override ``decay_scale`` (halo size) and/or
``max_weight`` (importance at the landmark itself) for just that one
landmark; see ``configs/landmark_weight_field.yaml`` for the syntax.

Purely additive: every existing point field on the file (heatmap_<label>,
landmark_distance, curvature, ...) is preserved untouched; this only adds one
more array alongside them.

Usage:
    python scripts/write_landmark_weight_field.py configs/landmark_weight_field.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.resnet_lddmm.config import load_yaml_mapping
from src.resnet_lddmm.io import load_shape, export_shape, FrameTransform
from src.resnet_lddmm.landmark_weight_field import (
    LandmarkWeightFieldCfg,
    compute_landmark_weight_field,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="path to a landmark-weight-field YAML config "
                                        "(see configs/landmark_weight_field.yaml)")
    args = parser.parse_args()

    cfg = LandmarkWeightFieldCfg.from_dict(load_yaml_mapping(args.config))

    shape = load_shape(cfg.vtp)
    weight = compute_landmark_weight_field(shape, cfg)

    # Additive: start from every field the file already carries, add one more.
    point_fields = dict(shape.point_fields or {})
    point_fields[cfg.output_field] = weight.numpy()

    output_path = cfg.output or cfg.vtp
    # shape.points is already in WORLD coordinates (load_shape never normalises),
    # so an identity transform writes them back unchanged -- export_shape's
    # invert() is what every other exporter in this codebase uses to leave the
    # normalised frame; here there is no normalised frame to leave.
    export_shape(shape.points[0], shape.faces, FrameTransform.identity(), output_path,
                label="TemplateWithWeightField", point_fields=point_fields)

    def _describe(entry):
        base = f"{entry.label}(scale={entry.decay_scale}, max={entry.max_weight}"
        if entry.decay_axes:
            axes = ", ".join(f"{axis}=[{neg},{pos}]" for axis, (neg, pos) in entry.decay_axes.items())
            base += f", axes={{{axes}}}"
        return base + ")"

    details = ", ".join(_describe(entry) for entry in cfg.entries)
    print(f"[write_landmark_weight_field] {cfg.output_field!r} from {len(cfg.entries)} "
          f"landmark(s) ({details}), min_weight={cfg.min_weight}: {weight.shape[0]} vertices, "
          f"range [{weight.min().item():.4g}, {weight.max().item():.4g}] -> {output_path}")


if __name__ == "__main__":
    main()
