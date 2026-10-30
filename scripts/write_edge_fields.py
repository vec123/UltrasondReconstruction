#!/usr/bin/env python
"""Bake edge fields onto point-cloud surfaces (e.g. fused ultrasound targets), in place.

A fused target is a partial face given as points: its rim and the borders of its
holes are where the fusion is least reliable, and where a template point that the
scan never saw finds its nearest scan point. ``surface_measures.compute_point_cloud_edges``
marks those edge points (angular gap of the tangent-plane neighbours) and writes:

  edge             1 on edge points, 0 elsewhere
  edge_distance    distance to the nearest edge point (the file's own units)
  edge_weight      smooth ramp, 0 on the edge -> 1 at --ramp-spacings point spacings inside
  <field>_edge     <field> x edge_weight, for every --combine field (default: support)

The data term uses them as (see ChamferSideCfg.inherit_target_field):

  target_to_source.field_name: support_edge           scan points near an edge count less
  source_to_target.inherit_target_field: true         a template point matched to an edge
                                                      point stops pulling

Purely additive: every existing point array is kept. Writes in place.

Usage:
    python scripts/write_edge_fields.py /home/victor/Escritorio/Data/fused_targets/targets/*.vtp
"""

import argparse
from pathlib import Path

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from src.vtk.fields import add_point_field
from src.vtk.io import load_vtp, save_vtp
from src.vtk.surface_measures import compute_point_cloud_edges


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vtp", nargs="+", help="point-cloud surfaces with a 'normal' array")
    parser.add_argument("--k", type=int, default=16, help="neighbours per point for the gap test")
    parser.add_argument("--gap-deg", type=float, default=120.0,
                        help="largest angular gap an interior point may show (default 120)")
    parser.add_argument("--ramp-spacings", type=float, default=3.0,
                        help="ramp width in point spacings (default 3, about 3 mm on the fused targets)")
    parser.add_argument("--combine", nargs="*", default=["support"],
                        help="fields to also write multiplied by edge_weight, as <field>_edge")
    args = parser.parse_args()

    shares = []
    for path in args.vtp:
        polydata = load_vtp(path)
        result = compute_point_cloud_edges(polydata, k=args.k, gap_deg=args.gap_deg,
                                           ramp_spacings=args.ramp_spacings)
        for name in args.combine:
            array = polydata.GetPointData().GetArray(name)
            if array is None:
                raise ValueError(f"{path} has no '{name}' point array to combine with edge_weight")
            add_point_field(polydata, (vtk_to_numpy(array) * result["weight"]).astype(np.float32),
                            field_name=f"{name}_edge")
        save_vtp(polydata, path)
        shares.append((result["edge"].mean(), (result["weight"] < 1.0).mean()))
        print(f"[edges] {Path(path).name}: {result['edge'].sum()} edge points "
              f"({result['edge'].mean():.1%}), spacing {result['spacing']:.3g}, ramp {result['ramp']:.3g} "
              f"(file units), {(result['weight'] < 1.0).mean():.1%} of points below full weight")
    shares = np.array(shares)
    print(f"[edges] {len(args.vtp)} file(s): edge points median {np.median(shares[:, 0]):.1%} "
          f"(range {shares[:, 0].min():.1%}-{shares[:, 0].max():.1%}), below full weight median "
          f"{np.median(shares[:, 1]):.1%}; wrote edge, edge_distance, edge_weight"
          + "".join(f", {name}_edge" for name in args.combine))


if __name__ == "__main__":
    main()
