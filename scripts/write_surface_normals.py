#!/usr/bin/env python
"""Bake a per-vertex ``normal`` point array onto an existing .vtp, from its faces.

The array ``data_name: field_weighted_point2plane_chamfer`` needs on its TARGET.
``io.load_shape`` reads a point field called ``normal`` into ``Shape.normals``,
and ``io.joint_normalize`` carries it through unchanged -- its single uniform
scale moves points but leaves normal DIRECTIONS untouched -- so baking the array
is all the wiring a training run needs.

NOT the primary path: ``scan_fusion`` writes this field itself
(``compute_normals: true``), on the final post-``min_support`` topology, which is
where a pipeline-produced target should get it. This script is the escape hatch,
for the cases the stage cannot cover:

  * artifacts baked before scan_fusion started emitting the field, where
    re-running fusion to add one array is not worth it,
  * targets produced outside the pipeline,
  * re-baking with a different ``--weighting`` without re-fusing.

Both paths call ``surface_measures.compute_vertex_normals``, so the convention
cannot drift between them. See that function for the averaging rules and for why
a mesh without polygons is refused rather than silently estimated.

Purely additive: every existing point array is preserved untouched. Writes in
place unless --output is given.

Usage:
    python scripts/write_surface_normals.py path/to/target.vtp
    python scripts/write_surface_normals.py in.vtp --output out.vtp --weighting area
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.vtk.io import load_vtp, save_vtp
from src.vtk.surface_measures import compute_vertex_normals


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vtp", help="mesh to bake normals onto (must have faces)")
    parser.add_argument("--output", default=None,
                        help="where to write (default: overwrite the input in place)")
    parser.add_argument("--field", default="normal",
                        help="point array name to write (default: 'normal', which "
                             "is what io.load_shape looks for)")
    parser.add_argument("--weighting", default="unit", choices=("unit", "area"),
                        help="how incident faces are averaged (default: unit, "
                             "matching scan_fusion's own default)")
    args = parser.parse_args()

    polydata = load_vtp(args.vtp)

    if polydata.GetPointData().GetArray(args.field) is not None:
        print(f"[normals] overwriting existing '{args.field}' array")

    normals = compute_vertex_normals(polydata, field_name=args.field,
                                     weighting=args.weighting)

    out_path = args.output or args.vtp
    save_vtp(polydata, out_path)

    lengths = np.linalg.norm(normals, axis=1)
    degenerate = int((lengths < 1e-6).sum())
    print(f"[normals] {args.vtp}")
    print(f"[normals]   {normals.shape[0]} vertices, {polydata.GetNumberOfPolys()} polys, "
          f"{args.weighting}-weighted")
    if degenerate:
        # Not orphan vertices (those have no incident face at all) but ones whose
        # incident triangles are degenerate or cancel exactly, e.g. a fold where
        # two opposite-facing faces meet. Reported, not repaired: there is no
        # tangent plane to invent there. The point-to-plane term scores them at
        # zero plane-distance, so they fall back to its point-to-point component.
        # A LARGE count means the mesh's topology is broken and point-to-plane is
        # measuring less than it appears to.
        print(f"[normals]   WARNING: {degenerate} normals are zero-length "
              f"({degenerate / len(lengths) * 100:.2f}%) -- no tangent plane at "
              f"those vertices; they contribute no plane term")
    print(f"[normals]   |n| in [{lengths.min():.6f}, {lengths.max():.6f}] (want ~1.0)")
    print(f"[normals]   wrote '{args.field}' -> {out_path}")


if __name__ == "__main__":
    main()
