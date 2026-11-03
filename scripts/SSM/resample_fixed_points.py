#!/usr/bin/env python
"""Resample point clouds to a fixed point count, keeping every point array.

CohortBatchLoader (src/resnet_lddmm/learning/loader/loaders.py) stacks a batch's cohort
members with torch.stack, so they need equal point counts; this is the step that gives
fused targets (variable point counts) that equal count before target_to_ssm_frame.py.

TODO: this fixed-point resampling is a stand-in. It is meant to be replaced by padding
+ a validity mask in CohortBatchLoader (and the chamfer data term, whose mask is
currently hard-coded all-ones) so cohort members can keep their own point count.
Remove this script's use once that lands.

A cloud with more than --points points is subsampled without replacement; one with
fewer keeps every point and fills the rest by sampling with replacement (repeats).

Also writes <output-dir>/original_scans.json ({name: source .vtp path}), so inspection
scripts can still show a member's ORIGINAL, face-complete surface - see
scripts/inspection/code_pca_views.py's target_path, which reads it (from here or from
its parent, since a frame conversion and a train/test split often sit between this
script's output and the training source a run's config.json records).

Usage:
    python scripts/SSM/resample_fixed_points.py \\
        --scan outputs/Casos_Mar/cohort/fused_targets/*.vtp \\
        --points 4096 \\
        --output-dir outputs/Casos_Mar/cohort/fused_targets_4096_fields
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from src.shared.vtk.create import create_polydata
from src.shared.vtk.fields import add_point_fields
from src.shared.vtk.io import load_vtp, save_vtp


def resample(scan, n_points, rng):
    """(points [n_points, 3], {field: array}) resampled from scan's own points and point data."""
    points = vtk_to_numpy(scan.GetPoints().GetData()).astype(np.float64)
    if len(points) >= n_points:
        idx = rng.choice(len(points), n_points, replace=False)
    else:
        extra = rng.choice(len(points), n_points - len(points), replace=True)
        idx = np.concatenate([np.arange(len(points)), extra])
    point_data = scan.GetPointData()
    fields = {point_data.GetArrayName(i): vtk_to_numpy(point_data.GetArray(i))[idx]
             for i in range(point_data.GetNumberOfArrays())}
    return points[idx], fields


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scan", required=True, nargs="+", help="fused target .vtp file(s)")
    parser.add_argument("--points", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    out = Path(args.output_dir)
    manifest_path = out / "original_scans.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    for path in args.scan:
        points, fields = resample(load_vtp(path), args.points, rng)
        resampled = add_point_fields(create_polydata(points), fields)
        name = Path(path).stem
        save_vtp(resampled, str(out / f"{name}.vtp"))
        manifest[name] = os.path.abspath(path)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(args.scan)} scan(s) at {args.points} points to {out}")


if __name__ == "__main__":
    main()
