#!/usr/bin/env python
"""Carry a scan from an aligned-template frame (mm) into a shape model's own frame.

The model's mean (``--model-template``, e.g. the template.vtp of a sample stage folder) and the
aligned template (``--aligned-template``, e.g. outputs/Casos_Mar/cohort/template_baby_mean_aligned.vtp)
are the same mesh, vertex for vertex, in two frames. The similarity between them
(B = s R A + t, Umeyama / Procrustes with scale) is exact, and its inverse carries any
scan from the aligned frame into the model frame:

    x_model = R^T (x - t) / s,   normal_model = R^T normal

Writes:
  <output-dir>/targets/<scan name>.vtp   the scan(s) in the model frame, every point array
                                         kept (``normal`` rotated)
  <output-dir>/template.vtp              the model template with the aligned template's
                                         point arrays (landmark_weight, ...), for the
                                         field-weighted data terms
  <output-dir>/similarity.json           s, R, t

Usage:
    python scripts/tools/target_to_ssm_frame.py \\
        --model-template /home/victor/Escritorio/Data/BabySSMData/template.vtp \\
        --aligned-template outputs/Casos_Mar/cohort/template_baby_mean_aligned.vtp \\
        --scan outputs/Casos_Mar/cohort/fused_targets_4096_fields/16_26_semanas.vtp \\
        --output-dir outputs/baby_ssm_cohort/fit_16_26
"""

import argparse
import os
from pathlib import Path

import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from src.shared.transforms.similarity import SimilarityTransform, umeyama_similarity_transform
from src.shared.vtk.io import load_vtp, save_vtp

VECTOR_ARRAYS = ("normal",)   # point arrays that rotate with the frame


def points_of(poly):
    return vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)


def set_points(poly, points):
    poly.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float32), deep=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-template", required=True)
    parser.add_argument("--aligned-template", required=True)
    parser.add_argument("--scan", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    model_t, aligned_t = load_vtp(args.model_template), load_vtp(args.aligned_template)
    a, b = points_of(model_t), points_of(aligned_t)
    if a.shape != b.shape:
        raise ValueError(f"templates differ in vertex count: {a.shape} vs {b.shape}")
    model_to_aligned = SimilarityTransform(umeyama_similarity_transform(a, b))
    scale, rot, trans = model_to_aligned.scale, model_to_aligned.rotation, model_to_aligned.translation
    residual = np.linalg.norm(scale * a @ rot.T + trans - b, axis=1)
    print(f"similarity: scale {scale:.2f}, residual mean {residual.mean():.2e} max {residual.max():.2e} (aligned units)")

    out = Path(args.output_dir)
    for path in args.scan:
        scan = load_vtp(path)
        set_points(scan, (points_of(scan) - trans) @ rot / scale)
        pd = scan.GetPointData()
        for name in VECTOR_ARRAYS:
            if pd.HasArray(name):
                n = vtk_to_numpy(pd.GetArray(name)).astype(np.float64) @ rot
                array = numpy_to_vtk(np.ascontiguousarray(n, dtype=np.float32), deep=True)
                array.SetName(name)
                pd.RemoveArray(name)
                pd.AddArray(array)
        save_vtp(scan, str(out / "targets" / Path(path).name))

    # The model template's points with the aligned template's point arrays (same vertex order).
    template = load_vtp(args.aligned_template)
    set_points(template, a)
    save_vtp(template, str(out / "template.vtp"))
    model_to_aligned.to_json(out / "similarity.json", model_template=args.model_template,
                             aligned_template=args.aligned_template, scans=args.scan)
    print(f"wrote {len(args.scan)} scan(s) + template to {out}")


if __name__ == "__main__":
    main()
