"""Stage: score and denoise aligned meshes by cross-subject consistency.

For each aligned mesh, every vertex gets a Gaussian-decay "consistency score"
based on its distance to the nearest vertex of every OTHER aligned mesh.
Vertices that agree across the ensemble score near 1.0; outliers score near
0. Optionally thresholds each mesh down to only its high-confidence vertices
and merges those into one combined scan.

Nearest-neighbor distances are computed with scipy.spatial.cKDTree, fully
vectorized over all of a mesh's points at once (no per-point Python loop) -
this is an approximation of the nearest point on the OTHER mesh's *surface*
(what a vtkCellLocator would give): for a reasonably dense, evenly sampled
mesh the difference is bounded by the local edge length, negligible next to
sigma_mm, and worth it for the large (~2-3 orders of magnitude) speedup over
querying one point at a time through VTK's Python bindings.

This stage does not move any geometry (it only scores/filters), so it does
not read or write a rigid transform - it only reads the scale factor already
saved by the align_to_template stage, to convert `sigma_mm` (a physical
distance) into the aligned mesh's own units.
"""

import glob
import os

import numpy as np
import vtk
from scipy.spatial import cKDTree
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from src.config.US.schema import ScanConsistencyConfig as Config
from src.vtk.fields import add_point_field
from src.vtk.io import load_vtp, save_vtp
from src.vtk.process import filter_largest_component

from src.transforms.similarity import median_scale


def build_point_trees(meshes):
    """Extracts each mesh's vertex positions and builds a KDTree over them,
    for fast vectorized nearest-neighbor distance queries."""
    points = [vtk_to_numpy(mesh.GetPoints().GetData()) for mesh in meshes]
    trees = [cKDTree(pts) for pts in points]
    return points, trees


def compute_point_consistency(points, target_idx, trees, sigma):
    """
    Computes a point-wise consistency score [0.0, 1.0] for mesh `target_idx`
    against every other mesh's KDTree using Gaussian decay. Vectorized: each
    other mesh costs one batched `tree.query` over all of the target mesh's
    points, instead of one query per point.
    """
    query_points = points[target_idx]
    num_other_meshes = len(trees) - 1
    if num_other_meshes <= 0:
        return np.ones(len(query_points), dtype=np.float32)

    two_sigma_sq = 2.0 * (sigma**2)
    total_weight = np.zeros(len(query_points), dtype=np.float64)

    for j, tree in enumerate(trees):
        if j == target_idx:
            continue
        dist, _ = tree.query(query_points, k=1, workers=-1)
        total_weight += np.exp(-(dist**2) / two_sigma_sq)

    return (total_weight / num_other_meshes).astype(np.float32)


def threshold_mesh_by_consistency(polydata, min_score):
    """Extracts cells and points whose consistency_score >= min_score."""
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "consistency_score")
    threshold.SetLowerThreshold(min_score)
    threshold.Update()

    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(threshold.GetOutput())
    geometry_filter.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(geometry_filter.GetOutput())
    cleaner.Update()

    return cleaner.GetOutput()


def backfill_missing_point_fields(meshes):
    """Adds a NaN-filled point array for any field absent from a mesh but
    present on another, so every mesh in the list carries the same field set.

    vtkAppendPolyData only keeps point fields common to ALL of its inputs -
    silently dropping any field that happens to be missing from even one
    subject - so this has to run before appending to actually merge every
    field into the combined scan, not just whichever ones every subject
    happened to share.
    """
    field_components = {}
    for mesh in meshes:
        point_data = mesh.GetPointData()
        for i in range(point_data.GetNumberOfArrays()):
            arr = point_data.GetArray(i)
            field_components.setdefault(arr.GetName(), arr.GetNumberOfComponents())

    for mesh in meshes:
        point_data = mesh.GetPointData()
        n_pts = mesh.GetNumberOfPoints()
        for name, n_components in field_components.items():
            if point_data.GetArray(name) is not None:
                continue
            fill = np.full((n_pts, n_components), np.nan, dtype=np.float32)
            vtk_arr = numpy_to_vtk(fill, deep=True)
            vtk_arr.SetName(name)
            point_data.AddArray(vtk_arr)


def denoise_and_merge(vtp_files, meshes, cfg: Config, threshold, denoised_dir, merged_suffix):
    """Runs one threshold/denoise/largest-component/merge pass over already-scored
    `meshes`, saving per-mesh output into `denoised_dir` and the combined union as
    `merged_high_confidence_scan{merged_suffix}.vtp` under cfg.output_merged_dir."""
    os.makedirs(denoised_dir, exist_ok=True)

    denoised_meshes = []
    for filepath, polydata in zip(vtp_files, meshes):
        base_name = os.path.basename(filepath)
        denoised_mesh = threshold_mesh_by_consistency(polydata, min_score=threshold)
        denoised_mesh = filter_largest_component(
            denoised_mesh,
            min_bridge_width=cfg.denoise_min_bridge_width,
            relative=cfg.denoise_min_bridge_relative,
        )
        out_denoised_path = os.path.join(denoised_dir, base_name)
        save_vtp(denoised_mesh, out_denoised_path)
        denoised_meshes.append(denoised_mesh)

    if not denoised_meshes:
        return

    backfill_missing_point_fields(denoised_meshes)

    denoised_append_filter = vtk.vtkAppendPolyData()
    for denoised_mesh in denoised_meshes:
        denoised_append_filter.AddInputData(denoised_mesh)
    denoised_append_filter.Update()

    merged_vtp_path = os.path.join(cfg.output_merged_dir, f"merged_high_confidence_scan{merged_suffix}.vtp")
    save_vtp(denoised_append_filter.GetOutput(), merged_vtp_path)
    print(f"Saved combined high-confidence scan (min_consistency_threshold={threshold:g}) to: {merged_vtp_path}")


def run(cfg: Config):
    os.makedirs(cfg.output_consistency_dir, exist_ok=True)

    thresholds = cfg.min_consistency_threshold
    sweeping = isinstance(thresholds, list)
    if not sweeping:
        thresholds = [thresholds]

    if cfg.apply_denoise_threshold:
        os.makedirs(cfg.output_denoised_dir, exist_ok=True)
        os.makedirs(cfg.output_merged_dir, exist_ok=True)

    vtp_files = sorted(glob.glob(os.path.join(cfg.input_aligned_dir, "*.vtp")))
    if not vtp_files:
        raise FileNotFoundError(f"No .vtp files found in '{cfg.input_aligned_dir}'.")

    print(f"Loading {len(vtp_files)} aligned meshes...")
    meshes = [load_vtp(f) for f in vtp_files]

    print("Building spatial search trees for fast nearest-neighbor queries...")
    points, trees = build_point_trees(meshes)

    # Distances are measured in template space, so sigma has to follow the alignment scale.
    sigma = cfg.sigma_mm
    if cfg.transforms_dir:
        sigma = cfg.sigma_mm * median_scale(cfg.transforms_dir)

    print(f"Computing consistency scores (Sigma = {cfg.sigma_mm} mm -> {sigma:.4f} template units)...")

    for idx, (filepath, polydata) in enumerate(zip(vtp_files, meshes)):
        base_name = os.path.basename(filepath)
        print(f"Scoring ({idx + 1}/{len(meshes)}): {base_name}...")

        scores = compute_point_consistency(points, idx, trees, sigma=sigma)
        add_point_field(polydata, scores, field_name="consistency_score")

        out_consistency_path = os.path.join(cfg.output_consistency_dir, base_name)
        save_vtp(polydata, out_consistency_path)

    if cfg.apply_denoise_threshold:
        for threshold in thresholds:
            denoised_dir = (
                os.path.join(cfg.output_denoised_dir, f"threshold_{threshold:g}")
                if sweeping
                else cfg.output_denoised_dir
            )
            merged_suffix = f"_threshold_{threshold:g}" if sweeping else ""
            print(f"\nDenoising at min_consistency_threshold = {threshold:g}...")
            denoise_and_merge(vtp_files, meshes, cfg, threshold, denoised_dir, merged_suffix)

    print("\nConsistency computation finished successfully!")
