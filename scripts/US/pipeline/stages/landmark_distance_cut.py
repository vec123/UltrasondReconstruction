"""Stage: keep only mesh points close enough to a landmark.

For each subject's aligned mesh and landmark set (paired by subject id),
computes every vertex's Euclidean distance to its nearest landmark and drops
vertices farther than `max_distance_mm`. Shapes coming out of
align_to_template live in scaled template space, so the mm cutoff - defined
in the original, pre-alignment mesh/volume (nrrd/ply) space - is converted
into that subject's own template-space units using the per-subject scale
factor align_to_template saved, before thresholding.

Cutting by distance alone can leave several disconnected islands around
different landmarks; with `keep_largest_component` only the single largest
connected piece of the cut mesh is kept.

Doesn't move geometry (only drops points), so it saves no transform.
"""

import glob
import os

import numpy as np
from scipy.spatial import cKDTree
from vtk.util.numpy_support import vtk_to_numpy

from src.config.US import LandmarkDistanceCutConfig as Config
from src.vtk.extract import threshold_vtp_by_point_field
from src.vtk.fields import add_point_field
from src.vtk.io import load_vtp, save_vtp
from src.vtk.process import filter_largest_component

from ..matching import match_by_subject_id
from ..transforms import load_subject_transform


def nearest_landmark_distance(mesh_points, lmk_points):
    """Euclidean distance from every mesh point to its nearest landmark."""
    tree = cKDTree(lmk_points)
    dists, _ = tree.query(mesh_points, k=1, workers=-1)
    return dists.astype(np.float32)


def run(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)

    shape_files = sorted(glob.glob(os.path.join(cfg.shapes_dir, "*.vtp")))
    if not shape_files:
        raise FileNotFoundError(f"No .vtp shapes found in '{cfg.shapes_dir}'.")

    lmk_files = sorted(glob.glob(os.path.join(cfg.landmarks_dir, "*.vtp")))
    if not lmk_files:
        raise FileNotFoundError(f"No .vtp landmarks found in '{cfg.landmarks_dir}'.")

    common_ids, (shape_map, lmk_map) = match_by_subject_id(
        [shape_files, lmk_files], strip_suffixes=("_landmarks",)
    )
    if not common_ids:
        raise ValueError("No matching subject IDs found between shape and landmark files.")

    print(f"Found {len(common_ids)} subjects to cut by landmark distance (max {cfg.max_distance_mm} mm).")

    for subject_id in common_ids:
        shape_path = shape_map[subject_id]
        lmk_path = lmk_map[subject_id]

        mesh = load_vtp(shape_path)
        lmk_pts = vtk_to_numpy(load_vtp(lmk_path).GetPoints().GetData())
        mesh_pts = vtk_to_numpy(mesh.GetPoints().GetData())

        max_distance = cfg.max_distance_mm
        if cfg.transforms_dir:
            scale = load_subject_transform(cfg.transforms_dir, subject_id).scale
            max_distance = cfg.max_distance_mm * scale

        dists = nearest_landmark_distance(mesh_pts, lmk_pts)
        add_point_field(mesh, dists, field_name="landmark_distance")

        print(
            f"{subject_id}: {mesh.GetNumberOfPoints()} points, cutting at {cfg.max_distance_mm} mm "
            f"-> {max_distance:.4f} shape units"
        )
        cut_mesh = threshold_vtp_by_point_field(
            mesh, "landmark_distance", max_distance, relative=False, keep="below"
        )

        if cfg.keep_largest_component:
            print("  keep largest component of the cut mesh")
            if cfg.min_bridge_width > 0:
                cut_mesh = filter_largest_component(cut_mesh, cfg.min_bridge_width, relative=cfg.min_bridge_relative)
            else:
                cut_mesh = filter_largest_component(cut_mesh)

        out_path = os.path.join(cfg.output_dir, os.path.basename(shape_path))
        save_vtp(cut_mesh, out_path)
        print(f"  Kept {cut_mesh.GetNumberOfPoints():,} points -> {out_path}")

    print("Landmark distance cut completed successfully!")
