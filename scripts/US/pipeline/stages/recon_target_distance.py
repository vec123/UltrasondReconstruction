"""Stage: nearest-neighbor (L2) point-to-point distance between a
reconstruction and its registration target, in both directions.

- recon -> target: for every recon vertex, distance to the nearest target
  vertex. Written onto the recon mesh as the point field 'recon_dist_2_target'.
- target -> recon: for every target vertex, distance to the nearest recon
  vertex. Written onto the target mesh as the point field 'target_dist_2_recon'.

This is a point-to-point (vertex-to-nearest-vertex) approximation of the true
point-to-surface distance, computed with scipy.spatial.cKDTree - same
approach as scan_consistency.py's consistency scoring.

Both meshes live in whatever coordinate frame align_to_template produced (the
pose AND scale of the SSM's own template, not the original scan) - two
different SSM/template runs generally rotate/translate/scale the SAME raw
subject data differently, so their outputs don't line up even when they
started from the identical isosurface. If `transforms_dir` is set, the
reported distances are divided by that subject's median similarity-transform
scale (the median over every sweep's saved transform in `transforms_dir` -
see pipeline/transforms.py's median_scale(), the same approach
scan_consistency.py uses to convert sigma_mm the other direction), so
distance numbers are comparable across runs/templates. If `subject_id` is
ALSO set, the saved mesh geometry is mapped back through the FULL inverse
(rotation + translation + scale) of that one specific saved transform - not
just rescaled - landing it back in that subject's original, pre-alignment
frame. Two datasets built from the same raw scan then overlap when viewed
together, not just share a comparable scale. Left unset, distances and
geometry stay in the meshes' own template space/units.
"""

import os

from scipy.spatial import cKDTree
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from src.config.US.schema import ReconTargetDistanceConfig as Config
from src.vtk.fields import add_point_field
from src.vtk.io import load_vtp, save_vtp

from src.transforms.similarity import load_subject_transform, median_scale


def nearest_neighbor_distances(query_points, reference_points):
    """For every point in `query_points`, the L2 distance to its nearest
    neighbor in `reference_points`."""
    tree = cKDTree(reference_points)
    distances, _ = tree.query(query_points, k=1, workers=-1)
    return distances


def rescale_points(polydata, factor):
    """Uniformly scales a polydata's point coordinates in place."""
    points = polydata.GetPoints()
    coords = vtk_to_numpy(points.GetData()) * factor
    points.SetData(numpy_to_vtk(coords, deep=True))


def run(cfg: Config):
    output_target_path = os.path.join(cfg.output_dir or os.path.dirname(cfg.target_path), "target_w_distance.vtp")
    output_recon_path = os.path.join(cfg.output_dir or os.path.dirname(cfg.recon_path), "recon_w_distance.vtp")

    target_mesh = load_vtp(cfg.target_path)
    recon_mesh = load_vtp(cfg.recon_path)

    target_points = vtk_to_numpy(target_mesh.GetPoints().GetData())
    recon_points = vtk_to_numpy(recon_mesh.GetPoints().GetData())

    if cfg.transforms_dir:
        scale = median_scale(cfg.transforms_dir)
        units = "physical units, original data scale"
    else:
        scale = 1.0
        units = "template units"

    if cfg.subject_id:
        inverse = load_subject_transform(cfg.transforms_dir, cfg.subject_id).inverse()
    else:
        inverse = None

    def place_geometry(mesh):
        if inverse is not None:
            return inverse.apply_polydata(mesh)
        if cfg.transforms_dir:
            rescale_points(mesh, 1.0 / scale)
        return mesh

    recon_dist_2_target = nearest_neighbor_distances(recon_points, target_points) / scale
    add_point_field(recon_mesh, recon_dist_2_target, field_name="recon_dist_2_target")
    recon_mesh = place_geometry(recon_mesh)
    save_vtp(recon_mesh, output_recon_path)
    print(
        f"recon_dist_2_target ({units}): mean={recon_dist_2_target.mean():.4f} "
        f"max={recon_dist_2_target.max():.4f} -> {output_recon_path}"
    )

    target_dist_2_recon = nearest_neighbor_distances(target_points, recon_points) / scale
    add_point_field(target_mesh, target_dist_2_recon, field_name="target_dist_2_recon")
    target_mesh = place_geometry(target_mesh)
    save_vtp(target_mesh, output_target_path)
    print(
        f"target_dist_2_recon ({units}): mean={target_dist_2_recon.mean():.4f} "
        f"max={target_dist_2_recon.max():.4f} -> {output_target_path}"
    )
