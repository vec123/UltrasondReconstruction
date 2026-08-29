"""Stage: fuse many aligned, per-subject scans into one consensus mesh.

scan_consistency's merged output is a literal union (vtkAppendPolyData) of
every subject's own points - since alignment is never pixel-perfect,
overlapping regions end up with several slightly-offset copies of the same
patch layered on top of each other, which reads as noise even though every
contributing point individually passed the consistency threshold.

This stage fuses instead of unions: for every vertex of a fixed reference
mesh (typically the same template subjects were aligned to), it takes a
Gaussian-weighted average of the nearest point from each subject - pooled
across all subjects, not per-subject-then-averaged - producing ONE clean
consensus position per reference vertex instead of N overlapping copies.
Reference vertices with too little cross-subject support (few/no subjects
nearby) are dropped rather than guessed at.

Plain distance-only weighting has a failure mode at real anatomical
discontinuities (e.g. a cleft): alignment is only ever a global rigid+scale
fit, so a sharp edge from different subjects/sweeps rarely lands at the
EXACT same spot - it's off by a mm or two, well within sigma_mm. Averaging
those slightly-offset copies together smears the discontinuity into a
smoothed-over surface, which reads as the feature "disappearing" even though
every contributing point was individually valid. To guard against this,
weighting is also normal-aware (bilateral-style, ``normal_power``): a
contributing point is down-weighted by how much its surface normal disagrees
with the reference vertex's, since the two sides of a real gap face
different directions - so points from the wrong side of a gap get excluded
even when they land spatially close enough to matter. Set ``normal_power: 0``
to fall back to plain distance-only weighting.

Output is a single fused .vtp, carrying the reference mesh's topology plus
four new fields: "support" (total kernel weight backing each vertex's fused
position), "n_contributors" (how many distinct subjects had a point within
range), "fit_residual" (the weighted RMS spread of the contributing points
around the fused position - how naturally they agree in SPACE, not just how
many/how much of them there are), and "roughness" (how naturally the fused
vertex sits among its neighbors in the OUTPUT mesh's own surface topology,
via the same n-ring local-surface-deviation measure ``roughness_filter``
uses - see ``compute_vertex_roughness``). These answer different questions:
a vertex can have every contributing subject agreeing tightly (low
fit_residual) yet still poke out as a bump/dent relative to its mesh
neighbors (high roughness), e.g. at the ragged edge where a low-support
region got dropped next to it.

The reference mesh's own point fields (e.g. the template's heatmap_*/
landmark_distance/mode_*_std fields) are carried straight through too, in
lockstep with the fused positions - the fused mesh starts as a 1:1 copy of
the reference's point order, so no resampling is needed. That same 1:1
order is what "reference_vertex_id" records: an index into the reference
mesh's OWN point array. It matters because min_support then drops
low-agreement vertices and ``extract_polydata_vertex_subset`` reindexes
what's left, so position i in the saved fused mesh no longer means
reference-point i - only "reference_vertex_id" still says which reference
vertex a surviving fused vertex came from. The same field is also written
onto a copy of the reference mesh saved alongside the fused output
("reference.vtp", identity-valued there), so both files carry the same
join key for downstream point-to-point lookups.
"""

import glob
import os

import numpy as np
from scipy.spatial import cKDTree
from vtk.util import numpy_support

from src.config.US.schema import ScanFusionConfig as Config
from src.geometry import mesh_vertex_normals
from src.vtk.create import create_polydata
from src.vtk.extract import extract_vtp_points_cells
from src.vtk.fields import add_point_field
from src.vtk.io import load_vtp, save_vtp
from src.vtk.surface_measures import compute_vertex_roughness
from src.vtk.process import extract_polydata_vertex_subset

from src.transforms.similarity import median_scale


def fuse_reference_points(ref_points, ref_normals, subject_data, sigma, max_dist_sigma=3.0, normal_power=2.0):
    """
    For each reference point, pools the nearest point from every subject
    (one batched KDTree query per subject, vectorized over all reference
    points at once - same pattern as scan_consistency's consistency scoring)
    into a Gaussian-weighted average position. ``subject_data`` is a list of
    ``(points, normals, tree)`` tuples, one per subject.

    Returns ``(fused, support, n_contributors, has_support, fit_residual)``:
    fused ``[N, 3]`` positions (only valid where ``has_support``), ``support``
    the total kernel weight behind each fused position, ``n_contributors`` how
    many subjects had a point within ``max_dist_sigma * sigma`` AND non-zero
    normal-agreement weight, and ``fit_residual`` the weighted RMS distance of
    each contributing point from the fused position - a measure of how
    naturally the contributors agree, independent of how much weight backs
    them (computed in the same pass via the weighted second-moment identity
    Var(X) = E[X.X] - |E[X]|^2, so no second loop over subjects is needed).
    """
    n_ref = len(ref_points)
    weighted_sum = np.zeros((n_ref, 3), dtype=np.float64)
    weighted_sum_sq = np.zeros(n_ref, dtype=np.float64)
    total_weight = np.zeros(n_ref, dtype=np.float64)
    n_contributors = np.zeros(n_ref, dtype=np.int32)

    two_sigma_sq = 2.0 * (sigma**2)
    max_dist = max_dist_sigma * sigma

    for points, normals, tree in subject_data:
        dist, idx = tree.query(ref_points, k=1, workers=-1)
        within_range = dist <= max_dist

        weight = np.zeros(n_ref, dtype=np.float64)
        weight[within_range] = np.exp(-(dist[within_range] ** 2) / two_sigma_sq)

        if normal_power > 0:
            cos_sim = np.einsum("ij,ij->i", ref_normals, normals[idx])
            weight *= np.clip(cos_sim, 0.0, 1.0) ** normal_power

        matched = points[idx]
        weighted_sum += matched * weight[:, None]
        weighted_sum_sq += weight * np.einsum("ij,ij->i", matched, matched)
        total_weight += weight
        n_contributors += (weight > 1e-6).astype(np.int32)

    has_support = total_weight > 1e-12
    fused = np.zeros_like(weighted_sum)
    fused[has_support] = weighted_sum[has_support] / total_weight[has_support, None]

    mean_sq_dist = np.zeros(n_ref, dtype=np.float64)
    mean_sq_dist[has_support] = weighted_sum_sq[has_support] / total_weight[has_support]
    variance = np.clip(mean_sq_dist - np.einsum("ij,ij->i", fused, fused), 0.0, None)
    fit_residual = np.sqrt(variance)

    return fused, total_weight, n_contributors, has_support, fit_residual


def run(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)

    subject_files = sorted(glob.glob(os.path.join(cfg.input_aligned_dir, "*.vtp")))
    if not subject_files:
        raise FileNotFoundError(f"No .vtp files found in '{cfg.input_aligned_dir}'.")

    print(f"Loading {len(subject_files)} aligned subject meshes to fuse...")
    subject_data = []
    for f in subject_files:
        pts, faces = extract_vtp_points_cells(load_vtp(f))
        normals = mesh_vertex_normals(verts=pts, faces=faces)
        subject_data.append((pts, normals, cKDTree(pts)))

    reference = load_vtp(cfg.reference_vtp)
    ref_points, ref_faces = extract_vtp_points_cells(reference)
    ref_normals = mesh_vertex_normals(verts=ref_points, faces=ref_faces)
    print(
        f"Reference topology: {len(ref_points):,} points, {len(ref_faces):,} triangles "
        f"({os.path.basename(cfg.reference_vtp)})."
    )

    sigma = cfg.sigma_mm
    if cfg.transforms_dir:
        sigma = cfg.sigma_mm * median_scale(cfg.transforms_dir)
    normal_note = f", normal_power = {cfg.normal_power:g}" if cfg.normal_power > 0 else " (normal weighting disabled)"
    print(f"Fusing with sigma = {cfg.sigma_mm} mm -> {sigma:.4f} template units, min_support = {cfg.min_support:g}{normal_note}...")

    fused, support, n_contributors, has_support, fit_residual = fuse_reference_points(
        ref_points, ref_normals, subject_data, sigma, normal_power=cfg.normal_power
    )

    keep_mask = has_support & (support >= cfg.min_support)
    print(f"{keep_mask.sum():,} of {len(ref_points):,} reference vertices kept (support >= {cfg.min_support:g}).")
    if not keep_mask.any():
        raise ValueError(f"No reference vertex reached min_support={cfg.min_support:g}; lower it or increase sigma_mm.")

    fused_mesh = create_polydata(fused, ref_faces)
    add_point_field(fused_mesh, support.astype(np.float32), field_name="support")
    add_point_field(fused_mesh, n_contributors.astype(np.float32), field_name="n_contributors")
    add_point_field(fused_mesh, fit_residual.astype(np.float32), field_name="fit_residual")

    # fused is still a 1:1, same-order copy of ref_points here (keep_mask
    # hasn't been applied yet), so this is the last point where "index i"
    # unambiguously means "reference vertex i" on both sides - carry the
    # reference's own fields through, and stamp that index itself as
    # "reference_vertex_id" so it survives the reindexing below.
    reference_vertex_id = np.arange(len(ref_points), dtype=np.int32)
    add_point_field(fused_mesh, reference_vertex_id, field_name="reference_vertex_id")
    ref_point_data = reference.GetPointData()
    for a in range(ref_point_data.GetNumberOfArrays()):
        name = ref_point_data.GetArrayName(a)
        add_point_field(fused_mesh, numpy_support.vtk_to_numpy(ref_point_data.GetArray(a)), field_name=name)

    fused_mesh = extract_polydata_vertex_subset(fused_mesh, keep_mask)

    print("compute post-fusion surface roughness (how naturally each vertex sits among its mesh neighbors)")
    compute_vertex_roughness(
        fused_mesh,
        n_rings=cfg.surface_fit_n_rings,
        scale_invariant=False,
        field_name="roughness",
        method=cfg.surface_fit_method,
        neighbors="knn",
        max_neighbors=32,
    )

    output_path = os.path.join(cfg.output_dir, "fused_scan.vtp")
    save_vtp(fused_mesh, output_path)
    print(f"Saved fused consensus scan to: {output_path}")

    # Copy of the reference mesh (untouched geometry/fields) plus the same
    # "reference_vertex_id" join key, saved alongside the fused output so
    # the two files can be matched up without depending on cfg.reference_vtp's
    # own path or on fused_mesh's point order (which keep_mask disturbed).
    reference_out = create_polydata(ref_points, ref_faces)
    for a in range(ref_point_data.GetNumberOfArrays()):
        name = ref_point_data.GetArrayName(a)
        add_point_field(reference_out, numpy_support.vtk_to_numpy(ref_point_data.GetArray(a)), field_name=name)
    add_point_field(reference_out, reference_vertex_id, field_name="reference_vertex_id")
    reference_output_path = os.path.join(cfg.output_dir, "reference.vtp")
    save_vtp(reference_out, reference_output_path)
    print(f"Saved reference mesh (with correspondence field) to: {reference_output_path}")

    return [output_path, reference_output_path]
