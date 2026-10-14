"""Stage: raw NRRD ultrasound volumes -> cleaned surface meshes (VTP).

Extracts a level-set surface from each volume, cleans/decimates it, and
attaches per-vertex vertex-position and normal fields (used by downstream
spectral/shape-model code). Optionally also removes large flat patches and
attaches a curvedness field and Laplace-Beltrami operators.

Roughness-based filtering used to live here too; it's now the standalone
``roughness_filter`` stage, run against this stage's output.

This is the first stage in the pipeline: its output mesh defines the
"original frame" that every later alignment stage's saved transform can map
back to.
"""

import glob
import os
from typing import List

import torch

from src.config.US.schema import NrrdToVtpConfig as Config
from src.geometry import (
    compute_mean_curvature,
    from_basis,
    get_operators,
    mesh_vertex_normals,
    to_basis,
)
from src.nrrd.preprocess import convert_single_nrrd, volume_dimension
from src.vtk.extract import get_vtp_faces, get_vtp_vertices, threshold_vtp_by_point_field
from src.vtk.fields import add_point_field
from src.vtk.io import save_vtp
from src.vtk.surface_measures import compute_flat_region_area, compute_vertex_curvedness
from src.vtk.process import clean_vtp, filter_largest_component, reduce_vtp


def process_one(nrrd_path, cfg: Config):
    """Runs the full extract -> clean -> filter -> field-attach chain on one NRRD volume."""
    print("nrrd to vtp")
    vtp = convert_single_nrrd(
        nrrd_path,
        cfg.contour_value,
        auto_contour=cfg.auto_contour,
        denoise=cfg.volume.denoise,
        denoise_radius=cfg.volume.denoise_radius,
        normalize=cfg.volume.normalize,
        clip_percentiles=cfg.volume.clip_percentiles,
        out_max=cfg.volume.norm_out_max,
        crop=cfg.volume.crop,
        crop_threshold=cfg.volume.crop_threshold,
        crop_margin=cfg.volume.crop_margin,
        four_d=cfg.volume.four_d,
    )

    print("clean mesh vtp")
    vtp = clean_vtp(vtp)

    print("keep largest component to vtp")
    vtp = filter_largest_component(vtp)

    print("reduce mesh vtp")
    vtp = reduce_vtp(vtp, reduction_factor=cfg.reduction_factor)

    if cfg.compute_curvedness:
        print("compute vertex curvedness (planar ~0, curved high)")
        compute_vertex_curvedness(
            vtp,
            n_rings=cfg.curvedness_n_rings,
            scale_invariant=cfg.curvedness_scale_invariant,
            field_name="curvedness",
        )

    if cfg.flat.enabled:
        print("compute flat region areas (big planar patches light up)")
        compute_flat_region_area(
            vtp,
            n_rings=cfg.flat.n_rings,
            flat_thresh=cfg.flat.thresh,
            field_name="flat_region_area",
        )
        if cfg.flat.max_area > 0:
            print(f"remove flat patches larger than {cfg.flat.max_area:.0%} of the surface")
            vtp = threshold_vtp_by_point_field(
                vtp, "flat_region_area", cfg.flat.max_area, keep="below"
            )

    print("keep largest component (post flat-region removal, if any)")
    if cfg.min_bridge_width > 0:
        vtp = filter_largest_component(vtp, cfg.min_bridge_width, relative=cfg.min_bridge_relative)
    else:
        vtp = filter_largest_component(vtp)

    # Per-vertex data must match the FILTERED mesh, so extract/compute after filtering.
    mesh_verts = get_vtp_vertices(vtp)
    mesh_faces = get_vtp_faces(vtp)
    normals_np = mesh_vertex_normals(verts=mesh_verts, faces=mesh_faces)
    # Normals are already unit-length; do NOT min-max them, that would destroy
    # their unit-vector meaning.
    compute_mean_curvature(torch.tensor(mesh_verts), torch.tensor(mesh_faces))

    frames = mass = evals = evecs = None
    if cfg.operators:
        _, mass, _, evals, evecs, _, _, _ = get_operators(
            mesh_verts, mesh_faces, k_eig=cfg.k_eig, op_cache_dir=cfg.cache_dir
        )

    functions = {"vertices": mesh_verts, "normals": normals_np}
    for key, function in functions.items():
        if cfg.operators:
            coeffs = to_basis(function, evecs, mass)
            func_reconstructed = from_basis(coeffs, evecs)

        for i in range(function.shape[1]):
            values = function[:, i]
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            add_point_field(vtp, values, field_name=f"True {key} {i}")
            if cfg.operators:
                add_point_field(vtp, func_reconstructed[:, i], field_name=f"recon {key} {i}")

    return vtp


def run(cfg: Config) -> List[str]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)
    nrrd_files = sorted(glob.glob(os.path.join(cfg.input_dir, "*.nrrd")))
    if not nrrd_files:
        raise ValueError(f"No .nrrd files found in {cfg.input_dir}")

    print(f"Found {len(nrrd_files)} .nrrd files to process.\n" + "-" * 40)

    output_paths = []
    for nrrd_path in nrrd_files:
        base_name = os.path.splitext(os.path.basename(nrrd_path))[0]
        output_path = os.path.join(cfg.output_dir, f"{base_name}.vtp")

        if cfg.volume.four_d == "skip" and volume_dimension(nrrd_path) == 4:
            print(f"  warning: skipping 4D (x, y, z, t) volume '{base_name}' - set volume.four_d "
                  f"to 'mean', 'median' or a frame index to load it as one scan.")
            # Later stages glob output_dir, so a mesh left from an earlier run would still go in.
            if os.path.exists(output_path):
                os.remove(output_path)
                print(f"  warning: removed its old mesh '{output_path}'.")
            continue

        print(f"Processing: {base_name}")

        vtp = process_one(nrrd_path, cfg)

        save_vtp(vtp, output_path)
        output_paths.append(output_path)

    if not output_paths:
        raise ValueError(f"Every .nrrd file in {cfg.input_dir} is 4D and volume.four_d is 'skip'.")
    return output_paths
