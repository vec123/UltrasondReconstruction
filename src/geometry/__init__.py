"""Flat re-export of the geometry package's public surface.

Several call sites already write ``from src.geometry import X`` expecting
this package to behave like one flat module. geometry.py is the numpy-only
implementation (mesh I/O, differential geometry, NRRD preprocessing, Umeyama
registration - everything that doesn't need torch); geometry_torch.py holds
only what genuinely needs to stay torch: compute_operators/get_operators/
get_all_operators/to_basis/from_basis, consumed as torch by the nrrd_to_vtp
preprocessing stage. A niche function that isn't part of this confirmed
surface (e.g. ``compute_hks``, ``find_knn``) is still reachable via
``src.geometry.geometry`` or ``src.geometry.geometry_torch`` directly.
"""

from src.geometry.geometry import (
    add_point_field,
    compute_mean_curvature,
    convert_single_nrrd,
    convert_vtu_to_vtp_vtk,
    create_polydata,
    crop_volume,
    decompose_similarity_transform,
    ensure_dir_exists,
    extract_arrays_from_polydata,
    filter_largest_component,
    get_vtp_faces,
    get_vtp_vertices,
    load_and_preprocess_volume,
    load_vtp_mesh,
    load_vtu,
    mesh_vertex_normals,
    otsu_threshold,
    preprocess_volume,
    save_vtk,
    save_vtp_mesh,
    umeyama_similarity_transform,
    vertex_normals,
)
from src.geometry.geometry_torch import (
    compute_operators,
    from_basis,
    get_all_operators,
    get_operators,
    to_basis,
    toNP,
)

__all__ = [
    "add_point_field",
    "compute_mean_curvature",
    "compute_operators",
    "convert_single_nrrd",
    "convert_vtu_to_vtp_vtk",
    "create_polydata",
    "crop_volume",
    "decompose_similarity_transform",
    "ensure_dir_exists",
    "extract_arrays_from_polydata",
    "filter_largest_component",
    "from_basis",
    "get_all_operators",
    "get_operators",
    "get_vtp_faces",
    "get_vtp_vertices",
    "load_and_preprocess_volume",
    "load_vtp_mesh",
    "load_vtu",
    "mesh_vertex_normals",
    "otsu_threshold",
    "preprocess_volume",
    "save_vtk",
    "save_vtp_mesh",
    "to_basis",
    "toNP",
    "umeyama_similarity_transform",
    "vertex_normals",
]
