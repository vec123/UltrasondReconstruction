"""Flat re-export of geometry.py's public surface.

Several call sites already write ``from src.geometry import X`` expecting
this package to behave like one flat module (mirroring how the file reads).
This re-exports the functions those call sites - and their natural siblings -
actually use; a niche function that isn't part of this confirmed surface
(e.g. ``compute_hks``, ``find_knn``) is still reachable via
``src.geometry.geometry`` directly.
"""

from src.geometry.geometry import (
    add_point_field,
    compute_mean_curvature,
    compute_operators,
    convert_single_nrrd,
    convert_vtu_to_vtp_vtk,
    create_polydata,
    crop_volume,
    decompose_similarity_transform,
    ensure_dir_exists,
    extract_arrays_from_polydata,
    filter_largest_component,
    from_basis,
    get_all_operators,
    get_operators,
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
    to_basis,
    toNP,
    umeyama_similarity_transform,
    vertex_normals,
    vertex_normals_numpy,
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
    "vertex_normals_numpy",
]
