"""Builds a single vtkPolyData carrying the mean_shape.py field set for one
sample of a shape model - total_std, mode_XX_std, curvature, landmark_distance,
nearest_landmark, heatmap_<name>, heatmap_landmarks - shared by
place_landmarks.py's 'x' export and any one-off script that dumps several
samples at once (e.g. to check whether a landmark tracks its anatomy across
the shape space).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from src.geometry.geometry_np import compute_mean_curvature
from src.geometry.heatmap import landmark_heatmaps
from src.geometry.vtk import add_point_field, create_polydata

# Matches fields.mode_count's default in MeanShapeFields (src/config/schema.py).
DEFAULT_MODE_COUNT = 5


def build_landmark_field_polydata(
    vertices: np.ndarray,
    faces: Optional[np.ndarray],
    placements: Dict[str, int],
    *,
    linear_basis: Optional[np.ndarray] = None,
    linear_sigma: Optional[np.ndarray] = None,
    mode_count: int = DEFAULT_MODE_COUNT,
    log: Optional[Callable[[str], None]] = None,
):
    """(vertices, faces) plus everything mean_shape.py would attach.

    ``linear_basis``/``linear_sigma`` are the *linear* (ambient-PCA)
    eigenfunctions/sqrt(eigenvalues) - total_std and mode_XX_std are always a
    property of that basis, never of whatever parametrization produced
    ``vertices`` (e.g. LDDMM), since mean_shape.py has no LDDMM concept to
    match. Omit them to skip those fields. ``placements`` (name -> vertex
    index) may be a subset of a model's full landmark set; distance/heatmap
    fields are omitted entirely if it is empty.
    """
    polydata = create_polydata(vertices, faces)

    if linear_basis is not None and linear_sigma is not None and linear_basis.shape[1]:
        n_vertices, n_modes = len(vertices), linear_basis.shape[1]
        per_mode_sq = (linear_basis.reshape(n_vertices, 3, n_modes) ** 2).sum(axis=1)
        eigenvalues = linear_sigma ** 2
        add_point_field(polydata, np.sqrt(per_mode_sq @ eigenvalues), "total_std")
        for k in range(min(mode_count, n_modes)):
            mode_std = linear_sigma[k] * np.sqrt(per_mode_sq[:, k])
            add_point_field(polydata, mode_std, f"mode_{k:02d}_std")

    try:
        curvature = compute_mean_curvature(vertices, faces)
        add_point_field(polydata, np.asarray(curvature, dtype=np.float64).reshape(-1), "curvature")
    except Exception as exc:                          # noqa: BLE001 - diagnostic field only
        if log:
            log(f"  curvature skipped: {type(exc).__name__}: {exc}")

    if placements:
        from scipy.spatial import cKDTree

        names = list(placements.keys())
        indices = np.array(list(placements.values()), dtype=np.int64)

        distance, nearest = cKDTree(vertices[indices]).query(vertices)
        add_point_field(polydata, distance.astype(np.float64), "landmark_distance")
        add_point_field(polydata, nearest.astype(np.int32), "nearest_landmark")

        per_landmark, combined = landmark_heatmaps(
            vertices, faces, names, indices,
            want_per_landmark=True, want_combined=True, combined_mode="max", log=log,
        )
        for name, values in per_landmark.items():
            add_point_field(polydata, values, f"heatmap_{'_'.join(str(name).split())}")
        add_point_field(polydata, combined, "heatmap_landmarks")

    return polydata
