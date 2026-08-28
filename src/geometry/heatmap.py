"""Heat-diffusion landmark fields, shared by mean_shape.py and place_landmarks.py.

Heat released at a landmark and diffused over the surface, following the mesh
geometry rather than straight-line distance - two points on opposite sides of
a fold stay far apart even where the Euclidean distance between them is
small. One backward Euler step of the heat equation, ``(M + tL)u = M delta``,
with ``L`` the cotangent Laplacian and ``M`` the mass matrix; the operator is
the same for every landmark, so it is factorised once and re-solved per
right-hand side. Larger ``t`` gives a wider, smoother bump; by Varadhan's
formula, ``sqrt(-4t log u)`` approaches geodesic distance as ``t`` goes to
zero.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def mean_edge_length(vertices: np.ndarray, faces: Optional[np.ndarray]) -> float:
    """Average triangle edge length, the natural unit for a diffusion time."""
    if faces is None or not len(faces):
        from scipy.spatial import cKDTree
        distance, _ = cKDTree(vertices).query(vertices, k=2)
        return float(distance[:, 1].mean())

    corners = vertices[faces]
    edges = np.concatenate([
        corners[:, 1] - corners[:, 0],
        corners[:, 2] - corners[:, 1],
        corners[:, 0] - corners[:, 2],
    ])
    return float(np.linalg.norm(edges, axis=1).mean())


def landmark_heatmaps(
    vertices: np.ndarray,
    faces: Optional[np.ndarray],
    names: Sequence[str],
    indices: np.ndarray,
    *,
    want_per_landmark: bool,
    want_combined: bool,
    combined_mode: str = "max",
    heat_sigma: Optional[float] = None,
    heat_normalize: bool = True,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    """Returns (per-landmark heatmaps, combined heatmap).

    ``per_landmark`` is a dict of ``float32`` arrays, one per name in
    ``names``, empty unless ``want_per_landmark``. ``combined`` is a single
    ``float32`` array, or ``None`` unless ``want_combined``.

    ``combined_mode``:
      max  element-wise maximum of the per-landmark maps, each normalised to
           peak 1, so every landmark reads exactly 1 at its own vertex
           whatever its neighbours are doing.
      sum  heat from a sum of Dirac deltas. Physically the field of all
           landmarks releasing at once, but overlapping bumps add, so a dense
           cluster outweighs an isolated landmark.
    """
    import robust_laplacian
    from scipy.sparse.linalg import splu

    if faces is not None and len(faces):
        laplacian, mass = robust_laplacian.mesh_laplacian(vertices, faces)
    else:
        laplacian, mass = robust_laplacian.point_cloud_laplacian(vertices)

    sigma = heat_sigma
    if sigma is None:
        sigma = 10.0 * mean_edge_length(vertices, faces)
        if log:
            log(f"  heat sigma {sigma:.4e} (auto: 10 x mean edge length)")
    elif log:
        log(f"  heat sigma {sigma:.4e} ({sigma / mean_edge_length(vertices, faces):.1f} edges)")
    if sigma <= 0:
        raise ValueError(f"heat_sigma must be positive, got {sigma}")
    time = sigma ** 2

    solve = splu((mass + time * laplacian).tocsc()).solve

    def diffuse(idx) -> np.ndarray:
        """Heat from a sum of Dirac deltas at ``idx``."""
        source = np.zeros(len(vertices))
        source[idx] = 1.0
        return solve(mass @ source)

    def to_peak(heat: np.ndarray) -> np.ndarray:
        peak = heat.max()
        return heat / peak if peak > 0 else heat

    combined_needs_each = want_combined and combined_mode == "max"
    per_landmark_raw: Dict[str, np.ndarray] = {}
    if want_per_landmark or combined_needs_each:
        per_landmark_raw = {name: diffuse([index]) for name, index in zip(names, indices)}

    per_landmark: Dict[str, np.ndarray] = {}
    if want_per_landmark:
        for name, heat in per_landmark_raw.items():
            values = to_peak(heat) if heat_normalize else heat
            per_landmark[name] = values.astype(np.float32)

    combined = None
    if want_combined:
        if combined_mode == "max":
            combined = np.maximum.reduce([to_peak(heat) for heat in per_landmark_raw.values()])
        else:
            combined = diffuse(indices)
            if heat_normalize:
                combined = to_peak(combined)
        combined = combined.astype(np.float32)

    return per_landmark, combined
