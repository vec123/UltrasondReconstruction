"""Per-vertex curvature signals on triangle meshes, computed the same way on any mesh.

The fshape data term (losses/fshape.py) compares a signal carried by the
template with the same signal on the target, so both must come from ONE
definition at ONE scale -- the ``curvature`` arrays already on the meshes do not
qualify (different tools, and units differing by ~4 orders of magnitude).

Principal curvatures come from a Gaussian-weighted quadric fit in each vertex's
own tangent frame, over every mesh vertex within ``radius``. The radius is the
smoothing scale: well above the scan noise (~1 mm on the fused ultrasound
surfaces) and below the features that should stand out (alar crease ~3-5 mm).
"""

from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from src.geometry.geometry import mesh_vertex_normals

SIGNAL_KINDS = ("curvedness", "mean_curvature")
_MIN_NEIGHBOURS = 6   # a quadric has 6 coefficients


def principal_curvatures(points: np.ndarray, faces: np.ndarray, radius: float,
                         query: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """k1 >= k2 at each queried vertex, in 1 / (units of ``points``).

    Fits ``z = a x^2 + b xy + c y^2 + d x + e y + g`` to the neighbours within
    ``radius``, in the frame (u, v, n) of the vertex normal n, weighted by
    ``exp(-|p - p0|^2 / (2 (radius/2)^2))``. The shape operator is approximated
    by the Hessian ``[[2a, b], [b, 2c]]`` (the linear terms absorb a slightly
    tilted normal). Signs follow the normal orientation: positive curvature bends
    TOWARD n.

    Args:
        points: [N, 3] vertex positions
        faces: [F, 3] triangles, used for the vertex normals only
        radius: neighbourhood radius, same units as points
        query: vertex indices to evaluate (default: all); neighbours are always
            drawn from the whole mesh.

    Returns:
        (k1, k2), each [len(query)]; NaN where fewer than 6 neighbours or a
        zero normal left the fit undetermined.
    """
    points = np.asarray(points, dtype=np.float64)
    normals = mesh_vertex_normals(points, np.asarray(faces))
    query = np.arange(len(points)) if query is None else np.asarray(query)
    tree = cKDTree(points)
    neighbourhoods = tree.query_ball_point(points[query], r=radius)
    h2 = 2.0 * (radius / 2.0) ** 2

    k1 = np.full(len(query), np.nan)
    k2 = np.full(len(query), np.nan)
    for out, (i, nb) in enumerate(zip(query, neighbourhoods)):
        n = normals[i]
        if len(nb) < _MIN_NEIGHBOURS or not np.any(n):
            continue
        helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = helper - helper.dot(n) * n
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        d = points[nb] - points[i]
        x, y, z = d @ u, d @ v, d @ n
        sw = np.sqrt(np.exp(-(d * d).sum(1) / h2))
        design = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], axis=1)
        coef, *_ = np.linalg.lstsq(design * sw[:, None], z * sw, rcond=None)
        a, b, c = coef[:3]
        eig = np.linalg.eigvalsh(np.array([[2 * a, b], [b, 2 * c]]))
        k1[out], k2[out] = eig[1], eig[0]
    return k1, k2


def curvature_signal(points: np.ndarray, faces: np.ndarray, radius: float, kind: str,
                     query: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
    """One scalar per queried vertex: ``curvedness`` or ``mean_curvature``.

    curvedness ``sqrt((k1^2 + k2^2) / 2)`` is unsigned, so independent of normal
    orientation: high on ridges, creases and tips, low on flat regions, without
    telling convex from concave. mean_curvature ``(k1 + k2) / 2`` does tell them
    apart, but its sign follows the normals -- two meshes compared by it must be
    oriented consistently (see losses/fshape.build_fshape_kwargs).

    Undetermined vertices (see principal_curvatures) are set to 0, i.e. "flat".

    Returns:
        (signal [len(query)], number of undetermined vertices set to 0)

    Raises:
        ValueError: unknown kind.
    """
    if kind not in SIGNAL_KINDS:
        raise ValueError(f"unknown curvature signal {kind!r}; expected one of {SIGNAL_KINDS}")
    k1, k2 = principal_curvatures(points, faces, radius, query)
    signal = np.sqrt(0.5 * (k1 ** 2 + k2 ** 2)) if kind == "curvedness" else 0.5 * (k1 + k2)
    missing = ~np.isfinite(signal)
    signal[missing] = 0.0
    return signal, int(missing.sum())
