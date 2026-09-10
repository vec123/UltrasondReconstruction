"""Per-vertex surface measures computed on vtkPolyData."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from src.vtk.extract import extract_vtp_points_cells
from src.vtk.fields import add_point_field


def _vertex_normals(verts, faces):
    """Area-weighted unit vertex normals (numpy in / numpy out)."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)  # magnitude ~ 2 * face area
    n_pts = verts.shape[0]
    vertex_normals = np.zeros((n_pts, 3))
    # bincount per corner/component is ~10x faster than np.add.at scatter.
    for i in range(3):
        for c in range(3):
            vertex_normals[:, c] += np.bincount(faces[:, i], weights=face_n[:, c], minlength=n_pts)
    return vertex_normals / (np.linalg.norm(vertex_normals, axis=-1, keepdims=True) + 1e-12)


def _tangent_frames(normals):
    """Orthonormal tangent vectors (t1, t2) for each unit normal."""
    # Pick the world axis least aligned with each normal, then Gram-Schmidt.
    a = np.zeros_like(normals)
    a[np.arange(normals.shape[0]), np.argmin(np.abs(normals), axis=1)] = 1.0
    t1 = np.cross(normals, a)
    t1 = t1 / (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12)
    t2 = np.cross(normals, t1)
    return t1, t2


def _stride_subsample_pairs(vi, vj, n_pts, max_neighbors):
    """Keep <= ~max_neighbors pairs per vertex (every stride-th within each row).

    Requires ``vi`` sorted (which COO-from-CSR guarantees)."""
    counts = np.bincount(vi, minlength=n_pts)
    counts_rep = np.repeat(counts, counts)
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    rank = np.arange(vi.size, dtype=np.int64) - offsets
    stride = np.maximum(counts_rep // max_neighbors, 1)
    keep = (rank % stride) == 0
    return vi[keep], vj[keep]


def _neighborhood_pairs(verts, faces, n_rings, neighbors="knn", max_neighbors=32, normals=None):
    """(i, j) index pairs pairing every vertex i with its spatial neighborhood.

    ``neighbors="knn"`` (fast, default): the k nearest vertices by Euclidean
    distance, with k sized to match an exact n-ring disc (~3*r*(r+1) vertices on
    a valence-6 triangulation) so the measurement SCALE is preserved. When
    ``max_neighbors`` caps the count, a fixed random subset of neighbor ranks is
    used -- near and far ranks alike -- which keeps the spatial extent while
    cutting the pair count. Pairs whose vertex normals oppose (dot <= 0) are
    dropped so nearby back-to-back sheets don't leak into each other's
    neighborhoods (requires ``normals``).

    ``neighbors="rings"`` (exact, slow): the original graph n-ring via sparse
    boolean reachability; ``max_neighbors`` still applies (strided subsample).
    """
    n_pts = verts.shape[0]

    if neighbors == "knn":
        k_ring = 3 * n_rings * (n_rings + 1)  # ~ n-ring disc size, center excluded
        k = min(k_ring, n_pts - 1)
        if max_neighbors and k > max_neighbors:
            rng = np.random.default_rng(0)
            cols = np.sort(rng.choice(np.arange(1, k + 1), size=max_neighbors, replace=False))
        else:
            cols = np.arange(1, k + 1)

        tree = cKDTree(verts)
        vi_parts, vj_parts = [], []
        chunk = 200_000  # bound the (chunk, k+1) index buffer
        for start in range(0, n_pts, chunk):
            stop = min(start + chunk, n_pts)
            _, idx = tree.query(verts[start:stop], k=k + 1, workers=-1)  # col 0 = self
            sub = idx[:, cols].astype(np.int32)
            vi_parts.append(np.repeat(np.arange(start, stop, dtype=np.int32), sub.shape[1]))
            vj_parts.append(sub.ravel())
        vi = np.concatenate(vi_parts)
        vj = np.concatenate(vj_parts)

        if normals is not None:
            same_side = np.einsum("ij,ij->i", normals[vi], normals[vj]) > 0.0
            vi, vj = vi[same_side], vj[same_side]
        return vi, vj

    if neighbors == "rings":
        # 1-ring vertex adjacency from the triangle edges (duplicates collapse to one entry).
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
        adj = sp.csr_matrix(
            (np.ones(edges.shape[0], dtype=np.float32), (edges[:, 0], edges[:, 1])),
            shape=(n_pts, n_pts),
        )
        adj = ((adj + adj.T) > 0).astype(np.float32)

        # Expand to the n-ring neighborhood via boolean reachability.
        reach = adj
        for _ in range(n_rings - 1):
            reach = ((reach + reach @ adj) > 0).astype(np.float32)

        coo = reach.tocoo()
        off_diag = coo.row != coo.col
        vi, vj = coo.row[off_diag], coo.col[off_diag]
        if max_neighbors:
            vi, vj = _stride_subsample_pairs(vi, vj, n_pts, max_neighbors)
        return vi, vj

    raise ValueError(f"Unknown neighbors {neighbors!r}; expected 'knn' or 'rings'.")


def _fit_local_quadrics(vi, disp, normals, n_pts):
    """Batched per-vertex least-squares quadric fit in the tangent frame.

    For each vertex a quadric ``h(u, v) = a*u + b*v + c*u^2 + d*u*v + e*v^2`` is
    fitted to its neighbors (u, v tangent coordinates, h signed height along the
    normal). The normal equations are accumulated per vertex with bincount and
    solved as a batched, ridge-regularized 5x5 system -- no per-vertex loop.

    Returns ``(beta [N,5], M [N,5,5], rhs [N,5], h_sq_sum [N])`` where ``beta``
    holds the coefficients (a, b, c, d, e) per vertex.
    """
    t1, t2 = _tangent_frames(normals)
    u = np.einsum("ij,ij->i", disp, t1[vi])
    v = np.einsum("ij,ij->i", disp, t2[vi])
    h = np.einsum("ij,ij->i", disp, normals[vi])  # signed height above tangent plane

    phi = np.stack([u, v, u * u, u * v, v * v], axis=1)  # [P, 5] design rows
    k = phi.shape[1]

    M = np.zeros((n_pts, k, k))
    rhs = np.zeros((n_pts, k))
    for a in range(k):
        rhs[:, a] = np.bincount(vi, weights=phi[:, a] * h, minlength=n_pts)
        for b in range(a, k):
            m_ab = np.bincount(vi, weights=phi[:, a] * phi[:, b], minlength=n_pts)
            M[:, a, b] = m_ab
            M[:, b, a] = m_ab
    h_sq_sum = np.bincount(vi, weights=h * h, minlength=n_pts)

    # Ridge keeps degenerate / tiny neighborhoods solvable (isolated vertex -> beta=0).
    trace = np.trace(M, axis1=1, axis2=2)
    lam = 1e-8 * trace / k + 1e-12
    M_reg = M + lam[:, None, None] * np.eye(k)
    beta = np.linalg.solve(M_reg, rhs[..., None])[..., 0]
    return beta, M, rhs, h_sq_sum


def _quadric_residual_sq_sum(vi, disp, normals, n_pts):
    """Per-vertex sum of squared residuals from the local quadric fit."""
    beta, M, rhs, h_sq_sum = _fit_local_quadrics(vi, disp, normals, n_pts)
    # ||h - phi @ beta||^2 accumulated per vertex, without materializing residuals:
    res_sq = (
        h_sq_sum
        - 2.0 * np.einsum("nk,nk->n", beta, rhs)
        + np.einsum("nk,nkl,nl->n", beta, M, beta)
    )
    return np.maximum(res_sq, 0.0)  # clamp negative round-off


def compute_vertex_normals(polydata, field_name="normal", weighting="unit"):
    """Assign each vertex the unit normal of the surface it sits on.

    The ONE place a persisted ``normal`` point array is produced, so the
    convention cannot drift between the pipeline stage that bakes it
    (scan_fusion) and the standalone re-bake (scripts/write_surface_normals.py).
    ``io.load_shape`` reads an array of this name into ``Shape.normals``, which
    is what the point-to-plane data term projects onto.

    ``weighting`` picks how the incident faces are averaged:

    - ``"unit"``: mean of the incident faces' UNIT normals. A sliver counts as
      much as a large triangle. Matches vtkPolyDataNormals (verified to
      |cos| > 0.9999 on 99.9% of a fused scan's vertices) and is what
      ``scan_fusion`` already uses for its normal-agreement weighting.
    - ``"area"``: area-weighted, via ``_vertex_normals``. More robust on
      irregular triangulations, where a sliver's noisy normal is otherwise
      counted in full.

    Derived from FACES, never estimated from the point cloud: a mesh without
    polys raises rather than silently falling back to a kNN-PCA fit, which is a
    different object with different error characteristics.

    Vertices whose incident faces are degenerate or cancel come back as the ZERO
    vector, never NaN (see geometry.mesh_vertex_normals). Consumers must treat
    zero as "no tangent plane here" -- the point-to-plane term scores them at
    zero plane-distance, falling back to its point-to-point component.

    If ``field_name`` is not None the result is attached to ``polydata`` as a
    3-component point field. Returns the normals as an ``[N, 3]`` numpy array.
    """
    verts, faces = extract_vtp_points_cells(polydata)
    if faces.size == 0:
        raise ValueError(
            f"compute_vertex_normals needs a triangulated mesh; polydata has "
            f"{verts.shape[0]} points and no polys. A bare point cloud would need "
            f"estimated (kNN-PCA) normals -- see geometry.vertex_normals -- which "
            f"is a deliberate choice, not a fallback."
        )

    if weighting == "unit":
        # Local import: geometry pulls in heavier deps than this module's own.
        from src.geometry import mesh_vertex_normals
        normals = mesh_vertex_normals(verts=verts, faces=faces)
    elif weighting == "area":
        normals = _vertex_normals(verts, faces)
    else:
        raise ValueError(f"unknown weighting {weighting!r}; expected 'unit' or 'area'")

    # Belt and braces: both helpers guard their own division, but a NaN reaching
    # a training loss is unrecoverable and silent until the loss itself goes NaN,
    # so never let one leave this function.
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)

    if field_name is not None:
        add_point_field(polydata, normals.astype(np.float32), field_name=field_name)
    return normals


def compute_vertex_roughness(polydata, n_rings=2, scale_invariant=True, field_name="roughness",
                             method="plane", neighbors="knn", max_neighbors=32):
    """Assign each vertex a scalar reflecting local surface roughness.

    ``method`` selects the reference surface the ``n_rings``-ring neighbors are
    measured against:

    - ``"plane"``: RMS distance from the tangent plane at the vertex. Flat
      regions -> ~0, but smooth curved regions keep a curvature baseline.
    - ``"quadric"``: RMS residual from a least-squares quadric fitted to the
      neighborhood in the tangent frame. Smooth curvature is absorbed by the
      fit, isolating bump/noise roughness. Needs enough neighbors for the
      5-parameter fit -- with ``n_rings=1`` on low-valence vertices the quadric
      can interpolate the data and roughness collapses to ~0; keep
      ``n_rings >= 2``.

    With ``scale_invariant=True`` the RMS deviation is divided by the mean
    neighbor distance, making the value dimensionless and comparable across
    meshes of different scale / decimation level.

    ``neighbors`` / ``max_neighbors`` control the neighborhood construction
    (see ``_neighborhood_pairs``): "knn" is a fast Euclidean approximation of
    the n-ring at the same spatial scale, "rings" the exact-but-slow graph ring.

    If ``field_name`` is not None the result is attached to ``polydata`` as a
    point scalar field. Returns the roughness as a ``[N]`` numpy array.
    """
    verts, faces = extract_vtp_points_cells(polydata)
    if faces.size == 0:
        raise ValueError("compute_vertex_roughness needs a triangulated mesh; polydata has no polys.")

    n_pts = verts.shape[0]
    normals = _vertex_normals(verts, faces)
    vi, vj = _neighborhood_pairs(verts, faces, n_rings, neighbors=neighbors,
                                 max_neighbors=max_neighbors, normals=normals)
    disp = verts[vj] - verts[vi]
    counts = np.bincount(vi, minlength=n_pts)
    counts_safe = np.maximum(counts, 1)  # isolated vertices keep roughness 0

    if method == "plane":
        # Distance of each neighbor from the tangent plane at vertex i.
        plane_dist = np.einsum("ij,ij->i", disp, normals[vi])
        dev_sq_sum = np.bincount(vi, weights=plane_dist ** 2, minlength=n_pts)
    elif method == "quadric":
        dev_sq_sum = _quadric_residual_sq_sum(vi, disp, normals, n_pts)
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'plane' or 'quadric'.")

    roughness = np.sqrt(dev_sq_sum / counts_safe)

    if scale_invariant:
        neigh_dist = np.linalg.norm(disp, axis=1)
        mean_dist = np.bincount(vi, weights=neigh_dist, minlength=n_pts) / counts_safe
        roughness = roughness / (mean_dist + 1e-12)

    roughness = roughness.astype(np.float32)

    if field_name is not None:
        add_point_field(polydata, roughness, field_name=field_name)

    return roughness


def compute_vertex_curvedness(polydata, n_rings=2, scale_invariant=True, field_name="curvedness",
                              neighbors="knn", max_neighbors=32):
    """Assign each vertex a scalar showing how curved (vs planar) its region is.

    Fits the same local quadric as ``compute_vertex_roughness(method="quadric")``
    and reads the principal curvatures off its second-order coefficients: the
    Hessian of ``h(u, v)`` is ``[[2c, d], [d, 2e]]``, whose eigenvalues k1, k2
    approximate the principal curvatures (exact up to the small fitted tangent
    slopes). Curvedness is Koenderink's ``sqrt((k1^2 + k2^2) / 2)``, i.e.
    ``sqrt(2c^2 + 2e^2 + d^2)``.

    ~0 -> planar region; larger -> more strongly curved. Because the value comes
    from the smooth FIT (not the residuals), bump/noise roughness largely stays
    out of it -- the complement of the "quadric" roughness field.

    Curvature has units 1/length; with ``scale_invariant=True`` it is multiplied
    by the mean neighbor distance, giving a dimensionless value comparable
    across meshes of different scale / decimation level.

    If ``field_name`` is not None the result is attached to ``polydata`` as a
    point scalar field. Returns the curvedness as a ``[N]`` numpy array.
    """
    verts, faces = extract_vtp_points_cells(polydata)
    if faces.size == 0:
        raise ValueError("compute_vertex_curvedness needs a triangulated mesh; polydata has no polys.")

    n_pts = verts.shape[0]
    normals = _vertex_normals(verts, faces)
    vi, vj = _neighborhood_pairs(verts, faces, n_rings, neighbors=neighbors,
                                 max_neighbors=max_neighbors, normals=normals)
    disp = verts[vj] - verts[vi]

    beta, _, _, _ = _fit_local_quadrics(vi, disp, normals, n_pts)
    c, d, e = beta[:, 2], beta[:, 3], beta[:, 4]
    # k1^2 + k2^2 of the Hessian [[2c, d], [d, 2e]] collapses to 4c^2 + 4e^2 + 2d^2.
    curvedness = np.sqrt(2.0 * c ** 2 + 2.0 * e ** 2 + d ** 2)

    if scale_invariant:
        counts_safe = np.maximum(np.bincount(vi, minlength=n_pts), 1)
        mean_dist = np.bincount(vi, weights=np.linalg.norm(disp, axis=1), minlength=n_pts) / counts_safe
        curvedness = curvedness * mean_dist

    curvedness = curvedness.astype(np.float32)

    if field_name is not None:
        add_point_field(polydata, curvedness, field_name=field_name)

    return curvedness


def compute_flat_region_area(polydata, n_rings=5, flat_thresh=0.05, field_name="flat_region_area",
                             neighbors="knn", max_neighbors=32):
    """Per-vertex surface-area fraction of the contiguous FLAT patch containing it.

    Discriminates LARGE planar regions from everything else -- pointwise
    flatness (curvedness ~0) cannot tell a big flat wall from a small flat spot,
    but patch AREA can. A vertex counts as flat when its scale-invariant
    tangent-plane RMS deviation (``compute_vertex_roughness`` with
    ``method="plane"``) over ``n_rings`` is below ``flat_thresh``. Contiguous
    flat vertices are grouped into patches, and every flat vertex is assigned
    its patch's surface area as a FRACTION of the total mesh area; non-flat
    vertices get 0.

    Big planar wall -> large value (its area share). Small flat spot -> tiny
    value. Curved or rough surface -> 0. To delete large flat walls, threshold
    this field with ``keep="below"`` at the maximum tolerated patch size, e.g.
    ``threshold_vtp_by_point_field(vtp, "flat_region_area", 0.05, keep="below")``.

    Patch area (not vertex count) makes this robust to decimation, which
    collapses flat regions into few large triangles.

    If ``field_name`` is not None the result is attached to ``polydata`` as a
    point scalar field. Returns the area fractions as a ``[N]`` numpy array.
    """
    verts, faces = extract_vtp_points_cells(polydata)
    if faces.size == 0:
        raise ValueError("compute_flat_region_area needs a triangulated mesh; polydata has no polys.")
    n_pts = verts.shape[0]

    plane_rms = compute_vertex_roughness(polydata, n_rings=n_rings, scale_invariant=True,
                                         field_name=None, method="plane",
                                         neighbors=neighbors, max_neighbors=max_neighbors)
    flat_mask = plane_rms < flat_thresh

    # Vertex areas: one third of each incident face's area.
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    vert_area = np.zeros(n_pts)
    for i in range(3):
        vert_area += np.bincount(faces[:, i], weights=face_area / 3.0, minlength=n_pts)

    # Connected components over edges whose BOTH endpoints are flat.
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    both_flat = flat_mask[edges[:, 0]] & flat_mask[edges[:, 1]]
    e = edges[both_flat]
    graph = sp.csr_matrix((np.ones(e.shape[0], dtype=np.int8), (e[:, 0], e[:, 1])),
                          shape=(n_pts, n_pts))
    n_comp, labels = connected_components(graph, directed=False)

    # Patch area per component (only flat vertices contribute), as total-area fraction.
    area_per_comp = np.bincount(labels, weights=np.where(flat_mask, vert_area, 0.0),
                                minlength=n_comp)
    flat_area = np.where(flat_mask, area_per_comp[labels], 0.0) / vert_area.sum()

    n_patches = int(np.unique(labels[flat_mask]).size) if flat_mask.any() else 0
    print(f"{int(flat_mask.sum()):,} flat vertices in {n_patches} patch(es); "
          f"largest covers {flat_area.max():.1%} of the surface.")

    flat_area = flat_area.astype(np.float32)
    if field_name is not None:
        add_point_field(polydata, flat_area, field_name=field_name)
    return flat_area
