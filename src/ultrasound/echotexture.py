"""Per-vertex "echotexture" measures sampled from a volume, not from mesh geometry.

Surface extraction (marching cubes at one contour value) collapses a volume's
full grayscale signal into a binary in/out decision; everything about how a
tissue actually echoes (placenta's homogeneous "cotyledon" speckle vs. fetal
bone's hyperechoic, shadow-casting return) lives in the grey levels that
threshold throws away, not in the resulting surface's shape. These functions
sample a small voxel neighborhood around each surface vertex back in the
source volume and summarize it as local intensity/gradient statistics - a
signal orthogonal to anything geometric (curvature, roughness, ...).

Callers are responsible for passing `points_world` in the SAME frame `volume`'s
own `spacing`/`origin` define (see `src.geometry.load_and_preprocess_volume`).
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def sample_echotexture(points_world, volume, spacing, origin, radius_mm=1.5, interp_order=1):
    """Samples local intensity/gradient statistics from `volume` at `points_world`.

    `volume` is a (Z, Y, X) numpy array (SimpleITK's `GetArrayFromImage` layout);
    `spacing`/`origin` are (sx, sy, sz) / (ox, oy, oz), as returned alongside it
    by `load_and_preprocess_volume` - i.e. world = origin + index * spacing,
    with NO direction-cosine rotation (matching how `convert_single_nrrd`
    builds the `vtkImageData` a surface is contoured from).

    `radius_mm` is the half-width of a cubic neighborhood (converted to a
    per-axis voxel radius via `spacing`, at least 1 voxel). The neighborhood
    mean/variance/gradient-mean are computed with one box filter pass over the
    WHOLE volume (equivalent to, but far cheaper than, gathering and reducing
    the actual per-point window), then bilinearly/nearest-sampled at each
    point - so cost scales with volume size once, not with the number of
    surface points.

    Returns a dict of `[N]` float32 arrays:
      - "intensity": trilinearly interpolated raw intensity at the point.
      - "local_mean" / "local_std": neighborhood intensity mean / std.
      - "gradient_mean" / "gradient_max": neighborhood mean / max of the
        intensity-gradient magnitude - large "gradient_max" with high
        "local_mean" suggests a hyperechoic (bone-like) reflector; a
        posterior drop in "local_mean" right behind such a point is the
        acoustic-shadow signature.
      - "in_bounds": 1.0 where the point falls inside the volume's voxel
        grid, 0.0 where it had to be clamped to the nearest edge voxel
        (`mode="nearest"`) - a large out-of-bounds count usually means
        `crop`/`crop_threshold`/`crop_margin` don't match the ones used to
        extract the surface.
    """
    volume = np.asarray(volume, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)  # (sx, sy, sz)
    origin = np.asarray(origin, dtype=np.float64)  # (ox, oy, oz)
    points_world = np.asarray(points_world, dtype=np.float64)

    spacing_zyx = spacing[::-1]
    radius_vox = np.maximum(np.round(radius_mm / spacing_zyx).astype(int), 1)
    box_size = tuple(int(2 * r + 1) for r in radius_vox)

    local_mean = ndimage.uniform_filter(volume, size=box_size, mode="nearest")
    local_mean_sq = ndimage.uniform_filter(volume ** 2, size=box_size, mode="nearest")
    local_var = np.maximum(local_mean_sq - local_mean ** 2, 0.0)
    del local_mean_sq
    local_std = np.sqrt(local_var)
    del local_var

    # Volume-sized temporaries (gz/gy/gx, grad_mag) are dropped as soon as
    # they're no longer needed - several full-volume float64 arrays alive at
    # once is real memory on a large ultrasound volume.
    gz, gy, gx = np.gradient(volume, spacing_zyx[0], spacing_zyx[1], spacing_zyx[2])
    grad_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    del gz, gy, gx
    grad_mean = ndimage.uniform_filter(grad_mag, size=box_size, mode="nearest")
    grad_max = ndimage.maximum_filter(grad_mag, size=box_size, mode="nearest")
    del grad_mag

    # world -> voxel index, in (x, y, z) order (see module docstring: no direction cosines).
    voxel_xyz = (points_world - origin) / spacing
    dims_xyz = np.array(volume.shape[::-1], dtype=np.float64)
    in_bounds = np.all((voxel_xyz >= 0) & (voxel_xyz <= dims_xyz - 1), axis=1)

    # ndimage.map_coordinates indexes the array in its own (z, y, x) axis order.
    coords_zyx = voxel_xyz[:, ::-1].T

    def sample(field):
        return ndimage.map_coordinates(
            field, coords_zyx, order=interp_order, mode="nearest"
        ).astype(np.float32)

    return {
        "intensity": sample(volume),
        "local_mean": sample(local_mean),
        "local_std": sample(local_std),
        "gradient_mean": sample(grad_mean),
        "gradient_max": sample(grad_max),
        "in_bounds": in_bounds.astype(np.float32),
    }


def sample_structure_tensor(points_world, volume, spacing, origin, radius_mm=1.5, interp_order=1):
    """Samples local gradient-STRUCTURE-TENSOR shape descriptors at `points_world`.

    A different question than `sample_echotexture`'s intensity/gradient
    MAGNITUDE statistics: this asks about local gradient ORIENTATION
    coherence - whether the edges in a neighborhood all point the same way,
    point every-which-way, or point along one shared plane.

    For each voxel, the gradient's outer product is averaged (box filter, same
    `radius_mm` integration window `sample_echotexture` uses) to give a
    smoothed 3x3 structure tensor. Its eigenvalues l1 >= l2 >= l3 >= 0
    describe the local edge geometry - NOTE this is the OPPOSITE pairing from
    Westin's diffusion-tensor shape metrics (there, one dominant eigenvalue
    means a needle-shaped tensor = a fiber/tube): here the tensor is built
    from the GRADIENT, so its dominant eigenvector points along the surface
    NORMAL, and a single dominant eigenvalue means the gradient direction is
    constant across the neighborhood - i.e. one flat SHEET, not a tube:
      - Sheet-like (a face/skull profile, a skin surface): the gradient
        direction barely varies (it's always ~normal to the same flat
        boundary) -> ONE dominant eigenvalue, l1 >> l2 ~ l3 -> high
        "planarity".
      - Tubular (umbilical cord, a limb's long axis): the gradient sweeps
        through every direction PERPENDICULAR to the tube's axis, but never
        points along the axis itself -> TWO comparable dominant eigenvalues,
        l1 ~ l2 >> l3 -> high "linearity".
      - Isotropic (speckle noise, or a flat/featureless patch): l1 ~ l2 ~ l3
        -> high "sphericity".
    "energy" (the trace, l1 + l2 + l3) is the total local gradient energy -
    near zero in a flat, edge-free region, where the three shape ratios above
    are numerically unstable/meaningless (0/~0) - treat a low-"energy"
    point's linearity/planarity/sphericity as unreliable.

    Eigen-decomposition runs AFTER sampling, on the (far smaller) N-point
    tensors, not per voxel - a per-voxel eigendecomposition over the whole
    volume would be both far more compute and far more memory than needed,
    since only the values at the mesh's own surface points are ever used.

    Returns a dict of `[N]` float32 arrays: "linearity", "planarity",
    "sphericity", "energy".
    """
    volume = np.asarray(volume, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)  # (sx, sy, sz)
    origin = np.asarray(origin, dtype=np.float64)  # (ox, oy, oz)
    points_world = np.asarray(points_world, dtype=np.float64)

    spacing_zyx = spacing[::-1]
    radius_vox = np.maximum(np.round(radius_mm / spacing_zyx).astype(int), 1)
    box_size = tuple(int(2 * r + 1) for r in radius_vox)

    gz, gy, gx = np.gradient(volume, spacing_zyx[0], spacing_zyx[1], spacing_zyx[2])

    voxel_xyz = (points_world - origin) / spacing
    coords_zyx = voxel_xyz[:, ::-1].T

    def sample_smoothed_product(a, b):
        smoothed = ndimage.uniform_filter(a * b, size=box_size, mode="nearest")
        return ndimage.map_coordinates(smoothed, coords_zyx, order=interp_order, mode="nearest")

    # One volume-sized product+filter pass at a time (freed once sampled),
    # rather than materializing all six at once.
    sxx = sample_smoothed_product(gx, gx)
    syy = sample_smoothed_product(gy, gy)
    szz = sample_smoothed_product(gz, gz)
    sxy = sample_smoothed_product(gx, gy)
    sxz = sample_smoothed_product(gx, gz)
    syz = sample_smoothed_product(gy, gz)
    del gx, gy, gz

    n_pts = sxx.shape[0]
    tensors = np.empty((n_pts, 3, 3), dtype=np.float64)
    tensors[:, 0, 0], tensors[:, 1, 1], tensors[:, 2, 2] = sxx, syy, szz
    tensors[:, 0, 1] = tensors[:, 1, 0] = sxy
    tensors[:, 0, 2] = tensors[:, 2, 0] = sxz
    tensors[:, 1, 2] = tensors[:, 2, 1] = syz

    eigvals = np.linalg.eigvalsh(tensors)  # ascending per point: l3 <= l2 <= l1
    l3, l2, l1 = eigvals[:, 0], eigvals[:, 1], eigvals[:, 2]
    energy = l1 + l2 + l3
    denom = np.maximum(l1, 1e-12)  # avoids 0/0 in flat, edge-free regions

    return {
        # One dominant eigenvalue (gradient direction ~constant) -> a flat sheet/edge.
        "planarity": ((l1 - l2) / denom).astype(np.float32),
        # Two comparable dominant eigenvalues (gradient sweeps a whole transverse
        # plane, never along one axis) -> a tube/rod running along that axis.
        "linearity": ((l2 - l3) / denom).astype(np.float32),
        "sphericity": (l3 / denom).astype(np.float32),
        "energy": energy.astype(np.float32),
    }


def compute_field_variability(points, field_values, radius_mm, max_neighbors=64):
    """For each named field, measures how much its value varies among vertices
    within `radius_mm` (Euclidean, in `points`' own units) of each point - the
    standard deviation of the field over each point's local neighborhood
    (itself included, so a point with no neighbors in range gets a defined 0,
    not NaN).

    Distinct from any field's own magnitude: a uniformly bright but flat
    region has high "local_mean" but low "local_mean" variability, while a
    patchy/heterogeneous texture (e.g. placenta's cotyledon structure, or the
    edge of a hyperechoic acoustic shadow) scores high on both fronts.

    `points` must be in the mesh's own (un-aligned) physical units - run this
    before any alignment stage rescales the mesh, or `radius_mm` stops meaning
    an actual physical distance.

    Neighbor search is BOUNDED by `max_neighbors`: each vertex's neighborhood
    is its `max_neighbors` nearest vertices (via one chunked, vectorized
    `cKDTree.query`, same pattern as `compute_vertex_roughness`'s "knn" mode),
    restricted to those actually within `radius_mm` - not every vertex within
    `radius_mm`. This caps cost to O(N * max_neighbors) regardless of how
    densely the mesh is sampled. An unbounded radius query
    (`query_pairs`/`query_ball_point`) on a densely, lightly-decimated surface
    can turn a modest `radius_mm` into hundreds of millions of pairs - a 10 mm
    ball easily holds thousands of vertices at sub-mm spacing - which is
    exactly the memory blow-up (observed as a C++ `std::bad_alloc`, not a
    Python `MemoryError`) this cap avoids. In a region denser than
    `max_neighbors` within `radius_mm`, the effective sampling radius shrinks
    below `radius_mm` (only the closest `max_neighbors` are seen) - a bounded-
    cost approximation.

    Returns a dict of `[N]` float32 arrays, one per key in `field_values`.
    """
    points = np.asarray(points, dtype=np.float64)
    n_pts = points.shape[0]
    k = min(max_neighbors, n_pts - 1)

    if k <= 0:
        return {name: np.zeros(n_pts, dtype=np.float32) for name in field_values}

    tree = cKDTree(points)
    vi_parts, vj_parts = [], []
    chunk = 200_000  # bound the (chunk, k+1) index/distance buffers
    for start in range(0, n_pts, chunk):
        stop = min(start + chunk, n_pts)
        dist, idx = tree.query(points[start:stop], k=k + 1, workers=-1)  # col 0 = self
        dist, idx = dist[:, 1:], idx[:, 1:]
        rows, cols = np.nonzero(dist <= radius_mm)
        vi_parts.append(np.arange(start, stop, dtype=np.int64)[rows])
        vj_parts.append(idx[rows, cols])
    vi = np.concatenate(vi_parts)
    vj = np.concatenate(vj_parts)

    counts_with_self = np.bincount(vi, minlength=n_pts) + 1

    result = {}
    for name, values in field_values.items():
        values = np.asarray(values, dtype=np.float64)
        neighbor_sum = np.bincount(vi, weights=values[vj], minlength=n_pts) if vi.size else np.zeros(n_pts)
        neighbor_sum_sq = np.bincount(vi, weights=values[vj] ** 2, minlength=n_pts) if vi.size else np.zeros(n_pts)

        total_sum = neighbor_sum + values
        total_sum_sq = neighbor_sum_sq + values ** 2

        mean = total_sum / counts_with_self
        var = np.maximum(total_sum_sq / counts_with_self - mean ** 2, 0.0)
        result[name] = np.sqrt(var).astype(np.float32)

    return result
