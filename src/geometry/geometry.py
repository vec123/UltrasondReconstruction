import os
import os.path
import hashlib
from multiprocessing import Pool

import numpy as np
import scipy
import scipy.sparse.linalg as sla
import scipy.spatial
from sklearn.neighbors import KDTree

import potpourri3d as pp3d

import SimpleITK as sitk
import vtk
from vtk.util import numpy_support


# == File utilities

def hash_arrays(arrs):
    running_hash = hashlib.sha1()
    for arr in arrs:
        binarr = arr.view(np.uint8)
        running_hash.update(binarr)
    return running_hash.hexdigest()

def random_rotation_matrix(randgen=None):
    """
    Creates a random rotation matrix.
    randgen: if given, a np.random.RandomState instance used for random numbers (for reproducibility)
    """
    # adapted from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randgen is None:
        randgen = np.random.RandomState()

    theta, phi, z = tuple(randgen.rand(3).tolist())

    theta = theta * 2.0*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0 # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)


# == Mesh/point-cloud geometry (numpy)

def normalize_field(f_flat, out_range=(0.0, 1.0)):
    """Min-max normalize a *scalar* field to out_range, kept in float.

    NOTE: This is a per-array min-max scaler for scalar fields (e.g. curvature)
    used for visualization/storage. Do NOT use it on vector fields such as
    surface normals - global min-max scaling destroys their unit length. Unit
    vectors are already normalized at construction (see mesh_vertex_normals).
    """
    f = np.asarray(f_flat, dtype=np.float32)
    f_min, f_max = f.min(), f.max()
    lo, hi = out_range
    return (f - f_min) / (f_max - f_min + 1e-8) * (hi - lo) + lo

def norm(x, highdim=False):
    """
    Computes norm of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return np.linalg.norm(x, axis=-1)

def norm2(x, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return dot(x, x)

def normalize(x, divide_eps=1e-6, highdim=False):
    """
    Computes x / ||x||. Given (shape,d), returns (shape,d) normalized along the last dimension
    """
    if(len(x.shape) == 1):
        raise ValueError("called normalize() on single vector of dim " +
                         str(x.shape) + " are you sure?")
    if(not highdim and x.shape[-1] > 4):
        raise ValueError("called normalize() with large last dimension " +
                         str(x.shape) + " are you sure?")
    return x / (norm(x, highdim=highdim) + divide_eps)[..., np.newaxis]

def cross(vec_A, vec_B):
    return np.cross(vec_A, vec_B)

def dot(vec_A, vec_B):
    return np.sum(vec_A * vec_B, axis=-1)

def face_coords(verts, faces):
    coords = verts[faces]
    return coords

# Given (..., 3) vectors and normals, projects out any components of vecs
# which lies in the direction of normals. Normals are assumed to be unit.

def project_to_tangent(vecs, unit_normals):
    dots = dot(vecs, unit_normals)
    return vecs - unit_normals * dots[..., np.newaxis]

def face_area(verts, faces):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)
    return 0.5 * norm(raw_normal)

def face_normals(verts, faces, normalized=True):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)

    if normalized:
        return normalize(raw_normal)

    return raw_normal

def neighborhood_normal(points):
    # points: (N, K, 3) array of neighborhood psoitions
    # points should be centered at origin
    # out: (N,3) array of normals
    (u, s, vh) = np.linalg.svd(points, full_matrices=False)
    normal = vh[:,2,:]
    return normal / np.linalg.norm(normal,axis=-1, keepdims=True)

def mesh_vertex_normals(verts, faces):
    """Vertex normals, by summing (unit) incident face normals. numpy in / out."""
    face_n = face_normals(verts, faces)

    vertex_normals = np.zeros(verts.shape)
    for i in range(3):
        np.add.at(vertex_normals, faces[:,i], face_n)

    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals,axis=-1,keepdims=True)

    return vertex_normals


def vertex_normals(verts, faces=None, n_neighbors_cloud=30, normalize=True):
    """
    Vertex normals for a mesh, or (via a local neighborhood fit) a point cloud. numpy in / out.
    """
    if faces is None:
        faces = np.array([], dtype=np.int64)

    verts = np.asarray(verts)
    faces = np.asarray(faces)

    if faces.size == 0:  # point cloud
        _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
        neigh_points = verts[neigh_inds, :]
        neigh_points = neigh_points - verts[:, np.newaxis, :]
        normals = neighborhood_normal(neigh_points)

    else:  # mesh
        normals = mesh_vertex_normals(verts, faces)

        # if any are NaN, wiggle slightly and recompute
        bad_mask = np.isnan(normals).any(axis=1, keepdims=True)
        if bad_mask.any():
            bbox = np.amax(verts, axis=0) - np.amin(verts, axis=0)
            scale = np.linalg.norm(bbox) * 1e-4
            wiggle = (np.random.RandomState(seed=777).rand(*verts.shape) - 0.5) * scale
            wiggle_verts = verts + bad_mask * wiggle
            normals = mesh_vertex_normals(wiggle_verts, faces)

        # if still NaN assign random normals (probably means unreferenced verts in mesh)
        bad_mask_flat = np.isnan(normals).any(axis=1)
        if bad_mask_flat.any():
            normals[bad_mask_flat] = (np.random.RandomState(seed=777).rand(bad_mask_flat.sum(), 3) - 0.5)
            norms = np.linalg.norm(normals[bad_mask_flat], axis=-1, keepdims=True)
            normals[bad_mask_flat] /= (norms + 1e-8)

    if normalize:
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / (norms + 1e-8)

    if np.isnan(normals).any():
        raise ValueError("NaN normals detected in calculation :(")

    return normals

def build_tangent_frames(verts, faces, normals=None):

    V = verts.shape[0]

    if normals is None:
        vert_normals = vertex_normals(verts, faces)  # (V,3)
    else:
        vert_normals = normals

    # = find an orthogonal basis

    basis_cand1 = np.tile([1.0, 0.0, 0.0], (V, 1))
    basis_cand2 = np.tile([0.0, 1.0, 0.0], (V, 1))

    basisX = np.where((np.abs(dot(vert_normals, basis_cand1)) < 0.9)[..., np.newaxis],
                       basis_cand1, basis_cand2)
    basisX = project_to_tangent(basisX, vert_normals)
    basisX = normalize(basisX)
    basisY = cross(vert_normals, basisX)
    frames = np.stack((basisX, basisY, vert_normals), axis=-2)

    if np.any(np.isnan(frames)):
        raise ValueError("NaN coordinate frame! Must be very degenerate")

    return frames

def edge_tangent_vectors(verts, frames, edges):
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    basisX = frames[edges[0, :], 0, :]
    basisY = frames[edges[0, :], 1, :]

    compX = dot(edge_vecs, basisX)
    compY = dot(edge_vecs, basisY)
    edge_tangent = np.stack((compX, compY), axis=-1)

    return edge_tangent

def build_grad(verts, edges, edge_tangent_vectors):
    """
    Build a (V, V) complex sparse matrix grad operator. Given real inputs at vertices, produces a complex (vector value) at vertices giving the gradient. All values pointwise.
    - verts, edges, edge_tangent_vectors: numpy
    - edges: (2, E)
    """
    edges_np = edges

    # Build outgoing neighbor lists
    N = verts.shape[0]
    vert_edge_outgoing = [[] for i in range(N)]
    for iE in range(edges_np.shape[1]):
        tail_ind = edges_np[0, iE]
        tip_ind = edges_np[1, iE]
        if tip_ind != tail_ind:
            vert_edge_outgoing[tail_ind].append(iE)

    # Build local inversion matrix for each vertex
    row_inds = []
    col_inds = []
    data_vals = []
    eps_reg = 1e-5
    for iV in range(N):
        n_neigh = len(vert_edge_outgoing[iV])

        lhs_mat = np.zeros((n_neigh, 2))
        rhs_mat = np.zeros((n_neigh, n_neigh + 1))
        ind_lookup = [iV]
        for i_neigh in range(n_neigh):
            iE = vert_edge_outgoing[iV][i_neigh]
            jV = edges_np[1, iE]
            ind_lookup.append(jV)

            edge_vec = edge_tangent_vectors[iE][:]
            w_e = 1.

            lhs_mat[i_neigh][:] = w_e * edge_vec
            rhs_mat[i_neigh][0] = w_e * (-1)
            rhs_mat[i_neigh][i_neigh + 1] = w_e * 1

        lhs_T = lhs_mat.T
        lhs_inv = np.linalg.inv(lhs_T @ lhs_mat + eps_reg * np.identity(2)) @ lhs_T

        sol_mat = lhs_inv @ rhs_mat
        sol_coefs = (sol_mat[0, :] + 1j * sol_mat[1, :]).T

        for i_neigh in range(n_neigh + 1):
            i_glob = ind_lookup[i_neigh]

            row_inds.append(iV)
            col_inds.append(i_glob)
            data_vals.append(sol_coefs[i_neigh])

    # build the sparse matrix
    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    data_vals = np.array(data_vals)
    mat = scipy.sparse.coo_matrix(
        (data_vals, (row_inds, col_inds)), shape=(
            N, N)).tocsc()

    return mat

def build_grad_point_cloud(verts, frames, n_neighbors_cloud=30):

    verts_np = verts

    _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')

    edge_inds_from = np.repeat(np.arange(verts.shape[0]), n_neighbors_cloud)
    edges = np.stack((edge_inds_from, neigh_inds.flatten()))
    edge_tangent_vecs = edge_tangent_vectors(verts, frames, edges)

    return build_grad(verts_np, edges, edge_tangent_vecs)

def compute_grads(verts, faces=None, L=None, normals=None):

    is_cloud = (faces is None)
    frames = build_tangent_frames(verts, faces, normals=normals)

    if is_cloud:
        grad_mat_np = build_grad_point_cloud(verts, frames)
    else:
        L_coo = L.tocoo()
        inds_row = L_coo.row
        inds_col = L_coo.col

        edges = np.stack((inds_row, inds_col), axis=0)
        edge_vecs = edge_tangent_vectors(verts, frames, edges)
        grad_mat_np = build_grad(verts, edges, edge_vecs)

    gradX_np = np.real(grad_mat_np)
    gradY_np = np.imag(grad_mat_np)

    return gradX_np, gradY_np


# Finds the k nearest (or farthest, if largest=True) neighbors of source on target.
# Return is (dists, indices), both [N, k], sorted by increasing distance (decreasing if largest=True).
def find_knn(points_source, points_target, k, largest=False, omit_diagonal=False, method='brute'):

    points_source = np.asarray(points_source)
    points_target = np.asarray(points_target)

    if omit_diagonal and points_source.shape[0] != points_target.shape[0]:
        raise ValueError("omit_diagonal can only be used when source and target are same shape")

    if method != 'cpu_kd' and points_source.shape[0] * points_target.shape[0] > 1e8:
        method = 'cpu_kd'
        print("switching to cpu_kd knn")

    if method == 'brute':

        diff_mat = points_source[:, np.newaxis, :] - points_target[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff_mat, axis=-1)

        if omit_diagonal:
            np.fill_diagonal(dist_mat, np.inf)

        # NumPy has no direct topk for all rows; use argpartition then sort the k selected.
        if largest:
            indices = np.argpartition(-dist_mat, k, axis=1)[:, :k]
        else:
            indices = np.argpartition(dist_mat, k, axis=1)[:, :k]

        rows = np.arange(dist_mat.shape[0])[:, np.newaxis]
        dists = dist_mat[rows, indices]

        order = np.argsort(dists, axis=1)
        if largest:
            order = order[:, ::-1]
        dists = dists[rows, order]
        indices = indices[rows, order]

        return dists, indices

    elif method == 'cpu_kd':

        if largest:
            raise ValueError("can't do largest with cpu_kd")

        tree = KDTree(points_target)

        k_search = k+1 if omit_diagonal else k
        _, neighbors = tree.query(points_source, k=k_search)

        if omit_diagonal:
            # Mask out self element
            mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

            # make sure we mask out exactly one element in each row, in rare case of many duplicate points
            mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False

            neighbors = neighbors[mask].reshape((neighbors.shape[0], neighbors.shape[1]-1))

        dists = np.linalg.norm(points_source[:, np.newaxis, :] - points_target[neighbors], axis=-1)

        return dists, neighbors

    else:
        raise ValueError("unrecognized method")


def normalize_positions(pos, faces=None, method='mean', scale_method='max_rad'):
    # center and unit-scale positions

    if method == 'mean':
        # center using the average point position
        pos = pos - np.mean(pos, axis=-2, keepdims=True)
    elif method == 'bbox':
        # center via the middle of the axis-aligned bounding box
        bbox_min = np.min(pos, axis=-2)
        bbox_max = np.max(pos, axis=-2)
        center = (bbox_max + bbox_min) / 2.
        pos = pos - np.expand_dims(center, axis=-2)
    else:
        raise ValueError("unrecognized method")

    if scale_method == 'max_rad':
        scale = np.max(norm(pos), axis=-1, keepdims=True)[..., np.newaxis]
        pos = pos / scale
    elif scale_method == 'area':
        if faces is None:
            raise ValueError("must pass faces for area normalization")
        coords = pos[faces]
        vec_A = coords[:, 1, :] - coords[:, 0, :]
        vec_B = coords[:, 2, :] - coords[:, 0, :]
        face_areas = np.linalg.norm(np.cross(vec_A, vec_B), axis=1) * 0.5
        total_area = np.sum(face_areas)
        scale = (1. / np.sqrt(total_area))
        pos = pos * scale
    else:
        raise ValueError("unrecognized scale method")
    return pos


def farthest_point_sampling(points, n_sample):
    # Returns a |V| boolean mask with n_sample elements set to true.

    N = points.shape[0]
    if(n_sample > N): raise ValueError("not enough points to sample")

    chosen_mask = np.zeros(N, dtype=bool)
    min_dists = np.full(N, np.inf, dtype=points.dtype)

    # pick the centermost first point
    points = normalize_positions(points)
    i = int(np.argmin(norm2(points), axis=0))
    chosen_mask[i] = True

    for _ in range(n_sample-1):

        # update distance
        dists = norm2(points[i, :][np.newaxis, :] - points)
        min_dists = np.minimum(dists, min_dists)

        # take the farthest
        i = int(np.argmax(min_dists, axis=0))
        chosen_mask[i] = True

    return chosen_mask


def compute_geodesic_distance(verts, faces, source_idx=0):
    """Geodesic distance from a source vertex (heat method). numpy in / out."""
    solver = pp3d.MeshHeatMethodDistanceSolver(np.asarray(verts), np.asarray(faces))
    return solver.compute_distance(source_idx)

def compute_mean_curvature(verts, faces=None):
    """
    Computes mean curvature (H) for meshes or point clouds.
    If faces is None, uses a distance-weighted point cloud Laplacian.
    """
    # Ensure inputs are numpy arrays
    v = verts.cpu().numpy() if hasattr(verts, 'cpu') else verts

    if faces is not None:
        # --- Mesh Case: Cotan Laplacian ---
        f = faces.cpu().numpy() if hasattr(faces, 'cpu') else faces
        L = pp3d.cotan_laplacian(v, f)
        # For meshes, the vertex mass is the dual area
        mass = pp3d.vertex_areas(v, f)
        inv_mass = 1.0 / (mass + 1e-12)
        lap_pos = L @ v
        # Curvature is proportional to the norm of the laplacian of the position
        H = np.linalg.norm(lap_pos, axis=1) / 2.0

    else:
        # --- Point Cloud Case: Gaussian-weighted Laplacian ---
        # 1. Build a spatial index to find neighbors
        tree = KDTree(v)
        # Get 20 nearest neighbors to approximate local connectivity
        dist, idx = tree.query(v, k=20)

        # 2. Compute Gaussian weights (sigma based on average distance)
        sigma = np.mean(dist)
        weights = np.exp(-(dist**2) / (2 * sigma**2))

        # 3. Construct sparse Laplacian
        n = v.shape[0]
        rows = np.repeat(np.arange(n), 20)
        cols = idx.flatten()
        data = weights.flatten()

        # Simple graph Laplacian: L = D - W
        W = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        W = (W + W.T) / 2  # Make symmetric
        D = scipy.sparse.diags(np.array(W.sum(axis=1)).flatten())
        L = D - W

        lap_pos = L @ v
        H = np.linalg.norm(lap_pos, axis=1) / 2.0

    return H

def compute_harmonic_potential(L, boundary_indices, boundary_values):
    """
    Solves Lu = 0 with fixed Dirichlet boundary conditions.
    L: (V, V) sparse Laplacian
    boundary_indices: list of vertex indices
    boundary_values: list of values for those indices
    """
    n = L.shape[0]
    # Create a system where we pin the boundary values
    # We modify L to set identity at boundary rows
    L_mod = L.tolil()
    rhs = np.zeros(n)

    for idx, val in zip(boundary_indices, boundary_values):
        L_mod[idx, :] = 0
        L_mod[idx, idx] = 1
        rhs[idx] = val

    return sla.spsolve(L_mod.tocsr(), rhs)

def compute_hks(evals, evecs, scales):
    """
    Inputs:
      - evals: (K,) eigenvalues
      - evecs: (V,K) values
      - scales: (S,) times
    Outputs:
      - (V,S) hks values
    """
    evals = np.asarray(evals)
    evecs = np.asarray(evecs)
    scales = np.asarray(scales)

    # expand batch
    if evals.ndim == 1:
        expand_batch = True
        evals = evals[np.newaxis, :]
        evecs = evecs[np.newaxis, :, :]
        scales = scales[np.newaxis, :]
    else:
        expand_batch = False

    power_coefs = np.exp(-evals[:, np.newaxis, :] * scales[:, :, np.newaxis])[:, np.newaxis, :, :]  # (B,1,S,K)
    terms = power_coefs * (evecs * evecs)[:, :, np.newaxis, :]  # (B,V,S,K)

    out = np.sum(terms, axis=-1)  # (B,V,S)

    return out[0] if expand_batch else out

def compute_hks_autoscale(evals, evecs, count):
    # these scales roughly approximate those suggested in the hks paper
    scales = np.logspace(-2, 0., num=count)
    return compute_hks(evals, evecs, scales)

def geodesic_label_errors(target_verts, target_faces, pred_labels, gt_labels, normalization='diameter', geodesic_cache_dir=None):
    """
    Return a vector of distances between predicted and ground-truth lables (normalized by geodesic diameter or area)

    This method is SLOW when it needs to recompute geodesic distances.
    """

    target_verts = np.asarray(target_verts)
    target_faces = np.asarray(target_faces)

    pred_labels = np.asarray(pred_labels)
    gt_labels = np.asarray(gt_labels)

    dists = get_all_pairs_geodesic_distance(target_verts, target_faces, geodesic_cache_dir)

    result_dists = dists[pred_labels, gt_labels]

    if normalization == 'diameter':
        geodesic_diameter = np.max(dists)
        normalized_result_dists = result_dists / geodesic_diameter
    elif normalization == 'area':
        total_area = np.sum(face_area(target_verts, target_faces))
        normalized_result_dists = result_dists / np.sqrt(total_area)
    else:
        raise ValueError('unrecognized normalization')

    return normalized_result_dists

# This function and the helper class below are to support parallel computation of all-pairs geodesic distance
def all_pairs_geodesic_worker(verts, faces, i):
    import igl

    N = verts.shape[0]

    sources = np.array([i])[:,np.newaxis]
    targets = np.arange(N)[:,np.newaxis]
    dist_vec = igl.exact_geodesic(verts, faces, sources, targets)

    return dist_vec

class AllPairsGeodesicEngine(object):
    def __init__(self, verts, faces):
        self.verts = verts
        self.faces = faces
    def __call__(self, i):
        return all_pairs_geodesic_worker(self.verts, self.faces, i)


def get_all_pairs_geodesic_distance(verts_np, faces_np, geodesic_cache_dir=None):
    """
    Return a gigantic VxV dense matrix containing the all-pairs geodesic distance matrix. Internally caches, recomputing only if necessary.

    (numpy in, numpy out)
    """

    # need libigl for geodesic call
    try:
        import igl
    except ImportError as e:
        raise ImportError("Must have python libigl installed for all-pairs geodesics. `conda install -c conda-forge igl`")

    # Check the cache
    found = False
    if geodesic_cache_dir is not None:
        ensure_dir_exists(geodesic_cache_dir)
        hash_key_str = str(hash_arrays((verts_np, faces_np)))

        i_cache_search = 0
        while True:

            search_path = os.path.join(
                geodesic_cache_dir,
                hash_key_str + "_" + str(i_cache_search) + ".npz")

            try:
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile["verts"]
                cache_faces = npzfile["faces"]

                if (not np.array_equal(verts_np, cache_verts)) or (not np.array_equal(faces_np, cache_faces)):
                    i_cache_search += 1
                    continue

                found = True
                result_dists = npzfile["dist"]
                break

            except FileNotFoundError:
                break

    if not found:

        print("Computing all-pairs geodesic distance (warning: SLOW!)")

        N = verts_np.shape[0]

        try:
            pool = Pool(None) # on 8 processors
            engine = AllPairsGeodesicEngine(verts_np, faces_np)
            outputs = pool.map(engine, range(N))
        finally: # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()

        result_dists = np.array(outputs)

        result_dists = np.nan_to_num(result_dists, nan=np.nan, posinf=np.nan, neginf=np.nan)

        result_dists = np.fmin(result_dists, np.transpose(result_dists))

        max_dist = np.nanmax(result_dists)
        result_dists = np.nan_to_num(result_dists, nan=max_dist, posinf=max_dist, neginf=max_dist)

        print("...finished computing all-pairs geodesic distance")

        if geodesic_cache_dir is not None:

            print("saving geodesic distances to cache: " + str(geodesic_cache_dir))

            np.savez(search_path,
                     verts=verts_np,
                     faces=faces_np,
                     dist=result_dists
                     )

    return result_dists


# == Point-set registration (Umeyama / Procrustes)

def umeyama_similarity_transform(src_pts, dst_pts, with_scale=True):
    """
    Computes the optimal similarity transformation (s * R * x + t) mapping
    src_pts onto dst_pts using Umeyama's algorithm / Procrustes analysis.

    Parameters
    ----------
    src_pts, dst_pts : (N, 3) arrays of corresponding points.
    with_scale : if False, the transform is rigid (s == 1.0).

    Returns
    -------
    M : (4, 4) homogeneous transformation matrix.
    """
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)

    src_mean = np.mean(src_pts, axis=0)
    dst_mean = np.mean(dst_pts, axis=0)

    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean

    n = src_pts.shape[0]

    # Covariance matrix
    H = (src_centered.T @ dst_centered) / n
    U, S, Vt = np.linalg.svd(H)

    # Rotation
    R = Vt.T @ U.T

    # Reflection correction
    d = np.ones(3)
    if np.linalg.det(R) < 0:
        d[-1] = -1
        R = Vt.T @ np.diag(d) @ U.T

    # Scale calculation
    if with_scale:
        src_var = np.sum(src_centered**2) / n
        s = np.sum(S * d) / src_var
    else:
        s = 1.0

    # Translation
    t = dst_mean - s * (R @ src_mean)

    # Build 4x4 matrix
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t

    return M


def decompose_similarity_transform(M):
    """
    Splits a 4x4 similarity matrix into its scale, rotation and translation
    parts, such that the transform is `s * R @ x + t`.
    """
    A = np.asarray(M)[:3, :3]

    # Columns of s * R all have length s, since R is orthonormal.
    s = float(np.mean(np.linalg.norm(A, axis=0)))
    R = A / s
    t = np.asarray(M)[:3, 3].copy()

    return s, R, t


# == NRRD / volume preprocessing (SimpleITK + VTK)

def preprocess_volume(sitk_image, denoise="median", radius=1,
                      normalize=True, clip_percentiles=(1.0, 99.0), out_max=255.0):
    """
    Speckle-reduce and (optionally) intensity-normalize an ultrasound volume
    before level-set extraction.

    Denoising and normalization are independent:
      - denoise ("median"/"curvature"/None): edge-preserving speckle reduction.
      - normalize (bool): percentile-clip and rescale intensities to [0, out_max]
        so a contour value becomes a *relative* level comparable across scans.
        Set normalize=False to keep the original intensity scale (so an absolute
        contour value keeps its old meaning) while still denoising.

    Returns a float32 numpy array in (z, y, x) order (same layout as
    sitk.GetArrayFromImage), so the caller can build vtkImageData unchanged.
    """
    # 1. Edge-preserving speckle reduction in image space (spacing-aware).
    if denoise == "median":
        r = int(radius)
        sitk_image = sitk.Median(sitk_image, [r, r, r])
    elif denoise == "curvature":
        # Anisotropic (curvature) diffusion: smooths speckle while preserving edges.
        sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
        cflow = sitk.CurvatureFlowImageFilter()
        cflow.SetTimeStep(0.0625)
        cflow.SetNumberOfIterations(5)
        sitk_image = cflow.Execute(sitk_image)
    elif denoise not in (None, "none"):
        raise ValueError(f"Unknown denoise mode: {denoise!r}")

    arr = sitk.GetArrayFromImage(sitk_image).astype(np.float32)

    # 2. Optional percentile clip (robust to bright specular outliers) + rescale.
    if normalize:
        lo, hi = np.percentile(arr, clip_percentiles)
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo + 1e-8) * out_max
    return arr


def crop_volume(sitk_image, threshold=0, margin=0, verbose=True):
    """
    Crop an ultrasound volume to the bounding box of its content, then trim an
    extra border off every side.

    This is the 3D analog of cutting the black borders off a 2D scan: it removes
    the empty padding around the acquired cone so downstream processing focuses
    on real anatomy.

      - threshold: intensity level defining "content" for the bounding box; the
        box is the tightest cube containing all voxels > threshold. Applied on
        the *raw* intensities (crop runs before denoise/normalize). Use None to
        skip bounding-box detection and only apply `margin`.
      - margin: how much extra border to cut off *every* side (the "how much"
        hyperparameter). An int is voxels per side; a float in (0, 1) is a
        fraction of each axis length. 0 = crop to the content box only.

    Cropping is done via SimpleITK slicing, which updates the image origin so the
    remaining voxels keep their correct physical coordinates. Returns a (possibly
    smaller) sitk.Image.
    """
    arr = sitk.GetArrayFromImage(sitk_image)          # (z, y, x)
    shape = np.array(arr.shape)

    # 1. Base bounding box from content above threshold (else the full volume).
    if threshold is not None:
        mask = arr > threshold
        if not mask.any():
            if verbose:
                print(f"crop_volume: no voxel > {threshold}; skipping crop.")
            return sitk_image
        coords = np.argwhere(mask)
        lo = coords.min(axis=0)            # (z0, y0, x0)
        hi = coords.max(axis=0) + 1        # exclusive upper bound
    else:
        lo = np.zeros(3, dtype=int)
        hi = shape.copy()

    # 2. Trim an extra margin off every side (int voxels, or float fraction).
    axis_len = hi - lo
    if isinstance(margin, float) and 0.0 < margin < 1.0:
        m = np.round(axis_len * margin).astype(int)
    else:
        m = np.full(3, int(margin), dtype=int)
    lo = lo + m
    hi = hi - m

    # 3. Guard against over-cropping, keep at least 1 voxel per axis.
    lo = np.clip(lo, 0, shape - 1)
    hi = np.clip(hi, lo + 1, shape)

    z0, y0, x0 = (int(v) for v in lo)
    z1, y1, x1 = (int(v) for v in hi)
    cropped = sitk_image[x0:x1, y0:y1, z0:z1]          # sitk is (x, y, z); updates origin

    if verbose:
        print(f"crop_volume: {tuple(int(s) for s in shape[::-1])} (x,y,z) -> "
              f"{tuple(cropped.GetSize())}")
    return cropped


def otsu_threshold(array, nbins=256):
    """
    Computes Otsu's threshold: the intensity level that maximizes between-
    class variance, splitting `array` into a low- and a high-intensity class.

    This is the standard parameter-free way to pick a foreground/background
    cutoff from a volume's own histogram, so the same threshold *setting*
    (auto_contour=True below) adapts per scan instead of requiring a hand-
    tuned absolute contour_value that only holds for one scanner/source.
    """
    finite = np.asarray(array, dtype=np.float64)
    finite = finite[np.isfinite(finite)]

    hist, bin_edges = np.histogram(finite, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    weight1 = np.cumsum(hist).astype(np.float64)
    weight2 = weight1[-1] - weight1

    sum1 = np.cumsum(hist * bin_centers)
    mean1 = np.divide(sum1, weight1, out=np.zeros_like(sum1), where=weight1 > 0)
    sum2 = sum1[-1] - sum1
    mean2 = np.divide(sum2, weight2, out=np.zeros_like(sum2), where=weight2 > 0)

    # Between-class variance at every candidate split; skip the last bin edge
    # (no "class 2" left to split against).
    variance12 = weight1[:-1] * weight2[:-1] * (mean1[:-1] - mean2[:-1]) ** 2
    best_idx = int(np.argmax(variance12))
    return float(bin_centers[best_idx])


def load_and_preprocess_volume(nrrd_path, preprocess=True, denoise="median", denoise_radius=1,
                               normalize=True, clip_percentiles=(1.0, 99.0), out_max=255.0,
                               crop=False, crop_threshold=0, crop_margin=0):
    """
    Loads a NRRD file and applies the same crop -> denoise/normalize pipeline
    `convert_single_nrrd` uses before contouring, returning the array together
    with the spacing/origin of the frame its voxel indices live in.

    Factored out so any other code that needs to map a WORLD-space point (e.g.
    a `convert_single_nrrd`-derived surface vertex) back to a voxel index -
    such as sampling texture around a mesh's own surface points - uses the
    exact same frame the surface was extracted in, rather than an
    independently-recomputed (and possibly drifted) crop/spacing/origin.

    Returns (np_array [z, y, x], spacing (sx, sy, sz), origin (ox, oy, oz)).
    """
    sitk_image = sitk.ReadImage(nrrd_path)

    # Crop first so spacing/origin below reflect the cropped volume.
    if crop:
        sitk_image = crop_volume(sitk_image, threshold=crop_threshold,
                                 margin=crop_margin)

    spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()

    if preprocess:
        np_array = preprocess_volume(sitk_image, denoise=denoise,
                                     radius=denoise_radius,
                                     normalize=normalize,
                                     clip_percentiles=clip_percentiles,
                                     out_max=out_max)
    else:
        np_array = sitk.GetArrayFromImage(sitk_image)

    return np_array, spacing, origin


def convert_single_nrrd(nrrd_path, contour_value=None, auto_contour=False, preprocess=True,
                        denoise="median", denoise_radius=1,
                        normalize=True, clip_percentiles=(1.0, 99.0), out_max=255.0,
                        crop=False, crop_threshold=0, crop_margin=0):
    """
    Converts a single NRRD file to a VTP file.

    When crop=True the volume is first cropped to its content bounding box (plus
    crop_margin trimmed off each side). When preprocess=True the volume is then
    denoised (and, if normalize=True, intensity-normalized to [0, out_max])
    before contouring. With normalize=True contour_value is a level relative to
    out_max; with normalize=False it stays on the original intensity scale.

    When auto_contour=True, contour_value is IGNORED and instead computed per
    volume via otsu_threshold() on the (preprocessed) intensity histogram -
    use this to get a consistent surface across NRRD sources with different
    intensity scales/calibrations, without hand-tuning contour_value per source.
    """
    np_array, spacing, origin = load_and_preprocess_volume(
        nrrd_path, preprocess=preprocess, denoise=denoise, denoise_radius=denoise_radius,
        normalize=normalize, clip_percentiles=clip_percentiles, out_max=out_max,
        crop=crop, crop_threshold=crop_threshold, crop_margin=crop_margin,
    )

    if auto_contour:
        contour_value = otsu_threshold(np_array)
        print(f"convert_single_nrrd: auto contour_value (Otsu) = {contour_value:.4g}")

    # 2. Convert to vtkImageData
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(np_array.shape[2], np_array.shape[1], np_array.shape[0])
    vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])
    vtk_image.SetOrigin(origin[0], origin[1], origin[2])

    vtk_data_array = numpy_support.numpy_to_vtk(
        num_array=np_array.ravel(),
        deep=True,
        array_type=numpy_support.get_vtk_array_type(np_array.dtype)
    )
    vtk_data_array.SetName("Scalars")
    vtk_image.GetPointData().SetScalars(vtk_data_array)

    # 3. Convert vtkImageData to vtkPolyData (Geometry generation)
    if contour_value is not None:
        contour_filter = vtk.vtkMarchingCubes()
        contour_filter.SetInputData(vtk_image)
        contour_filter.SetValue(0, contour_value)
        contour_filter.Update()
        poly_data = contour_filter.GetOutput()
    else:
        geometry_filter = vtk.vtkGeometryFilter()
        geometry_filter.SetInputData(vtk_image)
        geometry_filter.Update()
        poly_data = geometry_filter.GetOutput()


    return poly_data


# == VTK polydata helpers

def load_vtp_mesh(vtp_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    return reader.GetOutput()

def get_vtp_vertices(polydata):

    points = polydata.GetPoints()

    # Convert vtkPoints to a numpy array
    # vtk_array_to_numpy provides a direct view or copy of the underlying data
    vertices = numpy_support.vtk_to_numpy(points.GetData())

    return vertices

def get_vtp_faces(polydata):
    """Triangle connectivity ``[F, 3]`` from a vtkPolyData. Does not triangulate
    non-triangular polys first - see ``extract_arrays_from_polydata`` /
    ``src.vtk.extract.extract_vtp_points_cells`` for that."""
    connectivity = numpy_support.vtk_to_numpy(polydata.GetPolys().GetConnectivityArray())
    return connectivity.reshape(-1, 3)

def load_vtu(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()  # This executes the reader

    # Return the actual data object, not the port
    return reader.GetOutput()
def convert_vtu_to_vtp_vtk(vtu_data):
    """
    Converts a VTU object to VTP, with internal checks to ensure
    valid input and non-empty output.
    """
    # 1. Check if the input is valid
    if vtu_data is None:
        raise ValueError("Input vtu_data is None.")

    print(f"Input object type: {vtu_data.GetClassName()}")
    print(f"Input points: {vtu_data.GetNumberOfPoints()}")

    if vtu_data.GetNumberOfPoints() == 0:
        raise ValueError("Input VTU has 0 points. Conversion aborted.")

    # 2. Setup the filter
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(vtu_data)
    surface_filter.Update()

    # 3. Retrieve the output
    vtp_output = surface_filter.GetOutput()

    # 4. Check if the filter actually produced something
    if vtp_output is None or vtp_output.GetNumberOfPoints() == 0:
        raise RuntimeError("Surface extraction failed: The resulting VTP is empty.")

    print(f"Conversion successful. Output points: {vtp_output.GetNumberOfPoints()}")

    return vtp_output


def extract_arrays_from_polydata(poly_data):
    points = numpy_support.vtk_to_numpy(poly_data.GetPoints().GetData())
    cells = numpy_support.vtk_to_numpy(poly_data.GetPolys().GetConnectivityArray())
    return points, cells.reshape(-1, 3)


def filter_largest_component(poly_data):
    """Loads a VTP, extracts only the largest connected component, and saves it."""

    print(f"Filter largest component for mesh with {poly_data.GetNumberOfPoints()} points.")

    # 2. Extract the largest connected component
    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(poly_data)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()

    largest_mesh = connectivity.GetOutput()

    print(f"largest component has {largest_mesh.GetNumberOfPoints()} points.")

    return largest_mesh

def clean_mesh(polydata):
    """
    Isolates the single largest connected component, removes topological defects,
    and decimates the triangle density by the specified reduction factor.
    """

    # 1. Initialize VTP Reader
    raw_mesh = polydata

    orig_verts = raw_mesh.GetNumberOfPoints()
    orig_cells = raw_mesh.GetNumberOfCells()
    print(f"Original Count : {orig_verts:,} vertices | {orig_cells:,} triangles")

    # 2. Isolate the absolute largest component (strips away tiny floating noise)
    print("Applying Connectiviy filter...")
    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(raw_mesh)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()

    # 3. Clean up topology (merges coincident points, eliminates zero-area triangles)
    print("Cleaning mesh topology and removing degenerate geometry...")
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(connectivity.GetOutput())
    cleaner.Update()
    out = cleaner.GetOutput()
    return out

def reduce_mesh(polydata, reduction_factor=0.90):
    # 4. Decimate the mesh using Quadric Error Metrics
    # Keeps structural landmarks (edges, curves) while aggressively thinning flat zones.

    raw_mesh = polydata
    orig_verts = raw_mesh.GetNumberOfPoints()
    orig_cells = raw_mesh.GetNumberOfCells()
    print(f"original Count: {orig_verts:,} vertices | {orig_cells:,} triangles")

    print(f"Decimating triangles by {reduction_factor * 100:.1f}%...")
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(raw_mesh)
    decimate.SetTargetReduction(reduction_factor)
    decimate.Update()

    reduced_mesh = decimate.GetOutput()

    final_verts = reduced_mesh.GetNumberOfPoints()
    final_cells = reduced_mesh.GetNumberOfCells()

    print("-" * 60)
    print(f"Optimized Count: {final_verts:,} vertices | {final_cells:,} triangles")
    print(f"Data Reduction : {((orig_verts - final_verts) / orig_verts) * 100:.2f}% vertices removed.")
    print("-" * 60)

    return reduced_mesh

def create_polydata(points, faces=None):
    """Creates a VTK PolyData object from points and optional faces."""
    polydata = vtk.vtkPolyData()

    # Set Points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(points.astype(np.float32)))
    polydata.SetPoints(vtk_points)

    # Set Faces
    if faces is not None:
        # Create a vtkIdTypeArray for cell connectivity
        # Format: [3, i1, i2, i3, 3, j1, j2, j3, ...]
        n_faces = faces.shape[0]
        # Prepend '3' (number of vertices per triangle) to every row
        cells_flat = np.hstack([np.ones((n_faces, 1), dtype=np.int64) * 3, faces.astype(np.int64)])

        # Convert to vtkIdTypeArray
        cell_array = vtk.vtkCellArray()
        # Newer VTK versions prefer SetCells with an offset array,
        # but this conversion is the standard way to feed legacy-style data:
        cell_array.SetCells(n_faces, numpy_support.numpy_to_vtk(cells_flat.flatten(), deep=True, array_type=vtk.VTK_ID_TYPE))

        polydata.SetPolys(cell_array)
    else:
        # Fallback to points/vertices
        vertices = vtk.vtkCellArray()
        for i in range(len(points)):
            vertices.InsertNextCell(1, [i])
        polydata.SetVerts(vertices)

    return polydata

def create_polydata_w_lines(points, lines):
    """
    points: (N, 3) array
    lines: (M, 2) array of index pairs [start_idx, end_idx]
    """
    polydata = vtk.vtkPolyData()

    # 1. Set Points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(points.astype(np.float32)))
    polydata.SetPoints(vtk_points)

    # 2. Set Lines
    # VTK expects lines in the format [2, idx1, idx2, 2, idx3, idx4, ...]
    n_lines = lines.shape[0]
    lines_flat = np.hstack([np.ones((n_lines, 1), dtype=np.int64) * 2, lines.astype(np.int64)])

    cell_array = vtk.vtkCellArray()
    cell_array.SetCells(n_lines, numpy_support.numpy_to_vtk(lines_flat.flatten(), deep=True, array_type=vtk.VTK_ID_TYPE))

    polydata.SetLines(cell_array)
    return polydata

def add_point_field(polydata, field_data, field_name="field"):
    """Adds a scalar or vector field to the existing PolyData."""
    vtk_array = numpy_support.numpy_to_vtk(field_data, deep=True)
    vtk_array.SetName(field_name)
    polydata.GetPointData().AddArray(vtk_array)
    return polydata

def save_vtp_mesh(polydata, vtk_path, binary=True):
    """Saves the PolyData object to disk using the XML format (VTP)."""
    print("Saving to:", vtk_path)
    os.makedirs(os.path.dirname(vtk_path), exist_ok=True)
    # Use vtkXMLPolyDataWriter instead of vtkPolyDataWriter
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(vtk_path)
    writer.SetInputData(polydata)

    if binary:
        # For XML files, 'binary' is the default and is handled automatically.
        # You can specify the data mode if needed:
        writer.SetDataModeToAppended()
        writer.SetCompressorTypeToZLib() # Optional: compresses the file
    else:
        writer.SetDataModeToAscii()

    return writer.Write() == 1

def save_vtk(polydata, vtk_path, binary=True):
    """Saves the PolyData object to disk."""
    print("saving to ", vtk_path)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(vtk_path)
    writer.SetInputData(polydata)
    if binary:
        writer.SetFileTypeToBinary()
    else:
        writer.SetFileTypeToASCII()
    return writer.Write() == 1
