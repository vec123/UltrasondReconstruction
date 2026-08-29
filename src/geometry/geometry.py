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



# == VTK polydata helpers

def load_vtp_mesh(vtp_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    return reader.GetOutput()


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

