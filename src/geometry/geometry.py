import os
import os.path
import random
import hashlib
from multiprocessing import Pool

import numpy as np
# we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)
import scipy
import scipy.sparse.linalg as sla
import scipy.spatial
import torch
from torch.distributions.categorical import Categorical
import sklearn.neighbors

import robust_laplacian
import potpourri3d as pp3d

import SimpleITK as sitk
import vtk
from vtk.util import numpy_support


# == Pytorch/numpy utilities

def toNP(x):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    return x.detach().to(torch.device('cpu')).numpy()

def label_smoothing_log_loss(pred, labels, smoothing=0.0):
    n_class = pred.shape[-1]
    one_hot = torch.zeros_like(pred)
    one_hot[labels] = 1.
    one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
    loss = -(one_hot * pred).sum(dim=-1).mean()
    return loss


# Randomly rotate points.
# Torch in, torch out
# Note fornow, builds rotation matrix on CPU.
def random_rotate_points(pts, randgen=None):
    R = random_rotation_matrix(randgen)
    R = torch.from_numpy(R).to(device=pts.device, dtype=pts.dtype)
    return torch.matmul(pts, R)

def random_rotate_points_y(pts):
    angles = torch.rand(1, device=pts.device, dtype=pts.dtype) * (2. * np.pi)
    rot_mats = torch.zeros(3, 3, device=pts.device, dtype=pts.dtype)
    rot_mats[0,0] = torch.cos(angles)
    rot_mats[0,2] = torch.sin(angles)
    rot_mats[2,0] = -torch.sin(angles)
    rot_mats[2,2] = torch.cos(angles)
    rot_mats[1,1] = 1.

    pts = torch.matmul(pts, rot_mats)
    return pts

# Numpy sparse matrix to pytorch
def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()

# Pytorch sparse to numpy csc matrix
def sparse_torch_to_np(A):
    if len(A.shape) != 2:
        raise RuntimeError("should be a matrix-shaped type; dim is : " + str(A.shape))

    indices = toNP(A.indices())
    values = toNP(A.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsc()

    return mat


# Hash a list of numpy arrays
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

# Python string/file utilities
def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)


# == Mesh/point-cloud geometry (torch)

def normalize_field(f_flat, out_range=(0.0, 1.0)):
    """Min-max normalize a *scalar* field to out_range, kept in float.

    NOTE: This is a per-array min-max scaler for scalar fields (e.g. curvature)
    used for visualization/storage. Do NOT use it on vector fields such as
    surface normals - global min-max scaling destroys their unit length. Unit
    vectors are already normalized at construction (see mesh_vertex_normals).

    The previous implementation cast the result to int32, quantizing the field
    to ~100 integer levels and discarding all sub-integer precision; it now
    stays in float.
    """
    f = np.asarray(f_flat, dtype=np.float32)
    f_min, f_max = f.min(), f.max()
    lo, hi = out_range
    return (f - f_min) / (f_max - f_min + 1e-8) * (hi - lo) + lo

def MSE_loss(f,f_rec):
     loss = torch.mean((f - f_rec)**2)
     return loss

def norm(x, highdim=False):
    """
    Computes norm of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return torch.norm(x, dim=len(x.shape) - 1)

def norm2(x, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return dot(x, x)

def normalize(x, divide_eps=1e-6, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    if(len(x.shape) == 1):
        raise ValueError("called normalize() on single vector of dim " +
                         str(x.shape) + " are you sure?")
    if(not highdim and x.shape[-1] > 4):
        raise ValueError("called normalize() with large last dimension " +
                         str(x.shape) + " are you sure?")
    return x / (norm(x, highdim=highdim) + divide_eps).unsqueeze(-1)

def face_coords(verts, faces):
    coords = verts[faces]
    return coords

def cross(vec_A, vec_B):
    return torch.cross(vec_A, vec_B, dim=-1)

def dot(vec_A, vec_B):
    return torch.sum(vec_A * vec_B, dim=-1)

# Given (..., 3) vectors and normals, projects out any components of vecs
# which lies in the direction of normals. Normals are assumed to be unit.

def project_to_tangent(vecs, unit_normals):
    dots = dot(vecs, unit_normals)
    return vecs - unit_normals * dots.unsqueeze(-1)

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
    # numpy in, numpy out
    (u, s, vh) = np.linalg.svd(points, full_matrices=False)
    normal = vh[:,2,:]
    return normal / np.linalg.norm(normal,axis=-1, keepdims=True)

def mesh_vertex_normals(verts, faces):
    # numpy in / out
    face_n = toNP(face_normals(torch.tensor(verts), torch.tensor(faces))) # ugly torch <---> numpy

    vertex_normals = np.zeros(verts.shape)
    for i in range(3):
        np.add.at(vertex_normals, faces[:,i], face_n)

    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals,axis=-1,keepdims=True)

    return vertex_normals


def vertex_normals(verts, faces, n_neighbors_cloud=30):
    verts_np = toNP(verts)

    if faces.numel() == 0: # point cloud

        _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
        neigh_points = verts_np[neigh_inds,:]
        neigh_points = neigh_points - verts_np[:,np.newaxis,:]
        normals = neighborhood_normal(neigh_points)

    else: # mesh

        normals = mesh_vertex_normals(verts_np, toNP(faces))

        # if any are NaN, wiggle slightly and recompute
        bad_normals_mask = np.isnan(normals).any(axis=1, keepdims=True)
        if bad_normals_mask.any():
            bbox = np.amax(verts_np, axis=0) - np.amin(verts_np, axis=0)
            scale = np.linalg.norm(bbox) * 1e-4
            wiggle = (np.random.RandomState(seed=777).rand(*verts.shape)-0.5) * scale
            wiggle_verts = verts_np + bad_normals_mask * wiggle
            normals = mesh_vertex_normals(wiggle_verts, toNP(faces))

        # if still NaN assign random normals (probably means unreferenced verts in mesh)
        bad_normals_mask = np.isnan(normals).any(axis=1)
        if bad_normals_mask.any():
            normals[bad_normals_mask,:] = (np.random.RandomState(seed=777).rand(*verts.shape)-0.5)[bad_normals_mask,:]
            normals = normals / np.linalg.norm(normals, axis=-1)[:,np.newaxis]


    normals = torch.from_numpy(normals).to(device=verts.device, dtype=verts.dtype)

    if torch.any(torch.isnan(normals)): raise ValueError("NaN normals :(")

    return normals


def vertex_normals_numpy(verts, faces=None, n_neighbors_cloud=30, normalize=True):
    """
    Pure NumPy implementation of vertex normal computation.
    """
    # Initialize faces if None
    if faces is None:
        faces = np.array([], dtype=np.int64)

    # Ensure inputs are numpy arrays
    verts = np.asarray(verts)
    faces = np.asarray(faces)

    if faces.size == 0:  # Point Cloud
        # Assuming find_knn returns indices as a numpy array
        _, neigh_inds = find_knn_numpy(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
        neigh_points = verts[neigh_inds, :]
        neigh_points = neigh_points - verts[:, np.newaxis, :]
        normals = neighborhood_normal(neigh_points)

    else:  # Mesh
        normals = mesh_vertex_normals(verts, faces)

        # Handle NaNs: wiggle and recompute
        bad_mask = np.isnan(normals).any(axis=1, keepdims=True)
        if bad_mask.any():
            bbox = np.amax(verts, axis=0) - np.amin(verts, axis=0)
            scale = np.linalg.norm(bbox) * 1e-4
            wiggle = (np.random.RandomState(seed=777).rand(*verts.shape) - 0.5) * scale
            wiggle_verts = verts + bad_mask * wiggle
            normals = mesh_vertex_normals(wiggle_verts, faces)

        # Handle remaining NaNs: assign random normals
        bad_mask_flat = np.isnan(normals).any(axis=1)
        if bad_mask_flat.any():
            normals[bad_mask_flat] = (np.random.RandomState(seed=777).rand(bad_mask_flat.sum(), 3) - 0.5)
            # Pre-normalize random normals to ensure consistency
            norms = np.linalg.norm(normals[bad_mask_flat], axis=-1, keepdims=True)
            normals[bad_mask_flat] /= (norms + 1e-8)

    # Global normalization
    if normalize:
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / (norms + 1e-8)

    # Check for NaNs
    if np.isnan(normals).any():
        raise ValueError("NaN normals detected in calculation :(")

    return normals

def build_tangent_frames(verts, faces, normals=None):

    V = verts.shape[0]
    dtype = verts.dtype
    device = verts.device

    if normals == None:
        vert_normals = vertex_normals(verts, faces)  # (V,3)
    else:
        vert_normals = normals

    # = find an orthogonal basis

    basis_cand1 = torch.tensor([1, 0, 0]).to(device=device, dtype=dtype).expand(V, -1)
    basis_cand2 = torch.tensor([0, 1, 0]).to(device=device, dtype=dtype).expand(V, -1)

    basisX = torch.where((torch.abs(dot(vert_normals, basis_cand1))
                          < 0.9).unsqueeze(-1), basis_cand1, basis_cand2)
    basisX = project_to_tangent(basisX, vert_normals)
    basisX = normalize(basisX)
    basisY = cross(vert_normals, basisX)
    frames = torch.stack((basisX, basisY, vert_normals), dim=-2)

    if torch.any(torch.isnan(frames)):
        raise ValueError("NaN coordinate frame! Must be very degenerate")

    return frames

def build_grad_point_cloud(verts, frames, n_neighbors_cloud=30):

    verts_np = toNP(verts)
    frames_np = toNP(frames)

    _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
    neigh_points = verts_np[neigh_inds,:]
    neigh_vecs = neigh_points - verts_np[:,np.newaxis,:]

    # TODO this could easily be way faster. For instance we could avoid the weird edges format and the corresponding pure-python loop via some numpy broadcasting of the same logic. The way it works right now is just to share code with the mesh version. But its low priority since its preprocessing code.

    edge_inds_from = np.repeat(np.arange(verts.shape[0]), n_neighbors_cloud)
    edges = np.stack((edge_inds_from, neigh_inds.flatten()))
    edge_tangent_vecs = edge_tangent_vectors(verts, frames, edges)

    return build_grad(verts_np, torch.tensor(edges), edge_tangent_vecs)


def edge_tangent_vectors(verts, frames, edges):
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    basisX = frames[edges[0, :], 0, :]
    basisY = frames[edges[0, :], 1, :]

    compX = dot(edge_vecs, basisX)
    compY = dot(edge_vecs, basisY)
    edge_tangent = torch.stack((compX, compY), dim=-1)

    return edge_tangent


def build_grad(verts, edges, edge_tangent_vectors):
    """
    Build a (V, V) complex sparse matrix grad operator. Given real inputs at vertices, produces a complex (vector value) at vertices giving the gradient. All values pointwise.
    - edges: (2, E)
    """

    edges_np = toNP(edges)
    edge_tangent_vectors_np = toNP(edge_tangent_vectors)

    # TODO find a way to do this in pure numpy?

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


def compute_geodesic_distance(verts, faces, source_idx=0):
    """P(x) = Geodesic distance from a source vertex."""
    # Solver requires numpy
    solver = pp3d.MeshHeatMethodDistanceSolver(verts.detach().cpu().numpy(),
                                               faces.detach().cpu().numpy())
    dist = solver.compute_distance(source_idx)
    return torch.from_numpy(dist).float().to(verts.device)

def compute_mean_curvature(verts, faces):
    """P(x) = Mean Curvature (approx via Cotan Laplacian)."""
    # |H * N| = ||Delta * pos|| / 2
    L = pp3d.cotan_laplacian(verts.cpu().numpy(), faces.cpu().numpy())
    mass = pp3d.vertex_areas(verts.cpu().numpy(), faces.cpu().numpy())
    # Laplacian of position
    lap_pos = L @ verts.cpu().numpy()
    # Curvature is proportional to the norm of the laplacian of the position
    H = np.linalg.norm(lap_pos, axis=1) / 2.0
    return torch.from_numpy(H).float().to(verts.device)

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

    return torch.from_numpy(sla.spsolve(L_mod.tocsr(), rhs)).float()

def compute_heat_kernel_signature(evals, evecs, t):
    """HKS(x, t) = sum_i exp(-lambda_i * t) * phi_i(x)^2"""
    # evals: (k,), evecs: (V, k)
    weights = torch.exp(-evals * t)
    hks = (evecs**2) @ weights
    return hks

def compute_operators(verts, faces, k_eig, normals=None, compute_features= True):
    """
    Builds spectral operators for a mesh/point cloud. Constructs mass matrix, eigenvalues/vectors for Laplacian, and gradient matrix.

    See get_operators() for a similar routine that wraps this one with a layer of caching.

    Torch in / torch out.

    Arguments:
      - vertices: (V,3) vertex positions
      - faces: (F,3) list of triangular faces. If empty, assumed to be a point cloud.
      - k_eig: number of eigenvectors to use

    Returns:
      - frames: (V,3,3) X/Y/Z coordinate frame at each vertex. Z coordinate is normal (e.g. [:,2,:] for normals)
      - massvec: (V) real diagonal of lumped mass matrix
      - L: (VxV) real sparse matrix of (weak) Laplacian
      - evals: (k) list of eigenvalues of the Laplacian
      - evecs: (V,k) list of eigenvectors of the Laplacian
      - gradX: (VxV) sparse matrix which gives X-component of gradient in the local basis at the vertex
      - gradY: same as gradX but for Y-component of gradient

    PyTorch doesn't seem to like complex sparse matrices, so we store the "real" and "imaginary" (aka X and Y) gradient matrices separately, rather than as one complex sparse matrix.

    Note: for a generalized eigenvalue problem, the mass matrix matters! The eigenvectors are only othrthonormal with respect to the mass matrix, like v^H M v, so the mass (given as the diagonal vector massvec) needs to be used in projections, etc.
    """

    device = verts.device
    dtype = verts.dtype
    V = verts.shape[0]
    is_cloud = faces.numel() == 0

    eps = 1e-8

    verts_np = toNP(verts).astype(np.float64)
    faces_np = toNP(faces)
    frames = build_tangent_frames(verts, faces, normals=normals)
    frames_np = toNP(frames)

    # Build the scalar Laplacian
    if is_cloud:
        L, M = robust_laplacian.point_cloud_laplacian(verts_np)
        massvec_np = M.diagonal()
    else:
        # L, M = robust_laplacian.mesh_laplacian(verts_np, faces_np)
        # massvec_np = M.diagonal()
        L = pp3d.cotan_laplacian(verts_np, faces_np, denom_eps=1e-10)
        massvec_np = pp3d.vertex_areas(verts_np, faces_np)
        massvec_np += eps * np.mean(massvec_np)

    if(np.isnan(L.data).any()):
        raise RuntimeError("NaN Laplace matrix")
    if(np.isnan(massvec_np).any()):
        raise RuntimeError("NaN mass matrix")

    # Read off neighbors & rotations from the Laplacian
    L_coo = L.tocoo()
    inds_row = L_coo.row
    inds_col = L_coo.col

    # === Compute the eigenbasis
    if k_eig > 0:
        # Prepare matrices
        L_eigsh = (L + scipy.sparse.identity(L.shape[0])*eps).tocsc()
        massvec_eigsh = massvec_np
        Mmat = scipy.sparse.diags(massvec_eigsh)
        eigs_sigma = eps

        failcount = 0
        while True:
            try:
                # We would be happy here to lower tol or maxiter since we don't need these to be super precise, but for some reason those parameters seem to have no effect
                evals_np, evecs_np = sla.eigsh(L_eigsh, k=k_eig, M=Mmat, sigma=eigs_sigma)

                # Clip off any eigenvalues that end up slightly negative due to numerical weirdness
                evals_np = np.clip(evals_np, a_min=0., a_max=float('inf'))

                break
            except Exception as e:
                print(e)
                if(failcount > 3):
                    raise ValueError("failed to compute eigendecomp")
                failcount += 1
                print("--- decomp failed; adding eps ===> count: " + str(failcount))
                L_eigsh = L_eigsh + scipy.sparse.identity(L.shape[0]) * (eps * 10**failcount)


    else: #k_eig == 0
        evals_np = np.zeros((0))
        evecs_np = np.zeros((verts.shape[0],0))


    # == Build gradient matrices

    # For meshes, we use the same edges as were used to build the Laplacian. For point clouds, use a whole local neighborhood
    if is_cloud:
        grad_mat_np = build_grad_point_cloud(verts, frames)
    else:
        edges = torch.tensor(np.stack((inds_row, inds_col), axis=0), device=device, dtype=faces.dtype)
        edge_vecs = edge_tangent_vectors(verts, frames, edges)
        grad_mat_np = build_grad(verts, edges, edge_vecs)


    # Split complex gradient in to two real sparse mats (torch doesn't like complex sparse matrices)
    gradX_np = np.real(grad_mat_np)
    gradY_np = np.imag(grad_mat_np)

    # === Convert back to torch
    massvec = torch.from_numpy(massvec_np).to(device=device, dtype=dtype)
    L = sparse_np_to_torch(L).to(device=device, dtype=dtype)
    evals = torch.from_numpy(evals_np).to(device=device, dtype=dtype)
    evecs = torch.from_numpy(evecs_np).to(device=device, dtype=dtype)
    gradX = sparse_np_to_torch(gradX_np).to(device=device, dtype=dtype)
    gradY = sparse_np_to_torch(gradY_np).to(device=device, dtype=dtype)

    features = {}
    if compute_features:
        # 1. Geodesic Distance (from vertex 0 as default)
        #features['geodesic'] = compute_geodesic_distance(verts, faces, source_idx=0)

        # 2. Mean Curvature
        features['mean_curvature'] = compute_mean_curvature(verts, faces)

        # 3. Heat Kernel Signature
        if k_eig > 0:
            features['hks'] = compute_heat_kernel_signature(evals, evecs, t=1.0)

    return frames, massvec, L, evals, evecs, gradX, gradY, features


def get_all_operators(verts_list, faces_list, k_eig, op_cache_dir=None, normals=None):
    N = len(verts_list)

    frames = [None] * N
    massvec = [None] * N
    L = [None] * N
    evals = [None] * N
    evecs = [None] * N
    gradX = [None] * N
    gradY = [None] * N
    features = [None] * N

    inds = [i for i in range(N)]
    # process in random order
    # random.shuffle(inds)

    for num, i in enumerate(inds):
        print("get_all_operators() processing {} / {} {:.3f}%".format(num, N, num / N * 100))
        if normals is None:
            outputs = get_operators(verts_list[i], faces_list[i], k_eig, op_cache_dir)
        else:
            outputs = get_operators(verts_list[i], faces_list[i], k_eig, op_cache_dir, normals=normals[i])
        frames[i] = outputs[0]
        massvec[i] = outputs[1]
        L[i] = outputs[2]
        evals[i] = outputs[3]
        evecs[i] = outputs[4]
        gradX[i] = outputs[5]
        gradY[i] = outputs[6]
        features[i] = outputs[7]

    return frames, massvec, L, evals, evecs, gradX, gradY, features

def get_operators(verts, faces, k_eig=128, op_cache_dir=None, normals=None, overwrite_cache=False):
    """
    See documentation for compute_operators(). This essentailly just wraps a call to compute_operators, using a cache if possible.
    All arrays are always computed using double precision for stability, then truncated to single precision floats to store on disk, and finally returned as a tensor with dtype/device matching the `verts` input.
    """

    device = verts.device
    dtype = verts.dtype
    verts_np = toNP(verts)
    faces_np = toNP(faces)
    is_cloud = faces.numel() == 0

    if(np.isnan(verts_np).any()):
        raise RuntimeError("tried to construct operators from NaN verts")

    # Check the cache directory
    # Note 1: Collisions here are exceptionally unlikely, so we could probably just use the hash...
    #         but for good measure we check values nonetheless.
    # Note 2: There is a small possibility for race conditions to lead to bucket gaps or duplicate
    #         entries in this cache. The good news is that that is totally fine, and at most slightly
    #         slows performance with rare extra cache misses.
    found = False
    if op_cache_dir is not None:
        ensure_dir_exists(op_cache_dir)
        hash_key_str = str(hash_arrays((verts_np, faces_np)))
        # print("Building operators for input with hash: " + hash_key_str)

        # Search through buckets with matching hashes.  When the loop exits, this
        # is the bucket index of the file we should write to.
        i_cache_search = 0
        while True:

            # Form the name of the file to check
            search_path = os.path.join(
                op_cache_dir,
                hash_key_str + "_" + str(i_cache_search) + ".npz")

            try:
                # print('loading path: ' + str(search_path))
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile["verts"]
                cache_faces = npzfile["faces"]
                cache_k_eig = npzfile["k_eig"].item()

                # If the cache doesn't match, keep looking
                if (not np.array_equal(verts, cache_verts)) or (not np.array_equal(faces, cache_faces)):
                    i_cache_search += 1
                    print("hash collision! searching next.")
                    continue

                # print("  cache hit!")

                # If we're overwriting, or there aren't enough eigenvalues, just delete it; we'll create a new
                # entry below more eigenvalues
                if overwrite_cache:
                    print("  overwriting cache by request")
                    os.remove(search_path)
                    break

                if cache_k_eig < k_eig:
                    print("  overwriting cache --- not enough eigenvalues")
                    os.remove(search_path)
                    break

                if "L_data" not in npzfile:
                    print("  overwriting cache --- entries are absent")
                    os.remove(search_path)
                    break


                def read_sp_mat(prefix):
                    data = npzfile[prefix + "_data"]
                    indices = npzfile[prefix + "_indices"]
                    indptr = npzfile[prefix + "_indptr"]
                    shape = npzfile[prefix + "_shape"]
                    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
                    return mat

                # This entry matches! Return it.
                frames = npzfile["frames"]
                mass = npzfile["mass"]
                L = read_sp_mat("L")
                evals = npzfile["evals"][:k_eig]
                evecs = npzfile["evecs"][:,:k_eig]
                gradX = read_sp_mat("gradX")
                gradY = read_sp_mat("gradY")

                frames = torch.from_numpy(frames).to(device=device, dtype=dtype)
                mass = torch.from_numpy(mass).to(device=device, dtype=dtype)
                L = sparse_np_to_torch(L).to(device=device, dtype=dtype)
                evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
                evecs = torch.from_numpy(evecs).to(device=device, dtype=dtype)
                gradX = sparse_np_to_torch(gradX).to(device=device, dtype=dtype)
                gradY = sparse_np_to_torch(gradY).to(device=device, dtype=dtype)

                features = {}
                if "feature_keys" in npzfile:
                    for key in npzfile["feature_keys"]:
                        feat_val = npzfile[f"feat_{key}"]
                        features[key] = torch.from_numpy(feat_val).to(device=device, dtype=dtype)
                else:
                    # Fallback if loading an old cache file without features
                    features = None

                found = True

                break

            except FileNotFoundError:
                print("  cache miss -- constructing operators")
                break

            except Exception as E:
                print("unexpected error loading file: " + str(E))
                print("-- constructing operators")
                break

    if not found:

        # No matching entry found; recompute.
        frames, mass, L, evals, evecs, gradX, gradY, features = compute_operators(verts, faces, k_eig, normals=normals)

        dtype_np = np.float32

        # Store it in the cache
        if op_cache_dir is not None:

            L_np = sparse_torch_to_np(L).astype(dtype_np)
            gradX_np = sparse_torch_to_np(gradX).astype(dtype_np)
            gradY_np = sparse_torch_to_np(gradY).astype(dtype_np)

            save_dict = {
                "verts": verts_np.astype(dtype_np),
                "frames": toNP(frames).astype(dtype_np),
                "faces": faces_np,
                "k_eig": k_eig,
                "mass": toNP(mass).astype(dtype_np),
                "L_data": L_np.data.astype(dtype_np),
                "L_indices": L_np.indices,
                "L_indptr": L_np.indptr,
                "L_shape": L_np.shape,
                "evals": toNP(evals).astype(dtype_np),
                "evecs": toNP(evecs).astype(dtype_np),
                "gradX_data": gradX_np.data.astype(dtype_np),
                "gradX_indices": gradX_np.indices,
                "gradX_indptr": gradX_np.indptr,
                "gradX_shape": gradX_np.shape,
                "gradY_data": gradY_np.data.astype(dtype_np),
                "gradY_indices": gradY_np.indices,
                "gradY_indptr": gradY_np.indptr,
                "gradY_shape": gradY_np.shape,
            }

            # Save features dictionary
            save_dict["feature_keys"] = list(features.keys())
            for key, val in features.items():
                save_dict[f"feat_{key}"] = toNP(val).astype(dtype_np)

            np.savez(search_path, **save_dict)
    return frames, mass, L, evals, evecs, gradX, gradY, features

def to_basis(values, basis, massvec):
    """
    Transform data in to an orthonormal basis (where orthonormal is wrt to massvec)
    Inputs:
      - values: (B,V,D)
      - basis: (B,V,K)
      - massvec: (B,V)
    Outputs:
      - (B,K,D) transformed values
    """
    basisT = basis.transpose(-2, -1)
    return torch.matmul(basisT, values * massvec.unsqueeze(-1))



def from_basis(values, basis):
    """
    Transform data out of an orthonormal basis
    Inputs:
      - values: (K,D)
      - basis: (V,K)
    Outputs:
      - (V,D) reconstructed values
    """
    if values.is_complex() or basis.is_complex():
        return cmatmul(ensure_complex(basis), ensure_complex(values))
    else:
        return torch.matmul(basis, values)

def to_laplace_basis(f, evecs, mass, P, sigma):
    # 1. Force everything to be (V, 1) column vectors immediately
    f = f.view(-1, 1)
    P = P.view(-1, 1)
    mass = mass.view(-1, 1)

    # 2. Decay: (V, 1)
    decay = torch.exp(-sigma * P)

    # 3. f_weighted: (V, 1) * (V, 1) = (V, 1) [ELEMENT-WISE]
    f_weighted = decay * f

    # 4. Coefficients: (K, V) @ (V, 1) = (K, 1)
    alphas = evecs.T @ (mass * f_weighted)

    # 5. Basis Functions: (V, K) * (V, 1) [BROADCASTING]
    # decay is (V, 1), evecs is (V, K), result is (V, K)
    localized_basis_fcts = evecs * decay

    return alphas.view(-1, 1), localized_basis_fcts

def from_laplace_basis(alphas, evecs, P, sigmas):
    # Ensure P is (V, 1) to enable broadcasting against (1, K)
    if P.dim() == 1:
        P = P.view(-1, 1)

    decay = torch.exp(-P * sigmas.t())
    scaled_basis = evecs * decay

    reconstruction = scaled_basis @ alphas.view(-1, 1)

    return reconstruction, scaled_basis

def compute_hks(evals, evecs, scales):
    """
    Inputs:
      - evals: (K) eigenvalues
      - evecs: (V,K) values
      - scales: (S) times
    Outputs:
      - (V,S) hks values
    """

    # expand batch
    if len(evals.shape) == 1:
        expand_batch = True
        evals = evals.unsqueeze(0)
        evecs = evecs.unsqueeze(0)
        scales = scales.unsqueeze(0)
    else:
        expand_batch = False

    # TODO could be a matmul
    power_coefs = torch.exp(-evals.unsqueeze(1) * scales.unsqueeze(-1)).unsqueeze(1) # (B,1,S,K)
    terms = power_coefs * (evecs * evecs).unsqueeze(2)  # (B,V,S,K)

    out = torch.sum(terms, dim=-1) # (B,V,S)

    if expand_batch:
        return out.squeeze(0)
    else:
        return out

def compute_hks_autoscale(evals, evecs, count):
    # these scales roughly approximate those suggested in the hks paper
    scales = torch.logspace(-2, 0., steps=count, device=evals.device, dtype=evals.dtype)
    return compute_hks(evals, evecs, scales)

def normalize_positions(pos, faces=None, method='mean', scale_method='max_rad'):
    # center and unit-scale positions

    if method == 'mean':
        # center using the average point position
        pos = (pos - torch.mean(pos, dim=-2, keepdim=True))
    elif method == 'bbox':
        # center via the middle of the axis-aligned bounding box
        bbox_min = torch.min(pos, dim=-2).values
        bbox_max = torch.max(pos, dim=-2).values
        center = (bbox_max + bbox_min) / 2.
        pos -= center.unsqueeze(-2)
    else:
        raise ValueError("unrecognized method")

    if scale_method == 'max_rad':
        scale = torch.max(norm(pos), dim=-1, keepdim=True).values.unsqueeze(-1)
        pos = pos / scale
    elif scale_method == 'area':
        if faces is None:
            raise ValueError("must pass faces for area normalization")
        coords = pos[faces]
        vec_A = coords[:, 1, :] - coords[:, 0, :]
        vec_B = coords[:, 2, :] - coords[:, 0, :]
        face_areas = torch.norm(torch.cross(vec_A, vec_B, dim=-1), dim=1) * 0.5
        total_area = torch.sum(face_areas)
        scale = (1. / torch.sqrt(total_area))
        pos = pos * scale
    else:
        raise ValueError("unrecognized scale method")
    return pos

# Finds the k nearest neighbors of source on target.
# Return is two tensors (distances, indices). Returned points will be sorted in increasing order of distance.
def find_knn(points_source, points_target, k, largest=False, omit_diagonal=False, method='brute'):

    if omit_diagonal and points_source.shape[0] != points_target.shape[0]:
        raise ValueError("omit_diagonal can only be used when source and target are same shape")

    if method != 'cpu_kd' and points_source.shape[0] * points_target.shape[0] > 1e8:
        method = 'cpu_kd'
        print("switching to cpu_kd knn")

    if method == 'brute':

        # Expand so both are NxMx3 tensor
        points_source_expand = points_source.unsqueeze(1)
        points_source_expand = points_source_expand.expand(-1, points_target.shape[0], -1)
        points_target_expand = points_target.unsqueeze(0)
        points_target_expand = points_target_expand.expand(points_source.shape[0], -1, -1)

        diff_mat = points_source_expand - points_target_expand
        dist_mat = norm(diff_mat)

        if omit_diagonal:
            torch.diagonal(dist_mat)[:] = float('inf')

        result = torch.topk(dist_mat, k=k, largest=largest, sorted=True)
        return result

    elif method == 'cpu_kd':

        if largest:
            raise ValueError("can't do largest with cpu_kd")

        points_source_np = toNP(points_source)
        points_target_np = toNP(points_target)

        # Build the tree
        kd_tree = sklearn.neighbors.KDTree(points_target_np)

        k_search = k+1 if omit_diagonal else k
        _, neighbors = kd_tree.query(points_source_np, k=k_search)

        if omit_diagonal:
            # Mask out self element
            mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

            # make sure we mask out exactly one element in each row, in rare case of many duplicate points
            mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False

            neighbors = neighbors[mask].reshape((neighbors.shape[0], neighbors.shape[1]-1))

        inds = torch.tensor(neighbors, device=points_source.device, dtype=torch.int64)
        dists = norm(points_source.unsqueeze(1).expand(-1, k, -1) - points_target[inds])

        return dists, inds

    else:
        raise ValueError("unrecognized method")


def farthest_point_sampling(points, n_sample):
    # Torch in, torch out. Returns a |V| mask with n_sample elements set to true.

    N = points.shape[0]
    if(n_sample > N): raise ValueError("not enough points to sample")

    chosen_mask = torch.zeros(N, dtype=torch.bool, device=points.device)
    min_dists = torch.ones(N, dtype=points.dtype, device=points.device) * float('inf')

    # pick the centermost first point
    points = normalize_positions(points)
    i = torch.min(norm2(points), dim=0).indices
    chosen_mask[i] = True

    for _ in range(n_sample-1):

        # update distance
        dists = norm2(points[i,:].unsqueeze(0) - points)
        min_dists = torch.minimum(dists, min_dists)

        # take the farthest
        i = torch.max(min_dists,dim=0).indices.item()
        chosen_mask[i] = True

    return chosen_mask


def geodesic_label_errors(target_verts, target_faces, pred_labels, gt_labels, normalization='diameter', geodesic_cache_dir=None):
    """
    Return a vector of distances between predicted and ground-truth lables (normalized by geodesic diameter or area)

    This method is SLOW when it needs to recompute geodesic distances.
    """

    # move all to numpy cpu
    target_verts = toNP(target_verts)
    target_faces = toNP(target_faces)

    pred_labels = toNP(pred_labels)
    gt_labels = toNP(gt_labels)

    dists = get_all_pairs_geodesic_distance(target_verts, target_faces, geodesic_cache_dir)

    result_dists = dists[pred_labels, gt_labels]

    if normalization == 'diameter':
        geodesic_diameter = np.max(dists)
        normalized_result_dists = result_dists / geodesic_diameter
    elif normalization == 'area':
        total_area = torch.sum(face_area(torch.tensor(target_verts), torch.tensor(target_faces)))
        normalized_result_dists = result_dists / torch.sqrt(total_area)
    else:
        raise ValueError('unrecognized normalization')

    return normalized_result_dists

# This function and the helper class below are to support parallel computation of all-pairs geodesic distance
def all_pairs_geodesic_worker(verts, faces, i):
    import igl

    N = verts.shape[0]

    # TODO: this re-does a ton of work, since it is called independently each time. Some custom C++ code could surely make it faster.
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
        # print("Building operators for input with hash: " + hash_key_str)

        # Search through buckets with matching hashes.  When the loop exits, this
        # is the bucket index of the file we should write to.
        i_cache_search = 0
        while True:

            # Form the name of the file to check
            search_path = os.path.join(
                geodesic_cache_dir,
                hash_key_str + "_" + str(i_cache_search) + ".npz")

            try:
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile["verts"]
                cache_faces = npzfile["faces"]

                # If the cache doesn't match, keep looking
                if (not np.array_equal(verts_np, cache_verts)) or (not np.array_equal(faces_np, cache_faces)):
                    i_cache_search += 1
                    continue

                # This entry matches! Return it.
                found = True
                result_dists = npzfile["dist"]
                break

            except FileNotFoundError:
                break

    if not found:

        print("Computing all-pairs geodesic distance (warning: SLOW!)")

        # Not found, compute from scratch
        # warning: slowwwwwww

        N = verts_np.shape[0]

        try:
            pool = Pool(None) # on 8 processors
            engine = AllPairsGeodesicEngine(verts_np, faces_np)
            outputs = pool.map(engine, range(N))
        finally: # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()

        result_dists = np.array(outputs)

        # replace any failed values with nan
        result_dists = np.nan_to_num(result_dists, nan=np.nan, posinf=np.nan, neginf=np.nan)

        # we expect that this should be a symmetric matrix, but it might not be. Take the min of the symmetric values to make it symmetric
        result_dists = np.fmin(result_dists, np.transpose(result_dists))

        # on rare occaisions MMP fails, yielding nan/inf; set it to the largest non-failed value if so
        max_dist = np.nanmax(result_dists)
        result_dists = np.nan_to_num(result_dists, nan=max_dist, posinf=max_dist, neginf=max_dist)

        print("...finished computing all-pairs geodesic distance")

        # put it in the cache if possible
        if geodesic_cache_dir is not None:

            print("saving geodesic distances to cache: " + str(geodesic_cache_dir))

            # TODO we're potentially saving a double precision but only using a single
            # precision here; could save storage by always saving as floats
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
