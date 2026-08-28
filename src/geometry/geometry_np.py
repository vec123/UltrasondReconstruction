import numpy as np
from sklearn.neighbors import KDTree
import robust_laplacian
import potpourri3d as pp3d
import scipy.sparse.linalg as sla
import scipy.sparse as sp
from scipy.sparse.linalg import lobpcg
import pyamg
import scipy

#--------------------------------------- Basis functions
def cross(vec_A, vec_B):
    # np.cross does not use an axis argument for the dimension to cross
    # it assumes the last dimension by default, which is perfect for (..., 3)
    return np.cross(vec_A, vec_B)

def dot(vec_A, vec_B):
    # Use axis instead of dim
    return np.sum(vec_A * vec_B, axis=-1)

def norm(x, highdim=False):
    """
    Computes norm of an array of vectors along the last dimension.
    """
    # Use np.linalg.norm and axis instead of dim
    return np.linalg.norm(x, axis=-1)

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
    #face_n = toNP(face_normals(torch.tensor(verts), torch.tensor(faces))) # ugly torch <---> numpy
    face_n = faces
    vertex_normals = np.zeros(verts.shape)
    for i in range(3):
        np.add.at(vertex_normals, faces[:,i], face_n)

    vertex_normals = vertex_normals / (np.linalg.norm(vertex_normals,axis=-1,keepdims=True) +1e-8)

    return vertex_normals

def find_knn_numpy(points_source, points_target, k, largest=False, omit_diagonal=False, method='brute'):
    """
    Pure NumPy implementation of K-Nearest Neighbors search.
    """
    # Cast to numpy arrays
    points_source = np.asarray(points_source)
    points_target = np.asarray(points_target)

    if omit_diagonal and points_source.shape[0] != points_target.shape[0]:
        raise ValueError("omit_diagonal can only be used when source and target are same shape")

    # Automated switching to KDTree for large datasets
    if method != 'cpu_kd' and points_source.shape[0] * points_target.shape[0] > 1e8:
        method = 'cpu_kd'
        print("switching to cpu_kd knn")

    if method == 'brute':
        # NumPy broadcasting equivalent of unsqueeze/expand
        # Creates (N, 1, 3) and (1, M, 3) to get (N, M, 3) diff
        diff_mat = points_source[:, np.newaxis, :] - points_target[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff_mat, axis=-1)

        if omit_diagonal:
            np.fill_diagonal(dist_mat, np.inf)

        # NumPy does not have a direct topk for all rows, use argpartition
        if largest:
            # -k for largest, then sort descending
            indices = np.argpartition(-dist_mat, k, axis=1)[:, :k]
        else:
            # k for smallest
            indices = np.argpartition(dist_mat, k, axis=1)[:, :k]

        # Re-sort because argpartition is not sorted
        # Extract distances for the chosen indices
        rows = np.arange(dist_mat.shape[0])[:, np.newaxis]
        dists = dist_mat[rows, indices]

        # Sort results
        order = np.argsort(dists, axis=1, reverse=not largest)
        dists = dists[rows, order]
        indices = indices[rows, order]

        return dists, indices

    elif method == 'cpu_kd':
        if largest:
            raise ValueError("can't do largest with cpu_kd")

        tree = KDTree(points_target)

        k_search = k + 1 if omit_diagonal else k
        _, neighbors = tree.query(points_source, k=k_search)

        if omit_diagonal:
            # Mask out self-index
            mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]
            # Handle duplicate points edge case
            mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False
            neighbors = neighbors[mask].reshape((neighbors.shape[0], k))

        # Calculate distances for the identified neighbors
        # Using fancy indexing: points_target[neighbors] is (N, k, 3)
        diff = points_source[:, np.newaxis, :] - points_target[neighbors]
        dists = np.linalg.norm(diff, axis=-1)

        return dists, neighbors

    else:
        raise ValueError("unrecognized method")


def project_to_tangent(vecs, unit_normals):
    dots = dot(vecs, unit_normals)
    return vecs - unit_normals * dots[..., np.newaxis]

#--------------------------------------- Geometric functions
def vertex_normals(verts, faces=None, n_neighbors_cloud=30, normalize=True):
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


#--------------------------------------- Spectral Computations
def compute_spectral_operators_fast(verts, faces=None, k_eig=20):
    is_mesh = (faces is not None) and (faces.size > 0)
    L, M = robust_laplacian.mesh_laplacian(verts, faces) if is_mesh else robust_laplacian.point_cloud_laplacian(verts)

    # 1. Use Algebraic Multigrid (AMG) as a preconditioner
    # This is the 'secret sauce' for speed on 30k+ meshes
    ml = pyamg.smoothed_aggregation_solver(L.tocsr())
    M_prec = ml.aspreconditioner(cycle='V')

    # 2. Set up LOBPCG
    n = L.shape[0]
    X = np.random.rand(n, k_eig)

    # LOBPCG with AMG preconditioner
    evals, evecs = lobpcg(L, X, B=M, M=M_prec, largest=False, tol=1e-5, maxiter=200)

    # 3. Post-process
    idx = np.argsort(evals)
    evals = np.clip(evals[idx], 0, None)
    evecs = evecs[:, idx]

    return L, M, evals, evecs

def compute_spectral_operators(verts, faces=None, k_eig=20, downsampling = True):
    # 1. Downsampling (already in your code)
    if downsampling:
        if verts.shape[0] > 10000:
            print(f"Downsampling from {verts.shape[0]} vertices")
            step = verts.shape[0] // 10000
            verts = verts[::step]
            print(f"to {verts.shape[0]} vertices")
            faces = None

    # 2. Construction
    L, M = robust_laplacian.point_cloud_laplacian(verts)

    # 3. Transform to Standard Eigenvalue Problem (A * x = lambda * x)
    # A = M^-1 * L
    # We use the square root of the diagonal mass matrix for symmetric scaling
    M_diag = M.diagonal()
    M_diag[M_diag < 1e-12] = 1e-12
    inv_sqrt_M = sp.diags(1.0 / np.sqrt(M_diag))

    # Symmetric A matrix
    A = inv_sqrt_M @ L @ inv_sqrt_M

    # 4. Use LOBPCG (Iterative, NO LU factorization)
    # This avoids the "SuperLU" bottleneck entirely.
    n = A.shape[0]
    X = np.random.rand(n, k_eig)

    print("Solving eigenspace with LOBPCG...")
    evals, evecs_scaled = lobpcg(A, X, largest=False, tol=1e-3, maxiter=5000)

    # 5. Transform eigenvectors back to original space
    evecs = inv_sqrt_M @ evecs_scaled

    idx = np.argsort(evals)
    sorted_evals = evals[idx]
    formatted = [f"{val:.8f}" for val in sorted_evals[:5]]

    return L, M, sorted_evals, evecs[:, idx]

def test_LBO_spectrum(L, M, spectral_evals, spectral_evecs, k_eig):
    print("spectral_evals: ", spectral_evals)
    first_val = float(spectral_evals[0])

    print(f"First eigenvalue: {first_val:.2e}")

    # You can add additional checks here if needed
    if first_val < 0:
        print("Warning: First eigenvalue is negative (check your Laplacian construction).")
    print(f"Std dev of first eigenvector: {np.std(spectral_evecs[:, 0]):.2e}")
    ortho_check = spectral_evecs.T @ M @ spectral_evecs
    identity_diff = np.linalg.norm(ortho_check - np.eye(k_eig))
    print(f"Orthonormality error: {identity_diff:.2e}")
    residuals = []
    for i in range(k_eig):
        res = np.linalg.norm(L @ spectral_evecs[:, i] - spectral_evals[i] * (M @ spectral_evecs[:, i]))
        residuals.append(res)
    print(f"Mean residual: {np.mean(residuals):.2e}")
    if np.any(spectral_evals < -1e-6):
        print("Warning: Negative eigenvalues detected!")

def save_spectral_data(filename, L, M, evals, evecs):
    """
    Saves spectral operator data to a compressed .npz file.

    Arguments:
        filename: Path to save (e.g., 'data.npz')
        L: Sparse matrix (CSR)
        M: Sparse matrix (CSR)
        evals: (k,) array
        evecs: (V, k) array
    """
    # Save sparse matrices by storing their components (data, indices, indptr, shape)
    # This is more efficient than pickling
    np.savez_compressed(
        filename,
        L_data=L.data, L_indices=L.indices, L_indptr=L.indptr, L_shape=L.shape,
        M_data=M.data, M_indices=M.indices, M_indptr=M.indptr, M_shape=M.shape,
        evals=evals,
        evecs=evecs
    )
    print(f"Spectral data successfully saved to {filename}")

def load_spectral_data(filename):
    """Loads the spectral data back into memory."""
    data = np.load(filename)

    L = sp.csr_matrix((data['L_data'], data['L_indices'], data['L_indptr']), shape=data['L_shape'])
    M = sp.csr_matrix((data['M_data'], data['M_indices'], data['M_indptr']), shape=data['M_shape'])

    return L, M, data['evals'], data['evecs']




#---------------------------------------Compute Grads
def build_tangent_frames(verts, faces, normals=None):

    V = verts.shape[0]
    dtype = verts.dtype
    device = verts.device

    if normals == None:
        vert_normals = vertex_normals(verts, faces)  # (V,3)
    else:
        vert_normals = normals

    # = find an orthogonal basis

    basis_cand1 = np.tile([1.0, 0.0, 0.0], (V, 1))
    basis_cand2 = np.tile([0.0, 1.0, 0.0], (V, 1))

    basisX = np.where((np.abs(dot(vert_normals, basis_cand1))
                          < 0.9).unsqueeze(-1), basis_cand1, basis_cand2)
    basisX = project_to_tangent(basisX, vert_normals)
    basisX = normalize(basisX)
    basisY = cross(vert_normals, basisX)

    frames = np.stack((basisX, basisY, vert_normals), dim=-2)

    if np.any(np.isnan(frames)):
        raise ValueError("NaN coordinate frame! Must be very degenerate")

    return frames

def build_grad(verts, edges, edge_tangent_vectors):
    """
    Build a (V, V) complex sparse matrix grad operator. Given real inputs at vertices, produces a complex (vector value) at vertices giving the gradient. All values pointwise.
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

def edge_tangent_vectors(verts, frames, edges):
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    basisX = frames[edges[0, :], 0, :]
    basisY = frames[edges[0, :], 1, :]

    compX = dot(edge_vecs, basisX)
    compY = dot(edge_vecs, basisY)
    edge_tangent = np.stack((compX, compY), dim=-1)

    return edge_tangent

def build_grad_point_cloud(verts, frames, n_neighbors_cloud=30):

    verts_np = verts
    frames_np = frames


    _, neigh_inds = find_knn_numpy(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
    neigh_points = verts_np[neigh_inds,:]
    neigh_vecs = neigh_points - verts_np[:,np.newaxis,:]

    edge_inds_from = np.repeat(np.arange(verts.shape[0]), n_neighbors_cloud)
    edges = np.stack((edge_inds_from, neigh_inds.flatten()))
    edge_tangent_vecs = edge_tangent_vectors(verts, frames, edges)

    return build_grad(verts_np, edges, edge_tangent_vecs)

def compute_grads( verts, faces=None, L=None, normals=None):

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
