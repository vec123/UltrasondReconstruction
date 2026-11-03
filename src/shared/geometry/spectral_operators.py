"""Laplace-Beltrami spectral operators (numpy, LOBPCG + AMG-accelerated).

A second, numpy-only spectral pipeline alongside geometry.py's own
``compute_operators``/``get_operators`` (torch-in/torch-out, used by the
nrrd_to_vtp preprocessing stage). This one is what the SSM pipeline expects:
an eigendecomposition of the mesh/point-cloud Laplacian solved iteratively
(LOBPCG, with an algebraic-multigrid preconditioner for speed on large
meshes) rather than via the cached, sparse direct solve the other pipeline
uses. Keep the two separate rather than merged - they serve different
callers with different cost/precision tradeoffs.
"""

import numpy as np
import pyamg
import robust_laplacian
import scipy.sparse as sp
from scipy.sparse.linalg import lobpcg


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
