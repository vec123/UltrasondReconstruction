import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def solve_laplacian_deformation(template_vertices, target_vertices, S_target, L, lambda_reg=1.0):
    N = template_vertices.shape[0]
    
    # 1. Build the Linear System (N+M, N)
    lhs = sparse.vstack([lambda_reg * L, S_target], format='csr')
    
    # 2. Build RHS
    # Extract indices of the vertices that are constrained by S_target
    # This assumes S_target is a CSR matrix
    # The indices of the selected vertices are stored in S_target.indices
    selected_indices = S_target.indices
    
    # rhs_top: (N, 3)
    rhs_top = lambda_reg * (L @ template_vertices)
    
    # rhs_bottom: (M, 3) -> ONLY the vertices selected by the mask
    rhs_bottom = target_vertices[selected_indices] 
    
    rhs = np.vstack([rhs_top, rhs_bottom])
    
    # 3. Solve
    A = lhs.T @ lhs
    b = lhs.T @ rhs
    
    return spsolve(A, b)