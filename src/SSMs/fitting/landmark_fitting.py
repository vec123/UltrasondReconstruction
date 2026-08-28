
import numpy as np
from scipy.sparse.linalg import lsqr

def fit_landmarks(phi_r,
                  mean_r,
                  landmark_targets):

    b = (
        landmark_targets.reshape(-1)
        - mean_r.reshape(-1)
    )

    alpha, *_ = np.linalg.lstsq(
        phi_r,
        b,
        rcond=None
    )

    return alpha

def solve_coordinate(A, b):

    return lsqr(A, b)[0]