import numpy as np

def match_landmarks(model_names, model_indices, target_dict):
    """
    Returns matched arrays (source, target)
    """

    src = []
    tgt = []

    for name, idx in zip(model_names, model_indices):

        if name in target_dict:
            src.append(idx)
            tgt.append(target_dict[name])

    return np.array(src), np.array(tgt)

def extract_points_from_indices(vertices, indices):
    return vertices[indices]

import numpy as np


def estimate_similarity_transform(source_points,
                                  target_points,
                                  allow_scale=True):
    """
    Compute similarity transform parameters (R, t, s)
    aligning source -> target.

    Returns
    -------
    R : (3,3)
    t : (3,)
    s : float
    """

    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)

    assert source.shape == target.shape, "Point sets must match"

    # centroids
    mu_src = source.mean(axis=0)
    mu_tgt = target.mean(axis=0)

    X = source - mu_src
    Y = target - mu_tgt

    # covariance
    H = X.T @ Y

    # SVD (Kabsch)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # reflection fix
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # scale
    if allow_scale:
        var_src = np.sum(X ** 2)
        s = np.sum(S) / var_src
    else:
        s = 1.0

    # translation
    t = mu_tgt - s * (R @ mu_src)

    return R, t, s


def apply_similarity_transform(points,
                                R,
                                t,
                                s=1.0):
    """
    Apply similarity transform:
        x' = s * R x + t

    Parameters
    ----------
    points : (N,3)
    R : (3,3)
    t : (3,)
    s : float

    Returns
    -------
    transformed : (N,3)
    """

    points = np.asarray(points, dtype=np.float64)

    return s * (points @ R.T) + t


def load_similarity_npz(path):
    """Reads an (R, t, s) similarity transform from a .npz file.

    Written by scripts/pipeline.py as gmm_2_template.npz and by
    scripts/align_landmarks.py as rigid_transform.npz; the two disagree on the
    name of the translation ('t' vs 'T'), so both are accepted. A missing scale
    means 1.0.
    """

    data = np.load(str(path), allow_pickle=False)
    keys = list(data.keys())

    if "R" not in keys:
        raise KeyError(f"{path} has no 'R' array (found {keys})")
    t_key = next((k for k in ("t", "T") if k in keys), None)
    if t_key is None:
        raise KeyError(f"{path} has no translation array 't'/'T' (found {keys})")

    R = np.asarray(data["R"], dtype=np.float64)
    t = np.asarray(data[t_key], dtype=np.float64).flatten()
    s = float(np.asarray(data["s"]).flatten()[0]) if "s" in keys else 1.0

    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError(f"{path}: expected R (3,3) and t (3,), got {R.shape} and {t.shape}")

    return R, t, s


def invert_similarity_transform(R, t, s):
    """Inverse of x' = s*R*x + t, returned in the same (R, t, s) form."""

    R_inv = np.asarray(R, dtype=np.float64).T
    s_inv = 1.0 / float(s)
    t_inv = -s_inv * (R_inv @ np.asarray(t, dtype=np.float64).flatten())

    return R_inv, t_inv, s_inv


def similarity_tf_dict(point_dict, R, t, s):
    aligned_point_dict = {}
    
    # Ensure t is (3,) then convert to column vector (3, 1)
    t_vec = np.array(t).flatten().reshape(3, 1) 
    
    for name, point in point_dict.items():
        # point is (3,), need (3, 1)
        pt = np.asarray(point).reshape(3, 1)
        
        # Apply transformation: P' = s*R*P + t
        # Ensure R is (3,3) and pt is (3,1)
        aligned_pt = (s * (R @ pt)) + t_vec
        aligned_point_dict[name] = aligned_pt.flatten()
        
    return aligned_point_dict