"""Loaders that put every shape model behind one interface.

Whatever the file format, a model reaches the rest of the project as the same
four things - ``mean`` (3N,), ``eigenfunctions`` (3N, K), ``eigenvalues`` (K,)
and a landmark dictionary - plus a triangulation, which is what
:class:`~src.model.model.StatisticalShapeModel` and ``ShapeSpace`` are built
from. :func:`load_shape_model` picks the backend by looking at the path:

* a ``.mat`` file  -> the BabyFM PCA model, read as-is (:func:`load_mat_ssm`)
* a directory      -> a Deformetrica atlas, shot and linearised into the same
                      form (:mod:`src.model.deformetrica`)
"""

from pathlib import Path

import scipy.io as sio
import numpy as np


def rescale_eigenvalues(eigenvalues, n_vertices):
    """Puts the .mat's eigenvalues in the same inner product as its eigenfunctions.

    The file ships the two in different conventions, which is measurable rather
    than a matter of opinion:

    * ``Phi^T Phi = I`` to 4e-15, so the stored basis is Euclidean unit-norm;
    * yet ``sum(lambda)`` read as the coefficient variance in *that* basis makes
      the whole training population deviate from the mean by an RMS of 2e-6 per
      vertex, against a face 0.013 across - 0.015%, i.e. a claim that every baby
      scanned was the same shape. The eigenvalues are internally consistent
      (``lambda_k / sum(lambda)`` matches ``pctVar_per_eigen`` to 1.8e-3), so it
      is the absolute scale that is off, not the spectrum.

    The two candidate inner products differ by exactly the vertex count: an
    eigenvector normalised under ``<x,y>_N = (1/N) sum x_i y_i`` has Euclidean
    norm ``sqrt(N)``, and since the displacement ``u * a`` is the same object as
    ``(u/sqrt(N)) * (a sqrt(N))``, storing the basis unit-norm moves a factor of
    ``sqrt(N)`` into the coefficient. Its variance is therefore ``N * lambda``.

    Three things corroborate the factor:

    1. the model's own MATLAB sampler declares ``scale_factor =
       sqrt(size(refShape, 2))`` - exactly ``sqrt(N)`` - in syntheric_babies.m;
    2. corrected, the population's RMS per-vertex displacement is 2.69% of the
       face's extent, against 2.79% for the fetus model, whose eigenvalues come
       from our own decomposition and are consistent by construction;
    3. the same sampler draws coefficients as ``lambda_k * U(-3, 3) * 1e6``
       rather than from ``sqrt(lambda_k)``, and that ad-hoc ``1e6`` puts its
       widest mode-0 sample at 2.56 corrected standard deviations - i.e. right
       where its own "-3..+3" comment intends.

    Only the .mat needs this. A Deformetrica atlas is decomposed in
    :mod:`src.model.deformetrica` in the Euclidean basis throughout, so its
    eigenvalues are already coefficient variances.
    """
    return np.asarray(eigenvalues, dtype=np.float64) * float(n_vertices)


def normalize_ssm(mean, eigenfunctions, eigenvalues):
    """Centres the model on the origin and rescales it to unit RMS radius.

    The mean is centred and scaled; the eigenfunctions describe displacements,
    so they are scaled but not translated; the eigenvalues are variances, so
    they scale quadratically (``Var(sX) = s^2 Var(X)``).
    """
    mean_reshaped = np.asarray(mean).reshape(-1, 3)
    centroid = mean_reshaped.mean(axis=0)
    centered_mean = mean_reshaped - centroid

    rms_dist = np.sqrt(np.mean(np.sum(centered_mean ** 2, axis=1)))
    scale = 1.0 / rms_dist

    return (
        (centered_mean * scale).flatten(),
        eigenfunctions * scale,
        eigenvalues * (scale ** 2),
    )


def load_shape_model(path, normalize=True, atlas_options=None, verbose=False):
    """Reads any supported shape model as (mean, eigenfunctions, eigenvalues, landmarks).

    ``atlas_options`` is a mapping of keyword arguments for the Deformetrica
    backend (kernel width, shooting steps, truncation); it is ignored for a
    .mat model, which has no such choices to make.
    """
    if is_atlas_path(path):
        from src.model.deformetrica import load_atlas_ssm

        return load_atlas_ssm(path, normalize=normalize, verbose=verbose,
                              **dict(atlas_options or {}))

    return load_mat_ssm(str(path), normalize=normalize)


def load_shape_model_triangulation(path):
    """The model's (F, 3) triangulation, whichever format it came from."""
    if is_atlas_path(path):
        from src.model.deformetrica import load_atlas_triangulation

        return load_atlas_triangulation(path)

    return load_mat_triangulation(str(path))


def is_atlas_path(path) -> bool:
    """True when ``path`` is a Deformetrica atlas rather than a .mat model."""
    return Path(path).is_dir()


def load_mat_ssm(path, normalize=True):
    """The BabyFM model as (mean, eigenfunctions, eigenvalues, landmarks).

    The eigenvalues are rescaled on the way out - see :func:`rescale_eigenvalues`
    - because the file stores them in a different inner product from the one its
    eigenfunctions are normalised in.
    """
    # --- LOAD DATA ---
    mat_data = sio.loadmat(path, squeeze_me=True)
    FaceModel = mat_data['BabyFaceModel']

    # Extract components
    refShape = FaceModel['refShape'].item()
    eigenValues = FaceModel['eigenValues'].item().astype(np.float64) # Ensure float64
    eigenFunctions = FaceModel['eigenFunctions'].item()
    meanDeformation = FaceModel['meanDeformation'].item()
    lmks_vertsIND = FaceModel['landmark_verts'].item() - 1
    landmark_names = [str(n).strip() for n in FaceModel['landmark_names'].item()]

    # Reconstruct mean: (3N,)
    shapeMU = refShape.flatten(order='F')
    mean = shapeMU + meanDeformation

    eigenValues = rescale_eigenvalues(eigenValues, n_vertices=refShape.shape[1])

    landmark_data = {"verts ": lmks_vertsIND, "names ": landmark_names}

    if normalize:
        mean, eigenFunctions, eigenValues = normalize_ssm(mean, eigenFunctions, eigenValues)

    return mean, eigenFunctions, eigenValues, landmark_data

def load_mat_triangulation(path):
    """Face topology of the shape model, as a (F, 3) 0-based index array.

    The .mat carries 'triang' as (3, F) 1-based MATLAB indices; this returns the
    transposed, 0-based form the VTK helpers expect. Every vertex-corresponding
    mesh - the model, an instance, the template - shares this triangulation, so
    it is what turns them from point clouds into surfaces, and it gives oriented
    vertex normals instead of the sign-arbitrary ones a point cloud yields.
    """
    mat_data = sio.loadmat(path, squeeze_me=True)
    FaceModel = mat_data['BabyFaceModel']

    if 'triang' not in (FaceModel.dtype.names or ()):
        return None

    triang = np.asarray(FaceModel['triang'].item())
    if triang.ndim != 2:
        raise ValueError(f"'triang' should be 2-D, got shape {triang.shape}")
    if triang.shape[0] == 3 and triang.shape[1] != 3:
        triang = triang.T

    return triang.astype(np.int64) - 1


def load_mat_ssm_old(path, normalize = True):
    # --- LOAD DATA ---
    mat_data = sio.loadmat(path, squeeze_me=True)
    FaceModel = mat_data['BabyFaceModel']

    refShape = FaceModel['refShape'].item()
    pctVar_per_eigen = FaceModel['pctVar_per_eigen'].item()
    eigenValues = FaceModel['eigenValues'].item()
    eigenFunctions = FaceModel['eigenFunctions'].item()
    triang = FaceModel['triang'].item()
    meanDeformation= FaceModel['meanDeformation'].item()
    lmks_vertsIND = FaceModel['landmark_verts'].item() - 1
    landmark_names = [str(n).strip() for n in FaceModel['landmark_names'].item()]
    
    #  Reconstruct full mean
    shapeMU = refShape.flatten(order='F')
    mean = shapeMU + meanDeformation

    landmark_data = {"verts ": lmks_vertsIND, "names ": landmark_names}

    if normalize:
        # Normalize the Mean
        mean_reshaped = mean.reshape(-1, 3)
        centroid = mean_reshaped.mean(axis=0)
        centered_mean = mean_reshaped - centroid
        
       
    
        # Calculate scale: Root Mean Square distance to centroid
        rms_dist = np.sqrt(np.mean(np.sum(centered_mean**2, axis=1)))
        scale = 1.0 / rms_dist
        
        normalized_mean = (centered_mean * scale).flatten()
        
        # 3. Apply normalization to Eigenfunctions
        # The eigenfunctions define shape change, so they must be scaled by the same factor
        # but do NOT get translated by the centroid.
        normalized_eigenfunctions = eigenFunctions * scale
        normalized_eigenvalues = eigenValues * (scale ** 2)

        mean = normalized_mean
        eigenFunctions = normalized_eigenfunctions
        eigenValues = normalized_eigenvalues

    return mean, eigenFunctions, eigenValues, landmark_data #centroid, scale

""" 
def load_mat_ssm_(path):

    # --- LOAD DATA ---
    mat_data = sio.loadmat(path, squeeze_me=True)
    FaceModel = mat_data['BabyFaceModel']

    refShape = FaceModel['refShape'].item()
    pctVar_per_eigen = FaceModel['pctVar_per_eigen'].item()
    eigenValues = FaceModel['eigenValues'].item()
    eigenFunctions = FaceModel['eigenFunctions'].item()
    triang = FaceModel['triang'].item()
    meanDeformation= FaceModel['meanDeformation'].item()
    lmks_vertsIND = FaceModel['landmark_verts'].item() - 1
    landmark_names = [str(n).strip() for n in FaceModel['landmark_names'].item()]

    landmark_data = {"verts ": lmks_vertsIND,
                 "names ": landmark_names}
    
    shapeMU = refShape.flatten(order='F')
    mean = shapeMU + meanDeformation

    return mean, eigenFunctions, eigenValues, landmark_data
"""