"""Rigid/similarity transform tracking shared by every alignment stage.

Every alignment stage in the pipeline (align_to_reference, align_to_template,
...) moves each subject's mesh into a new coordinate frame. To be able to map
a fully-processed result back onto the subject's original scan, each stage
saves the transform it applied, PER SUBJECT, as a `SimilarityTransform`.

Transforms are chained via `previous_transforms_dir`: a stage that is handed
the directory of transforms written by the stage before it composes its own
newly computed transform with the previous one before saving, so every saved
transform is always CUMULATIVE - it maps a point from the very first
(pre-alignment) mesh frame directly into that stage's output frame. Reversing
alignment is then always just: load the last stage's transform for a subject
and invert it.
"""

import glob
import os

import numpy as np
import vtk

from src.vtk.transform import decompose_similarity_transform


def umeyama_similarity_transform(src_pts, dst_pts, with_scale=True, weights=None):
    """
    Computes the optimal similarity transformation (s * R * x + t) mapping
    src_pts onto dst_pts using Umeyama's algorithm / Procrustes analysis.

    Parameters
    ----------
    src_pts, dst_pts : (N, 3) arrays of corresponding points.
    with_scale : if False, the transform is rigid (s == 1.0).
    weights : optional (N,) array of non-negative per-correspondence weights;
        None weights every correspondence equally (the classic Umeyama fit).

    Returns
    -------
    M : (4, 4) homogeneous transformation matrix.
    """
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    w = np.ones(len(src_pts)) if weights is None else np.asarray(weights, dtype=np.float64)
    total = w.sum()

    src_mean = (w[:, None] * src_pts).sum(axis=0) / total
    dst_mean = (w[:, None] * dst_pts).sum(axis=0) / total

    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean

    # Weighted covariance matrix
    H = (w[:, None] * src_centered).T @ dst_centered / total
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
        src_var = (w * (src_centered ** 2).sum(axis=1)).sum() / total
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


class SimilarityTransform:
    """A 4x4 homogeneous similarity transform `s * R @ x + t`, with provenance."""

    def __init__(self, matrix, stage="", with_scale=True):
        self.matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        self.stage = stage
        self.with_scale = with_scale

    @classmethod
    def identity(cls, stage="", with_scale=True):
        return cls(np.eye(4), stage=stage, with_scale=with_scale)

    @classmethod
    def from_correspondences(cls, src_pts, dst_pts, with_scale=True, weights=None, stage=""):
        """Fits a similarity transform mapping src_pts onto dst_pts (Procrustes).

        `weights` optionally gives each correspondence a non-negative weight
        (e.g. for a weighted ICP); omitted, every correspondence counts equally.
        """
        M = umeyama_similarity_transform(src_pts, dst_pts, with_scale=with_scale, weights=weights)
        return cls(M, stage=stage, with_scale=with_scale)

    @property
    def scale(self):
        s, _, _ = decompose_similarity_transform(self.matrix)
        return s

    @property
    def rotation(self):
        _, R, _ = decompose_similarity_transform(self.matrix)
        return R

    @property
    def translation(self):
        return self.matrix[:3, 3].copy()

    def inverse(self):
        """Returns the transform that undoes this one."""
        return SimilarityTransform(
            np.linalg.inv(self.matrix), stage=f"inverse({self.stage})", with_scale=self.with_scale
        )

    def compose(self, other):
        """Returns the transform 'apply `other`, then apply `self`'."""
        return SimilarityTransform(
            self.matrix @ other.matrix, stage=self.stage, with_scale=self.with_scale and other.with_scale
        )

    def apply_points(self, points):
        """Applies the transform to an (N, 3) array of points."""
        points = np.asarray(points, dtype=np.float64)
        homo = np.hstack([points, np.ones((len(points), 1))])
        return (self.matrix @ homo.T).T[:, :3]

    def apply_points_dict(self, point_dict):
        """Applies the transform to a {name: (3,) point} dict, returning a new dict."""
        names = list(point_dict)
        transformed = self.apply_points(np.array([point_dict[name] for name in names]))
        return dict(zip(names, transformed))

    def apply_polydata(self, polydata):
        """Applies the transform to a vtkPolyData, returning a new vtkPolyData."""
        vtk_matrix = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_matrix.SetElement(i, j, self.matrix[i, j])

        transform = vtk.vtkTransform()
        transform.SetMatrix(vtk_matrix)

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputData(polydata)
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        return transform_filter.GetOutput()

    def save(self, path, labels=None, **extra):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        s, R, t = decompose_similarity_transform(self.matrix)
        kwargs = dict(
            M=self.matrix,
            R=R,
            t=t,
            s=s,
            with_scale=self.with_scale,
            stage=self.stage,
        )
        if labels is not None:
            kwargs["labels"] = np.asarray(labels, dtype=str)
        kwargs.update(extra)
        np.savez(path, **kwargs)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            with_scale = bool(data["with_scale"]) if "with_scale" in data else True
            stage = str(data["stage"]) if "stage" in data else ""

            if "M" in data:
                return cls(data["M"], stage=stage, with_scale=with_scale)

            # Older files (e.g. hand-rolled np.savez calls elsewhere) store the
            # transform as loose R/t/s instead of a 4x4 matrix, and sometimes
            # spell the translation key 'T' instead of 't'.
            if "R" not in data:
                raise KeyError(f"{path} has no 'M' or 'R' array (found {list(data.keys())})")
            t_key = next((k for k in ("t", "T") if k in data), None)
            if t_key is None:
                raise KeyError(f"{path} has no translation array 't'/'T' (found {list(data.keys())})")

            R = np.asarray(data["R"], dtype=np.float64)
            t = np.asarray(data[t_key], dtype=np.float64).flatten()
            s = float(np.asarray(data["s"]).flatten()[0]) if "s" in data else 1.0

            M = np.eye(4)
            M[:3, :3] = s * R
            M[:3, 3] = t
            return cls(M, stage=stage, with_scale=with_scale)


def subject_transform_path(transforms_dir, subject_id):
    return os.path.join(transforms_dir, f"{subject_id}.npz")


def save_subject_transform(transforms_dir, subject_id, transform, labels=None):
    transform.save(subject_transform_path(transforms_dir, subject_id), labels=labels)


def load_subject_transform(transforms_dir, subject_id):
    return SimilarityTransform.load(subject_transform_path(transforms_dir, subject_id))


def compose_with_previous(new_transform, subject_id, previous_transforms_dir):
    """
    Composes `new_transform` (this stage's own, local transform for
    `subject_id`) with the cumulative transform saved by the previous stage,
    if any. Returns a transform that maps directly from the very first
    (pre-alignment) frame into this stage's output frame.
    """
    if not previous_transforms_dir:
        return new_transform

    path = subject_transform_path(previous_transforms_dir, subject_id)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No previous transform found for subject '{subject_id}' at '{path}'."
        )

    previous = SimilarityTransform.load(path)
    return new_transform.compose(previous)


def median_scale(transforms_dir):
    """
    Reads the scale factor of every similarity transform saved in
    `transforms_dir` and returns the median. Used to convert a physical
    distance (e.g. a sigma in mm) into the units of an aligned/scaled space.
    """
    tf_files = sorted(glob.glob(os.path.join(transforms_dir, "*.npz")))
    if not tf_files:
        raise FileNotFoundError(f"No .npz transforms found in '{transforms_dir}'.")

    scales = [SimilarityTransform.load(f).scale for f in tf_files]

    scale = float(np.median(scales))
    if len(scales) > 1:
        print(
            f"Median similarity scale over {len(scales)} transforms: {scale:.4f} "
            f"(min {min(scales):.4f}, max {max(scales):.4f})"
        )
    else:
        print(f"Similarity scale from {os.path.basename(tf_files[0])}: {scale:.4f}")

    return scale
