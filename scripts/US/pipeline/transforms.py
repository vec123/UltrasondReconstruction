"""Rigid/similarity transform tracking shared by every alignment stage.

Every alignment stage in the pipeline (align_to_reference, align_to_template,
...) moves each subject's mesh into a new coordinate frame. To be able to map
a fully-processed result back onto the subject's original scan, each stage
saves the transform it applied, PER SUBJECT, as a `RigidTransform`.

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

from src.geometry import (
    decompose_similarity_transform,
    umeyama_similarity_transform,
)


class RigidTransform:
    """A 4x4 homogeneous similarity transform `s * R @ x + t`, with provenance."""

    def __init__(self, matrix, stage="", with_scale=True):
        self.matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        self.stage = stage
        self.with_scale = with_scale

    @classmethod
    def identity(cls, stage="", with_scale=True):
        return cls(np.eye(4), stage=stage, with_scale=with_scale)

    @classmethod
    def from_correspondences(cls, src_pts, dst_pts, with_scale=True, stage=""):
        """Fits a similarity transform mapping src_pts onto dst_pts (Procrustes)."""
        M = umeyama_similarity_transform(src_pts, dst_pts, with_scale=with_scale)
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
        return RigidTransform(
            np.linalg.inv(self.matrix), stage=f"inverse({self.stage})", with_scale=self.with_scale
        )

    def compose(self, other):
        """Returns the transform 'apply `other`, then apply `self`'."""
        return RigidTransform(
            self.matrix @ other.matrix, stage=self.stage, with_scale=self.with_scale and other.with_scale
        )

    def apply_points(self, points):
        """Applies the transform to an (N, 3) array of points."""
        points = np.asarray(points, dtype=np.float64)
        homo = np.hstack([points, np.ones((len(points), 1))])
        return (self.matrix @ homo.T).T[:, :3]

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

    def save(self, path, labels=None):
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
        np.savez(path, **kwargs)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            with_scale = bool(data["with_scale"]) if "with_scale" in data else True
            stage = str(data["stage"]) if "stage" in data else ""
            return cls(data["M"], stage=stage, with_scale=with_scale)


def subject_transform_path(transforms_dir, subject_id):
    return os.path.join(transforms_dir, f"{subject_id}.npz")


def save_subject_transform(transforms_dir, subject_id, transform, labels=None):
    transform.save(subject_transform_path(transforms_dir, subject_id), labels=labels)


def load_subject_transform(transforms_dir, subject_id):
    return RigidTransform.load(subject_transform_path(transforms_dir, subject_id))


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

    previous = RigidTransform.load(path)
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

    scales = [RigidTransform.load(f).scale for f in tf_files]

    scale = float(np.median(scales))
    if len(scales) > 1:
        print(
            f"Median similarity scale over {len(scales)} transforms: {scale:.4f} "
            f"(min {min(scales):.4f}, max {max(scales):.4f})"
        )
    else:
        print(f"Similarity scale from {os.path.basename(tf_files[0])}: {scale:.4f}")

    return scale
