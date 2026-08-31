"""Shared landmark CSV <-> vtkPolyData conversion, used by every alignment
stage that reads Slicer-style landmark CSVs ('x', 'y', 'z', 'label' columns)."""

import os

import numpy as np
import pandas as pd
import vtk
from vtk.util import numpy_support

from src.transforms.similarity import SimilarityTransform
from src.vtk.create import create_polydata
from src.vtk.io import save_vtp


def landmarks_csv_to_polydata(csv_path, vtp_path=None, lps_to_ras=False):
    """
    Reads a landmark CSV file, converts it into a vtkPolyData object, and
    attaches landmark labels as a point data array.

    Set lps_to_ras=True if the CSV (e.g. exported from 3D Slicer) is in LPS
    space while the surface meshes it's paired with are in RAS space.

    Returns (polydata, dataframe).
    """
    df = pd.read_csv(csv_path).copy()

    if lps_to_ras:
        df["x"] = -df["x"]
        df["y"] = -df["y"]

    points = df[["x", "y", "z"]].values.astype(np.float32)
    polydata = create_polydata(points)

    if "label" in df.columns:
        label_array = vtk.vtkStringArray()
        label_array.SetName("label")
        for label in df["label"].astype(str):
            label_array.InsertNextValue(label)
        polydata.GetPointData().AddArray(label_array)

    if vtp_path:
        os.makedirs(os.path.dirname(vtp_path), exist_ok=True)
        save_vtp(polydata, vtp_path)

    return polydata, df


def points_to_labeled_polydata(points, labels=None):
    """Builds a vtkPolyData point cloud with an optional 'label' string array."""
    polydata = create_polydata(np.asarray(points, dtype=np.float32))
    if labels is not None:
        label_array = vtk.vtkStringArray()
        label_array.SetName("label")
        for label in labels:
            label_array.InsertNextValue(str(label))
        polydata.GetPointData().AddArray(label_array)
    return polydata


def heatmap_landmark_vertex_indices(template_vtp, heatmap_prefix="heatmap_"):
    """Reads a template VTP's `{heatmap_prefix}{label}` point fields and returns
    {label: vertex_index}, the index of each field's peak (argmax) point - see
    align_to_template.py's extract_landmarks_from_heatmap_vtp, which does the
    same argmax but returns the peak's position instead of its index."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(template_vtp)
    reader.Update()
    point_data = reader.GetOutput().GetPointData()

    combined_field_name = f"{heatmap_prefix}landmarks"  # superposition of every other field, not itself a label
    indices = {}
    for i in range(point_data.GetNumberOfArrays()):
        name = point_data.GetArrayName(i)
        if name and name.startswith(heatmap_prefix) and name != combined_field_name:
            label = name[len(heatmap_prefix):]
            values = numpy_support.vtk_to_numpy(point_data.GetArray(name))
            indices[label] = int(np.argmax(values))

    if not indices:
        raise ValueError(f"No point fields starting with '{heatmap_prefix}' found in '{template_vtp}'.")
    return indices


def fit_landmark_alignment(vtp_points, template_vtp, landmark_path, lps_to_ras, with_scale, heatmap_prefix="heatmap_"):
    """Fits a similarity transform mapping `vtp_points`' own frame onto
    `landmark_path`'s frame, by matching labels between `template_vtp`'s
    heatmap-derived vertex indices (into `vtp_points`) and `landmark_path`'s
    labeled x,y,z positions."""
    template_indices = heatmap_landmark_vertex_indices(template_vtp, heatmap_prefix)
    _, subject_df = landmarks_csv_to_polydata(landmark_path, lps_to_ras=lps_to_ras)
    subject_df = subject_df.set_index("label")

    common_labels = [label for label in template_indices if label in subject_df.index]
    if len(common_labels) < 3:
        raise ValueError(
            f"Only {len(common_labels)} label(s) in common between '{template_vtp}' and "
            f"'{landmark_path}'; need at least 3 to fit a similarity transform."
        )

    vertex_indices = np.array([template_indices[label] for label in common_labels])
    src_pts = vtp_points[vertex_indices]
    dst_pts = subject_df.loc[common_labels, ["x", "y", "z"]].values.astype(float)

    return SimilarityTransform.from_correspondences(
        src_pts, dst_pts, with_scale=with_scale, stage="landmark_align"
    )
