"""Shared landmark CSV <-> vtkPolyData conversion, used by every alignment
stage that reads Slicer-style landmark CSVs ('x', 'y', 'z', 'label' columns)."""

import os

import numpy as np
import pandas as pd
import vtk

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
