
import os

import vtk

from src.vtk.io import load_vtp, save_vtp


def _read_legacy_vtk(vtk_path):
    """Read a legacy VTK PolyData file (.vtk) and return its vtkPolyData."""
    if not os.path.exists(vtk_path):
        raise FileNotFoundError(f"File not found: {vtk_path}")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_path)
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise ValueError(
            f"{vtk_path} did not read as non-empty PolyData. If it is a legacy grid "
            "(unstructured/structured), read it and use convert_vtu_to_vtp_vtk first."
        )
    return polydata


def _write_legacy_vtk(polydata, vtk_path, binary=True):
    """Write a vtkPolyData to a legacy VTK PolyData file (.vtk)."""
    os.makedirs(os.path.dirname(os.path.abspath(vtk_path)), exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(vtk_path)
    writer.SetInputData(polydata)
    if binary:
        writer.SetFileTypeToBinary()
    else:
        writer.SetFileTypeToASCII()
    # Write() returns 0 on failure WITHOUT raising, so an ignored return turns a failed
    # write into a silent no-op. Raise so callers can't miss it (mirrors save_vtp).
    if writer.Write() != 1:
        raise IOError(f"vtkPolyDataWriter failed to write {vtk_path}")
    return vtk_path


def vtk_to_vtp(src_path, dst_path=None, binary=True):
    """Convert a legacy VTK PolyData file (.vtk) to an XML PolyData file (.vtp).

    ``dst_path`` defaults to ``src_path`` with its extension swapped to ``.vtp``.
    Returns the path written."""
    if dst_path is None:
        dst_path = os.path.splitext(src_path)[0] + ".vtp"
    polydata = _read_legacy_vtk(src_path)
    save_vtp(polydata, dst_path, binary=binary)
    return dst_path


def vtp_to_vtk(src_path, dst_path=None, binary=True):
    """Convert an XML PolyData file (.vtp) to a legacy VTK PolyData file (.vtk).

    ``dst_path`` defaults to ``src_path`` with its extension swapped to ``.vtk``.
    Returns the path written."""
    if dst_path is None:
        dst_path = os.path.splitext(src_path)[0] + ".vtk"
    polydata = load_vtp(src_path)
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise ValueError(f"{src_path} read as empty PolyData; nothing to convert.")
    _write_legacy_vtk(polydata, dst_path, binary=binary)
    return dst_path


def convert_vtu_to_vtp_vtk(vtu_data):
    """
    Converts a VTU object to VTP, with internal checks to ensure 
    valid input and non-empty output.
    """
    # 1. Check if the input is valid
    if vtu_data is None:
        raise ValueError("Input vtu_data is None.")
        
    print(f"Input object type: {vtu_data.GetClassName()}")
    print(f"Input points: {vtu_data.GetNumberOfPoints()}")
    
    if vtu_data.GetNumberOfPoints() == 0:
        raise ValueError("Input VTU has 0 points. Conversion aborted.")

    # 2. Setup the filter
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(vtu_data)
    surface_filter.Update()
    
    # 3. Retrieve the output
    vtp_output = surface_filter.GetOutput()
    
    # 4. Check if the filter actually produced something
    if vtp_output is None or vtp_output.GetNumberOfPoints() == 0:
        raise RuntimeError("Surface extraction failed: The resulting VTP is empty.")
    
    print(f"Conversion successful. Output points: {vtp_output.GetNumberOfPoints()}")
    
    return vtp_output
