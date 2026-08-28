import vtk
from vtk.util import numpy_support
import numpy as np
import torch

def create_polydata(points, faces=None):
    """Creates a VTK PolyData object from points and optional faces."""

    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    polydata = vtk.vtkPolyData()
    
    # Set Points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(points.astype(np.float32)))
    polydata.SetPoints(vtk_points)
    
    # Set Faces
    if faces is not None:
        # Create a vtkIdTypeArray for cell connectivity
        # Format: [3, i1, i2, i3, 3, j1, j2, j3, ...]
        n_faces = faces.shape[0]
        # Prepend '3' (number of vertices per triangle) to every row
        cells_flat = np.hstack([np.ones((n_faces, 1), dtype=np.int64) * 3, faces.astype(np.int64)])
        
        # Convert to vtkIdTypeArray
        cell_array = vtk.vtkCellArray()
        # Newer VTK versions prefer SetCells with an offset array, 
        # but this conversion is the standard way to feed legacy-style data:
        cell_array.SetCells(n_faces, numpy_support.numpy_to_vtk(cells_flat.flatten(), deep=True, array_type=vtk.VTK_ID_TYPE))
        
        polydata.SetPolys(cell_array)
    else:
        # Fallback to points/vertices
        vertices = vtk.vtkCellArray()
        for i in range(len(points)):
            vertices.InsertNextCell(1, [i])
        polydata.SetVerts(vertices)
        
    return polydata


def create_polydata_w_lines(points, lines, verbose = False):
    """
    points: (N, 3) array
    lines: (M, 2) array of index pairs [start_idx, end_idx]
    """
    if verbose:
        print(f"creating poly with {points.shape} positions and {lines.shape} lines")
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
        if verbose:
            print(f"convert to {points.shape}")
    if isinstance(lines, torch.Tensor):
        lines = lines.detach().cpu().numpy()
        if verbose:
            print(f"convert to {lines.shape}")

    # Always expect (M, 2): one [start_idx, end_idx] pair per line. Reject the
    # transposed PyG edge_index (2, M) up front — otherwise VTK reinterprets node
    # indices as cell point-counts and reads out of bounds (random crashes).
    lines = np.asarray(lines)
    if lines.ndim != 2 or lines.shape[1] != 2:
        raise ValueError(
            f"create_polydata_w_lines expects lines of shape (M, 2), got {lines.shape}. "
            "Pass edges as [start, end] index pairs (transpose a (2, M) edge_index first)."
        )

    polydata = vtk.vtkPolyData()
    
    # 1. Set Points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(points.astype(np.float32)))
    polydata.SetPoints(vtk_points)
    
    # 2. Set Lines
    # VTK expects lines in the format [2, idx1, idx2, 2, idx3, idx4, ...]
    n_lines = lines.shape[0]
    lines_flat = np.hstack([np.ones((n_lines, 1), dtype=np.int64) * 2, lines.astype(np.int64)])
    
    cell_array = vtk.vtkCellArray()
    cell_array.SetCells(n_lines, numpy_support.numpy_to_vtk(lines_flat.flatten(), deep=True, array_type=vtk.VTK_ID_TYPE))
    
    polydata.SetLines(cell_array)
    return polydata


