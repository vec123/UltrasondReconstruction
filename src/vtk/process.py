
import numpy as np
import vtk

from src.vtk.create import create_polydata
from src.vtk.extract import extract_all_vtp_point_fields, extract_vtp_points_cells
from src.vtk.fields import add_point_fields


def filter_largest_component(poly_data, min_bridge_width=0, relative=False):
    """Extracts the largest connected component of `poly_data`.

    Args:
        min_bridge_width: NOT IMPLEMENTED. Callers pass this to first sever
            connections thinner than this width before taking the largest
            component (so a noise fragment joined to the main surface by a
            thin bridge doesn't count as "connected" to it). That erosion
            step needs a real algorithm this module doesn't have yet, so a
            nonzero value raises rather than silently falling back to plain
            largest-component extraction.
        relative: whether `min_bridge_width` is relative to mesh scale.
    """
    if min_bridge_width:
        raise NotImplementedError(
            "filter_largest_component(min_bridge_width=...) is not implemented - severing "
            "thin bridges before taking the largest connected component needs a real "
            "erosion algorithm that hasn't been ported into src/vtk yet. Call with the "
            "default min_bridge_width=0 for plain largest-connected-component extraction."
        )

    print(f"Filter largest component for mesh with {poly_data.GetNumberOfPoints()} points.")

    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(poly_data)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()

    largest_mesh = connectivity.GetOutput()

    print(f"largest component has {largest_mesh.GetNumberOfPoints()} points.")

    return largest_mesh


def extract_polydata_vertex_subset(poly_data, mask):
    """Keeps the points selected by `mask`, drops every face that references a
    removed point, and reindexes what's left. Point-data fields are carried
    over, subset the same way as the points.

    Args:
        poly_data: input vtkPolyData
        mask: boolean array of shape [N] (True = keep), or an integer array of
            point indices to keep
    """
    points, faces = extract_vtp_points_cells(poly_data)
    fields = extract_all_vtp_point_fields(poly_data)

    mask = np.asarray(mask)
    keep_idx = np.flatnonzero(mask) if mask.dtype == bool else np.unique(mask)

    old_to_new = np.full(points.shape[0], -1, dtype=np.int64)
    old_to_new[keep_idx] = np.arange(keep_idx.size)

    new_points = points[keep_idx]
    face_mask = np.all(old_to_new[faces] >= 0, axis=1)
    new_faces = old_to_new[faces[face_mask]]

    new_polydata = create_polydata(new_points, new_faces)

    new_fields = {name: arr[keep_idx] for name, arr in fields.items() if arr is not None}
    if new_fields:
        add_point_fields(new_polydata, new_fields)

    return new_polydata


def smooth_vtp(poly_data, iterations=30, pass_band=0.1, boundary_smoothing=True,
               feature_edge_smoothing=False):
    """Non-shrinking windowed-sinc smoothing (vtkWindowedSincPolyDataFilter).

    Moves vertex positions to reduce staircase/speckle noise without the
    shrinkage plain Laplacian smoothing would cause. Point count and
    connectivity are unchanged.
    """
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly_data)
    smoother.SetNumberOfIterations(iterations)
    smoother.SetPassBand(pass_band)
    smoother.SetBoundarySmoothing(boundary_smoothing)
    smoother.SetFeatureEdgeSmoothing(feature_edge_smoothing)
    smoother.Update()
    return smoother.GetOutput()


def clean_vtp(polydata, keep_largest_component=True):
    """
    Optionally isolates the single largest connected component, removes
    topological defects, and decimates the triangle density by the specified
    reduction factor.

    Args:
        polydata: input vtkPolyData
        keep_largest_component: if True (default), strips away every region
            except the largest connected one before cleaning topology. Set
            False to skip that step and clean the mesh as-is.
    """

    # 1. Initialize VTP Reader
    raw_mesh = polydata

    orig_verts = raw_mesh.GetNumberOfPoints()
    orig_cells = raw_mesh.GetNumberOfCells()
    print(f"Original Count : {orig_verts:,} vertices | {orig_cells:,} triangles")

    # 2. Isolate the absolute largest component (strips away tiny floating noise)
    to_clean = raw_mesh
    if keep_largest_component:
        print("Applying Connectiviy filter...")
        connectivity = vtk.vtkConnectivityFilter()
        connectivity.SetInputData(raw_mesh)
        connectivity.SetExtractionModeToLargestRegion()
        connectivity.Update()
        to_clean = connectivity.GetOutput()

    # 3. Clean up topology (merges coincident points, eliminates zero-area triangles)
    print("Cleaning mesh topology and removing degenerate geometry...")
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(to_clean)
    cleaner.Update()
    out = cleaner.GetOutput()
    return out

def reduce_vtp(polydata, reduction_factor=0.90):
    # 4. Decimate the mesh using Quadric Error Metrics
    # Keeps structural landmarks (edges, curves) while aggressively thinning flat zones.

    raw_mesh = polydata
    orig_verts = raw_mesh.GetNumberOfPoints()
    orig_cells = raw_mesh.GetNumberOfCells()
    print(f"original Count: {orig_verts:,} vertices | {orig_cells:,} triangles")

    print(f"Decimating triangles by {reduction_factor * 100:.1f}%...")
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(raw_mesh)
    decimate.SetTargetReduction(reduction_factor)
    decimate.Update()

    reduced_mesh = decimate.GetOutput()

    final_verts = reduced_mesh.GetNumberOfPoints()
    final_cells = reduced_mesh.GetNumberOfCells()

    print("-" * 60)
    print(f"Optimized Count: {final_verts:,} vertices | {final_cells:,} triangles")
    print(f"Data Reduction : {((orig_verts - final_verts) / orig_verts) * 100:.2f}% vertices removed.")
    print("-" * 60)

    return reduced_mesh
