
import vtk

def filter_vtp_largest_component(poly_data):
    """Loads a VTP, extracts only the largest connected component, and saves it."""

    print(f"Filter largest component for mesh with {poly_data.GetNumberOfPoints()} points.")

    # 2. Extract the largest connected component
    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(poly_data)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()

    largest_mesh = connectivity.GetOutput()

    print(f"largest component has {largest_mesh.GetNumberOfPoints()} points.")

    return largest_mesh


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
