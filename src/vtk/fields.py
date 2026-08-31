
from vtk.util import numpy_support
import torch
import numpy as np
import vtk

def add_point_field(polydata, field_data, field_name="field", verbose = False):
    """Adds a scalar or vector field to the existing PolyData."""
    
    if verbose:
        print(f"adding field {field_data.shape}")
    if isinstance(field_data, torch.Tensor):
        field_data = field_data.detach().cpu().numpy()
        if verbose:
            print(f"convert to {field_data.shape}")

    vtk_array = _as_vtk_array(field_data, field_name)
    _attach(polydata, vtk_array, field_name)

    if vtk_array.GetNumberOfComponents() == 3:
        polydata.GetPointData().SetActiveVectors(field_name)
    elif vtk_array.GetNumberOfComponents() == 1:
        polydata.GetPointData().SetActiveScalars(field_name)

    return polydata

def add_cell_field(polydata, field_data, field_name="field"):
    """Adds a scalar or vector field to the existing PolyData's cell data."""
    vtk_array = numpy_support.numpy_to_vtk(field_data, deep=True)
    vtk_array.SetName(field_name)
    polydata.GetCellData().AddArray(vtk_array)
    return polydata

def add_point_fields(polydata, fields):
    """Attach many point arrays at once, from a ``{name: array}`` mapping.

    The bulk counterpart of ``add_point_field``, and it deliberately does NOT set the
    active scalars/vectors. That is right for one named field the caller just computed
    and wrong for a copy of somebody else's thirty: whichever array happened to be
    added last would silently become what ParaView colours by.

    Args:
        polydata: vtkPolyData to add to, modified in place
        fields: {name: array} with arrays of shape [N] or [N, C]; N is NOT checked
            against the point count here -- see io.export_shape, which drops the
            arrays that no longer address these points

    Returns:
        The same polydata
    """
    for name, values in fields.items():
        _attach(polydata, _as_vtk_array(values, name), name)
    return polydata


def _as_vtk_array(field_data, field_name):
    """A named vtkDataArray from a torch tensor or numpy array of shape [N] or [N, C]."""
    if isinstance(field_data, torch.Tensor):
        field_data = field_data.detach().cpu().numpy()

    # Ensure numpy array and correct shape
    field_data = np.ascontiguousarray(field_data)
    if field_data.ndim == 1:
        vtk_array = numpy_support.numpy_to_vtk(field_data, deep=True)
        vtk_array.SetNumberOfComponents(1)
    elif field_data.ndim == 2:
        vtk_array = numpy_support.numpy_to_vtk(field_data, deep=True)
        vtk_array.SetNumberOfComponents(field_data.shape[1])
    else:
        raise ValueError(f"Unsupported field_data ndim={field_data.ndim}; expected 1 or 2.")

    vtk_array.SetName(field_name)
    return vtk_array


def _attach(polydata, vtk_array, field_name):
    """Add the array, replacing any existing one of the same name rather than duplicating."""
    existing = polydata.GetPointData().GetArray(field_name)
    if existing is not None:
        polydata.GetPointData().RemoveArray(field_name)

    polydata.GetPointData().AddArray(vtk_array)


def copy_point_data(src_polydata, dst_polydata):
    """Copies every point-data array from src onto dst, by vertex index.

    There is no correspondence search here, just an index-for-index copy -- src
    and dst must already share the same point count and ordering. Used to carry
    fields a reconstruction has no notion of (e.g. a template's heatmap_<label>
    landmark fields) onto output built from that same template's vertex
    ordering, such as an SSM fit's reconstructed mesh.

    Args:
        src_polydata: vtkPolyData to copy point-data arrays from
        dst_polydata: vtkPolyData to copy them onto, modified in place

    Returns:
        The same dst_polydata

    Raises:
        ValueError: src and dst have different point counts
    """
    if src_polydata.GetNumberOfPoints() != dst_polydata.GetNumberOfPoints():
        raise ValueError(
            f"copy_point_data: src has {src_polydata.GetNumberOfPoints()} points, dst has "
            f"{dst_polydata.GetNumberOfPoints()} -- they must match to copy by index"
        )

    src_pd = src_polydata.GetPointData()
    for i in range(src_pd.GetNumberOfArrays()):
        _attach(dst_polydata, src_pd.GetArray(i), src_pd.GetArrayName(i))
    return dst_polydata

