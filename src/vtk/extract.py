
import numpy as np
import vtk
from vtk.util import numpy_support


def extract_vtp_points_cells(poly_data):
    """Read points ``[N, 3]`` and triangle faces ``[F, 3]`` from a vtkPolyData.

    Meshes that carry non-triangular polygons (a crop or a decimation can leave
    quads behind) are triangulated first, so the connectivity array is always a
    flat run of triples. vtkTriangleFilter only splits cells -- the point array
    and its point data are passed through untouched -- so face indices stay
    valid against the points returned here.
    """
    polys = poly_data.GetPolys()
    offsets = numpy_support.vtk_to_numpy(polys.GetOffsetsArray())
    if offsets.size and not np.all(np.diff(offsets) == 3):
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(poly_data)
        tri.PassVertsOff()
        tri.PassLinesOff()
        tri.Update()
        poly_data = tri.GetOutput()
        polys = poly_data.GetPolys()

    points = numpy_support.vtk_to_numpy(poly_data.GetPoints().GetData())
    cells = numpy_support.vtk_to_numpy(polys.GetConnectivityArray())
    return points, cells.reshape(-1, 3)


def extract_vtp_point_fields(poly_data, field_names):
    """Read named point-data arrays from a vtkPolyData as numpy arrays.

    Returns a dict mapping each requested name to its array -- shape ``[N]`` for a scalar
    field, ``[N, C]`` for a multi-component field (e.g. ``[N, 3]`` normals) -- or ``None``
    when the field is absent. Missing fields are expected (bare point clouds carry no
    area/normal), so callers must handle ``None`` (see ``load_dataset``'s ones/zeros
    fallback)."""
    point_data = poly_data.GetPointData()
    result = {}
    for name in field_names:
        arr = point_data.GetArray(name) if point_data is not None else None
        result[name] = numpy_support.vtk_to_numpy(arr) if arr is not None else None
    return result


def threshold_vtp_by_point_field(poly_data, field_name, value, relative=False, keep="below"):
    """Keeps only the points whose `field_name` value satisfies a threshold.

    Drops every face that references a removed point and reindexes what's
    left (see ``src.vtk.process.extract_polydata_vertex_subset``).

    Args:
        field_name: point-data array to threshold on
        value: absolute field value, or (if ``relative=True``) a fraction in
            ``[0, 1]`` of the field's own ``[min, max]`` range
        relative: interpret `value` relative to the field's own range instead
            of as an absolute value
        keep: ``"below"`` keeps points with field <= threshold; ``"above"``
            keeps field >= threshold

    Raises:
        ValueError: if `field_name` is absent, or `keep` is not recognized
    """
    from src.vtk.process import extract_polydata_vertex_subset  # deferred: process imports this module

    fields = extract_vtp_point_fields(poly_data, [field_name])
    values = fields[field_name]
    if values is None:
        raise ValueError(f"point field {field_name!r} not found on this mesh.")

    threshold = value
    if relative:
        lo, hi = float(values.min()), float(values.max())
        threshold = lo + value * (hi - lo)

    if keep == "below":
        mask = values <= threshold
    elif keep == "above":
        mask = values >= threshold
    else:
        raise ValueError(f"keep must be 'below' or 'above', got {keep!r}")

    return extract_polydata_vertex_subset(poly_data, mask)


def extract_all_vtp_point_fields(poly_data):
    """Read EVERY numeric point-data array from a vtkPolyData, as ``{name: array}``.

    The unnamed counterpart of ``extract_vtp_point_fields``: used when the arrays are
    to be carried along rather than consumed, so the caller cannot know their names.
    Shapes are as above -- ``[N]`` scalar, ``[N, C]`` multi-component -- and dtypes are
    whatever the file stored, so an int label array stays int rather than becoming a
    float that happens to hold whole numbers.

    Non-numeric arrays (string/variant) are skipped: ``GetArray`` returns None for
    them, and nothing downstream can carry them anyway.
    """
    point_data = poly_data.GetPointData()
    if point_data is None:
        return {}

    result = {}
    for i in range(point_data.GetNumberOfArrays()):
        arr = point_data.GetArray(i)
        if arr is None or not arr.GetName():
            continue
        result[arr.GetName()] = numpy_support.vtk_to_numpy(arr)
    return result
