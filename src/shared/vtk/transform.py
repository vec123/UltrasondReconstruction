
import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

# Point arrays that hold unit normals by this project's convention. A similarity
# only rotates them: they keep length 1 and are not scaled with the points.
NORMAL_ARRAYS = ("normal",)

def decompose_similarity_transform(M):
    """
    Splits a 4x4 similarity matrix into its scale, rotation and translation
    parts, such that the transform is `s * R @ x + t`.
    """
    A = np.asarray(M)[:3, :3]

    # Columns of s * R all have length s, since R is orthonormal.
    s = float(np.mean(np.linalg.norm(A, axis=0)))
    R = A / s
    t = np.asarray(M)[:3, 3].copy()

    return s, R, t


def transform_polydata(polydata, R, t, s=1.0):
    """Applies the similarity transform x' = s*R*x + t to a whole mesh (see transform_polydata_matrix)."""
    matrix = np.eye(4)
    matrix[:3, :3] = float(s) * np.asarray(R, dtype=float)
    matrix[:3, 3] = np.asarray(t, dtype=float).flatten()
    return transform_polydata_matrix(polydata, matrix)


def transform_polydata_matrix(polydata, matrix):
    """Applies a 4x4 similarity matrix (s * R, t) to a whole mesh, returning a new vtkPolyData.

    Unlike create_polydata(points), this keeps the topology and every data
    array. vtkTransformFilter moves the points, the active normals and
    the active vectors (by s * R, right for a displacement), but it copies any
    other array as it is. So every NORMAL_ARRAYS array gets R n from the input,
    whatever its attribute: not rotated as a plain array, and not scaled to
    length s as the active vectors (which add_point_field makes of it).
    """
    matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, matrix[i, j])

    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_matrix)

    filt = vtk.vtkTransformFilter()  # vtkTransformPolyDataFilter is deprecated since VTK 9.7
    filt.SetTransform(transform)
    filt.SetInputData(polydata)
    filt.Update()
    output = filt.GetPolyDataOutput()

    _, R, _ = decompose_similarity_transform(matrix)
    source_data, output_data = polydata.GetPointData(), output.GetPointData()
    active_vectors = source_data.GetVectors()
    for name in NORMAL_ARRAYS:
        source = source_data.GetArray(name)
        if source is None or source.GetNumberOfComponents() != 3:
            continue
        normals = vtk_to_numpy(source)
        rotated = numpy_to_vtk(np.ascontiguousarray((normals.astype(np.float64) @ R.T).astype(normals.dtype)),
                               deep=True)
        rotated.SetName(name)
        if active_vectors is not None and active_vectors.GetName() == name:
            output_data.SetVectors(rotated)
        else:
            output_data.AddArray(rotated)    # replaces the array of the same name

    return output