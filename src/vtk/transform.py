
import vtk
import numpy as np

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
    """Applies the similarity transform x' = s*R*x + t to a whole mesh.

    Unlike create_polydata(points), this keeps the topology and every data
    array; normals and vectors are rotated along with the points.
    """
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    t = np.asarray(t, dtype=float).flatten()
    for i in range(3):
        for j in range(3):
            matrix.SetElement(i, j, float(s) * float(R[i][j]))
        matrix.SetElement(i, 3, float(t[i]))

    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)

    filt = vtk.vtkTransformPolyDataFilter()
    filt.SetTransform(transform)
    filt.SetInputData(polydata)
    filt.Update()

    return filt.GetOutput()