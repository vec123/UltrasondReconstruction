"""Stage: NRRD label volumes -> surface meshes in physical RAS world space.

Alternate entry point to the pipeline for segmentation-mask volumes (as
opposed to nrrd_to_vtp's raw grayscale ultrasound volumes): extracts an
isosurface at a fixed label value directly in voxel-index space, then
transforms it through the volume's spacing/origin/direction matrix and an
LPS -> RAS flip into physical world coordinates.
"""

import dataclasses
import glob
import os

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support


@dataclasses.dataclass
class Config:
    input_dir: str = ""
    output_dir: str = ""

    contour_value: float = 1.0  # label index/threshold for surface extraction
    reduction_factor: float = 0.3  # quadric decimation factor (0 = none, 0.5 = 50% reduction)


def sitk_to_vtk_image(sitk_img):
    """Converts a SimpleITK image to vtkImageData in voxel index space."""
    array = sitk.GetArrayFromImage(sitk_img)  # Shape: (Z, Y, X)

    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(sitk_img.GetSize())  # (X, Y, Z)

    vtk_array = numpy_support.numpy_to_vtk(num_array=array.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_img.GetPointData().SetScalars(vtk_array)
    return vtk_img


def transform_vtp_to_physical_ras(polydata, sitk_img):
    """
    Transforms VTK PolyData from voxel index space [0..Nx, 0..Ny, 0..Nz]
    to physical RAS world coordinates using the NRRD's spacing, origin, and
    direction matrix.
    """
    spacing = np.array(sitk_img.GetSpacing())
    origin = np.array(sitk_img.GetOrigin())
    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)

    # Voxel Index -> Physical LPS Matrix
    M_voxel_to_lps = np.eye(4)
    M_voxel_to_lps[:3, :3] = direction @ np.diag(spacing)
    M_voxel_to_lps[:3, 3] = origin

    # LPS -> RAS Conversion Matrix (flip X and Y)
    M_lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])

    M_total = M_lps_to_ras @ M_voxel_to_lps

    vtk_mat = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            vtk_mat.SetElement(r, c, M_total[r, c])

    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_mat)

    tf_filter = vtk.vtkTransformPolyDataFilter()
    tf_filter.SetInputData(polydata)
    tf_filter.SetTransform(transform)
    tf_filter.Update()

    return tf_filter.GetOutput()


def extract_and_clean_surface(sitk_img, contour_value, decimation):
    """Extracts isosurface, cleans topology, and optionally decimates mesh."""
    vtk_img = sitk_to_vtk_image(sitk_img)

    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(vtk_img)
    contour.SetValue(0, contour_value)
    contour.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(contour.GetOutput())
    cleaner.Update()

    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(cleaner.GetOutput())
    conn.SetExtractionModeToLargestRegion()
    conn.Update()

    mesh = conn.GetOutput()

    if decimation > 0.0:
        decimate = vtk.vtkQuadricDecimation()
        decimate.SetInputData(mesh)
        decimate.SetTargetReduction(decimation)
        decimate.Update()
        mesh = decimate.GetOutput()

    return mesh


def save_vtp(polydata, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filepath)
    writer.SetInputData(polydata)
    writer.Write()


def run(cfg: Config):
    nrrd_files = glob.glob(os.path.join(cfg.input_dir, "*.nrrd"))
    print(f"Found {len(nrrd_files)} NRRD files in '{cfg.input_dir}'.")

    for nrrd_path in nrrd_files:
        base_name = os.path.splitext(os.path.basename(nrrd_path))[0]
        out_vtp_path = os.path.join(cfg.output_dir, f"{base_name}.vtp")

        sitk_img = sitk.ReadImage(nrrd_path)
        raw_mesh = extract_and_clean_surface(sitk_img, cfg.contour_value, cfg.reduction_factor)
        world_ras_mesh = transform_vtp_to_physical_ras(raw_mesh, sitk_img)

        save_vtp(world_ras_mesh, out_vtp_path)
        print(f"Saved Physical RAS VTP: {out_vtp_path}")
