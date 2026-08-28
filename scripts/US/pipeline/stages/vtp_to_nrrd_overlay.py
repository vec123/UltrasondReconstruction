"""Stage: paint a VTP point cloud onto its reference NRRD volume as an RGB overlay.

Diagnostic tool: rasterizes the points of a VTP file into the voxel grid of a
reference NRRD (with a thickness dilation) and writes an RGB copy of that
volume with the points painted in `line_color`.

Optionally maps the VTP through the INVERSE of a subject's saved pipeline
transform first (see pipeline/transforms.py), so a fully aligned/denoised
downstream result can be overlaid on the subject's ORIGINAL, un-aligned scan
for visual QC.

Optionally (and independently) first fits a fresh similarity transform that
lands the VTP directly in the reference NRRD's own world frame, using a set
of labeled landmarks - see `landmark_align` / `fit_landmark_alignment`.

Any number of VTPs can be painted onto the same output, each in its own color -
`layers` lists them, all at equal footing (e.g. the reconstruction in red and
the original segmented mesh in green, for visual comparison).

Optionally also writes one binary label-map NRRD + matching 2-entry Slicer
color table (.ctbl) PER layer, into `label_maps_dir` - each is its own file,
so overlapping layers don't collide the way a single shared label map would
(one voxel can only hold one value, so wherever two layers' lines coincide,
whichever layer is later in `layers` silently overwrites the earlier one's
voxels there). Import each pair as its own Labelmap Volume + color table,
then import all of them into the SAME Segmentation node (Segmentations
module's Import/Export tab, one Import per pair) to get independently
toggleable segments that can freely overlap - see `_write_layer_label_map`.
"""

import os

import numpy as np
import SimpleITK as sitk
import vtk
from scipy.ndimage import binary_dilation
from vtk.util import numpy_support

from src.config.US import VtpToNrrdOverlayConfig as Config

from ..landmarks import landmarks_csv_to_polydata
from ..transforms import RigidTransform, load_subject_transform


def _load_vtp_points(vtp_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    poly_data = reader.GetOutput()

    vtk_pts = poly_data.GetPoints()
    if vtk_pts is None or vtk_pts.GetNumberOfPoints() == 0:
        raise ValueError(f"No points found in VTP file: {vtp_path}")

    return numpy_support.vtk_to_numpy(vtk_pts.GetData())


def _template_landmark_vertex_indices(template_vtp, heatmap_prefix):
    """Reads a template VTP's `{heatmap_prefix}{label}` point fields and returns
    {label: vertex_index}, the index of each field's peak (argmax) point - see
    align_to_template.py's extract_landmarks_from_heatmap_vtp, which does the
    same argmax but returns the peak's position instead of its index."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(template_vtp)
    reader.Update()
    point_data = reader.GetOutput().GetPointData()

    combined_field_name = f"{heatmap_prefix}landmarks"  # superposition of every other field, not itself a label
    indices = {}
    for i in range(point_data.GetNumberOfArrays()):
        name = point_data.GetArrayName(i)
        if name and name.startswith(heatmap_prefix) and name != combined_field_name:
            label = name[len(heatmap_prefix):]
            values = numpy_support.vtk_to_numpy(point_data.GetArray(name))
            indices[label] = int(np.argmax(values))

    if not indices:
        raise ValueError(f"No point fields starting with '{heatmap_prefix}' found in '{template_vtp}'.")
    return indices


def fit_landmark_alignment(vtp_points, template_vtp, landmark_path, lps_to_ras, with_scale, heatmap_prefix="heatmap_"):
    """Fits a similarity transform mapping `vtp_points`' own frame onto
    `landmark_path`'s frame, by matching labels between `template_vtp`'s
    heatmap-derived vertex indices (into `vtp_points`) and `landmark_path`'s
    labeled x,y,z positions."""
    template_indices = _template_landmark_vertex_indices(template_vtp, heatmap_prefix)
    _, subject_df = landmarks_csv_to_polydata(landmark_path, lps_to_ras=lps_to_ras)
    subject_df = subject_df.set_index("label")

    common_labels = [label for label in template_indices if label in subject_df.index]
    if len(common_labels) < 3:
        raise ValueError(
            f"Only {len(common_labels)} label(s) in common between '{template_vtp}' and "
            f"'{landmark_path}'; need at least 3 to fit a similarity transform."
        )

    vertex_indices = np.array([template_indices[label] for label in common_labels])
    src_pts = vtp_points[vertex_indices]
    dst_pts = subject_df.loc[common_labels, ["x", "y", "z"]].values.astype(float)

    return RigidTransform.from_correspondences(
        src_pts, dst_pts, with_scale=with_scale, stage="vtp_to_nrrd_overlay_landmark_align"
    )


def _rasterize_layer(vtp_path, thickness, dims, spacing, origin, dir_mat, transform_cfg=None, landmark_align_cfg=None):
    """Loads one VTP's points, optionally aligns them, and returns their
    dilated voxel mask (Z, Y, X order) against a reference volume's grid."""
    pts_world = _load_vtp_points(vtp_path)

    if landmark_align_cfg and landmark_align_cfg.landmark_path:
        alignment = fit_landmark_alignment(
            pts_world,
            template_vtp=landmark_align_cfg.template_vtp,
            landmark_path=landmark_align_cfg.landmark_path,
            lps_to_ras=landmark_align_cfg.lps_to_ras,
            with_scale=landmark_align_cfg.with_scale,
            heatmap_prefix=landmark_align_cfg.heatmap_prefix,
        )
        pts_world = alignment.apply_points(pts_world)
        print(f"Aligned VTP to '{landmark_align_cfg.landmark_path}' landmarks (scale={alignment.scale:.4f}).")

    if transform_cfg and transform_cfg.apply_inverse:
        transform = load_subject_transform(transform_cfg.manifest_dir, transform_cfg.subject_id)
        pts_world = transform.inverse().apply_points(pts_world)
        print(f"Applied inverse of '{transform.stage}' transform for subject '{transform_cfg.subject_id}'.")

    # Transform world coordinates -> voxel indices.
    inv_dir = np.linalg.inv(dir_mat)
    pts_shifted = pts_world - origin
    pts_voxel = np.dot(pts_shifted, inv_dir.T) / spacing
    indices = np.round(pts_voxel).astype(int)

    valid_mask = (
        (indices[:, 0] >= 0) & (indices[:, 0] < dims[0]) &
        (indices[:, 1] >= 0) & (indices[:, 1] < dims[1]) &
        (indices[:, 2] >= 0) & (indices[:, 2] < dims[2])
    )
    valid_indices = indices[valid_mask]

    # Create 3D boolean mask (Z, Y, X order).
    mask_3d = np.zeros((dims[2], dims[1], dims[0]), dtype=bool)
    mask_3d[valid_indices[:, 2], valid_indices[:, 1], valid_indices[:, 0]] = True

    # Apply 3D spherical dilation for thickness.
    if thickness > 0:
        r_vox = np.ceil(thickness / spacing).astype(int)
        lz = np.arange(-r_vox[2], r_vox[2] + 1) * spacing[2]
        ly = np.arange(-r_vox[1], r_vox[1] + 1) * spacing[1]
        lx = np.arange(-r_vox[0], r_vox[0] + 1) * spacing[0]
        Z, Y, X = np.meshgrid(lz, ly, lx, indexing="ij")

        sphere_kernel = (X**2 + Y**2 + Z**2) <= (thickness**2)
        mask_3d = binary_dilation(mask_3d, structure=sphere_kernel)

    return mask_3d


def _layer_name(layer):
    name = layer.name or os.path.splitext(os.path.basename(layer.vtp_path))[0]
    return name.replace(" ", "_")  # Slicer's .ctbl format is whitespace-delimited


def _write_layer_label_map(label_maps_dir, layer, mask_3d, ref_sitk):
    """Writes one layer's own binary (0/1) label-map NRRD plus a matching
    2-entry Slicer color table (.ctbl), named after the layer - a separate
    file per layer so overlapping layers never collide (each is a whole,
    independent segment on import, unlike a single shared multi-label map,
    where one voxel can only ever hold one layer's value)."""
    os.makedirs(label_maps_dir, exist_ok=True)
    name = _layer_name(layer)

    label_sitk = sitk.GetImageFromArray(mask_3d.astype(np.uint8))
    label_sitk.CopyInformation(ref_sitk)
    nrrd_path = os.path.join(label_maps_dir, f"{name}.nrrd")
    sitk.WriteImage(label_sitk, nrrd_path)

    r, g, b = layer.line_color
    ctbl_path = os.path.join(label_maps_dir, f"{name}.ctbl")
    with open(ctbl_path, "w") as f:
        f.write("# Color table file generated by vtp_to_nrrd_overlay\n# 2\n")
        f.write("0 Background 0 0 0 0\n")
        f.write(f"1 {name} {r} {g} {b} 255\n")

    print(f"Saved label map + color table for '{name}' to:\n{nrrd_path}\n{ctbl_path}")


def vtp_points_to_rgb_nrrd(nrrd_path, output_path, thickness, layers, label_maps_dir=None):
    """Paints each of `layers` (OverlayLayer) onto one RGB copy of `nrrd_path`,
    later layers painted on top of earlier ones where they overlap. If
    `label_maps_dir` is set, also writes each layer's own binary label-map
    NRRD + color table there - see module docstring for how to use those in
    Slicer to toggle layers independently, without one overwriting another's
    voxels the way the flattened RGB/paint-order output does."""
    # 1. Read reference ultrasound image.
    ref_sitk = sitk.ReadImage(nrrd_path)
    dims = ref_sitk.GetSize()
    spacing = np.array(ref_sitk.GetSpacing())
    origin = np.array(ref_sitk.GetOrigin())
    dir_mat = np.array(ref_sitk.GetDirection()).reshape(3, 3)

    gray_array = sitk.GetArrayFromImage(ref_sitk).astype(np.float32)
    gray_min, gray_max = gray_array.min(), gray_array.max()
    if gray_max > gray_min:
        gray_uint8 = ((gray_array - gray_min) / (gray_max - gray_min) * 255.0).astype(np.uint8)
    else:
        gray_uint8 = gray_array.astype(np.uint8)

    # 2. Rasterize and paint each layer in turn.
    rgb_array = np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)
    total_painted = 0
    for layer in layers:
        mask_3d = _rasterize_layer(
            layer.vtp_path,
            thickness=layer.thickness if layer.thickness is not None else thickness,
            dims=dims,
            spacing=spacing,
            origin=origin,
            dir_mat=dir_mat,
            transform_cfg=layer.transform,
            landmark_align_cfg=layer.landmark_align,
        )
        rgb_array[mask_3d] = layer.line_color
        if label_maps_dir:
            _write_layer_label_map(label_maps_dir, layer, mask_3d, ref_sitk)

        painted_count = np.count_nonzero(mask_3d)
        total_painted += painted_count
        print(f"Painted {painted_count} voxels in color {tuple(layer.line_color)} from '{layer.vtp_path}'.")

    out_sitk = sitk.GetImageFromArray(rgb_array, isVector=True)
    out_sitk.CopyInformation(ref_sitk)
    sitk.WriteImage(out_sitk, output_path)

    print(f"Success! Painted {total_painted} voxels total onto original scan.")
    print(f"Saved RGB NRRD to:\n{output_path}")


def run(cfg: Config):
    vtp_points_to_rgb_nrrd(
        nrrd_path=cfg.nrrd_path,
        output_path=cfg.output_path,
        thickness=cfg.thickness,
        layers=cfg.layers,
        label_maps_dir=cfg.label_maps_dir,
    )
