"""Stage: keep only mesh points inside a bounding box built from the landmarks.

For each subject's aligned mesh and landmark set (paired by subject id),
takes the axis-aligned bounding box of that subject's landmark points and
pads each of its 6 faces independently by its own configured offset. Any
mesh vertex falling outside the padded box is dropped.

Meshes are expected to be in physical RAS world space (x = Right,
y = Anterior, z = Superior - the convention nrrd_to_world_vtp / align_to_
reference produce and preserve, since later stages only ever apply
similarity transforms on top of it). That makes z the vertical (inferior /
superior) axis: z_min padding extends the box DOWN, toward the chin and
neck; z_max padding extends it UP, above the eyes/forehead. The defaults
below lean generous on z_min (keep more chin) and strict everywhere else,
per how this stage is meant to be used - adjust the six offsets in config
if a particular shapes_dir/landmarks_dir pair lives in a different frame.

Shapes coming out of align_to_template live in scaled template space, so
offsets - defined in the original, pre-alignment mesh/volume (nrrd/ply)
space - are converted into that subject's own template-space units using
the per-subject scale factor align_to_template saved, before padding (same
convention as landmark_distance_cut).

Cutting by a box can leave several disconnected islands (e.g. a bridge of
neck tissue severed by the z_min face); with `keep_largest_component` only
the single largest connected piece of the cut mesh is kept.

Doesn't move geometry (only drops points), so it saves no transform.
"""

import dataclasses
import glob
import os

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from src.vtk.io import load_vtp, save_vtp
from src.vtk.process import extract_polydata_vertex_subset, filter_largest_component

from ..matching import match_by_subject_id
from ..transforms import load_subject_transform


@dataclasses.dataclass
class Config:
    shapes_dir: str = ""
    landmarks_dir: str = ""
    output_dir: str = ""

    # align_to_template's transforms_dir, used to convert the offsets below
    # (defined in the original nrrd/ply space) into each subject's own
    # scaled template-space units. Leave blank if shapes_dir is already in
    # the original, unscaled mm space.
    transforms_dir: str = ""

    # Padding added to each face of the landmarks' bounding box (mm, original
    # nrrd/ply space). x = Right(+)/Left(-), y = Anterior(+)/Posterior(-),
    # z = Superior(+, above eyes)/Inferior(-, toward chin).
    x_min_offset_mm: float = 5.0
    x_max_offset_mm: float = 5.0
    y_min_offset_mm: float = 5.0
    y_max_offset_mm: float = 5.0
    z_min_offset_mm: float = 25.0  # inferior: generous, keeps more of the chin
    z_max_offset_mm: float = 5.0   # superior: strict, stays close above the eyes

    keep_largest_component: bool = True  # re-extract the largest component after the box cut
    min_bridge_width: float = 1.0  # bridges narrower than this do not connect regions after the cut; 0 disables
    min_bridge_relative: bool = False  # interpret min_bridge_width as a multiple of the mean edge length


def offsets_mm(cfg: Config):
    """Returns (mins, maxs) as (3,) arrays in [x, y, z] order."""
    mins = np.array([cfg.x_min_offset_mm, cfg.y_min_offset_mm, cfg.z_min_offset_mm], dtype=np.float64)
    maxs = np.array([cfg.x_max_offset_mm, cfg.y_max_offset_mm, cfg.z_max_offset_mm], dtype=np.float64)
    return mins, maxs


def landmarks_bounding_box(lmk_points, offset_mins, offset_maxs):
    """Axis-aligned bounding box of `lmk_points`, padded per-face by the given offsets."""
    box_min = lmk_points.min(axis=0) - offset_mins
    box_max = lmk_points.max(axis=0) + offset_maxs
    return box_min, box_max


def points_in_box(mesh_points, box_min, box_max):
    return np.all((mesh_points >= box_min) & (mesh_points <= box_max), axis=1)


def run(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)

    shape_files = sorted(glob.glob(os.path.join(cfg.shapes_dir, "*.vtp")))
    if not shape_files:
        raise FileNotFoundError(f"No .vtp shapes found in '{cfg.shapes_dir}'.")

    lmk_files = sorted(glob.glob(os.path.join(cfg.landmarks_dir, "*.vtp")))
    if not lmk_files:
        raise FileNotFoundError(f"No .vtp landmarks found in '{cfg.landmarks_dir}'.")

    common_ids, (shape_map, lmk_map) = match_by_subject_id(
        [shape_files, lmk_files], strip_suffixes=("_landmarks",)
    )
    if not common_ids:
        raise ValueError("No matching subject IDs found between shape and landmark files.")

    offset_mins, offset_maxs = offsets_mm(cfg)
    print(
        f"Found {len(common_ids)} subjects to cut by landmark bounding box "
        f"(mins {offset_mins.tolist()} mm, maxs {offset_maxs.tolist()} mm)."
    )

    for subject_id in common_ids:
        shape_path = shape_map[subject_id]
        lmk_path = lmk_map[subject_id]

        mesh = load_vtp(shape_path)
        lmk_pts = vtk_to_numpy(load_vtp(lmk_path).GetPoints().GetData()).astype(np.float64)
        mesh_pts = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64)

        subject_mins, subject_maxs = offset_mins, offset_maxs
        if cfg.transforms_dir:
            scale = load_subject_transform(cfg.transforms_dir, subject_id).scale
            subject_mins, subject_maxs = offset_mins * scale, offset_maxs * scale

        box_min, box_max = landmarks_bounding_box(lmk_pts, subject_mins, subject_maxs)
        print(
            f"{subject_id}: {mesh.GetNumberOfPoints()} points, box [{box_min.round(4).tolist()}, "
            f"{box_max.round(4).tolist()}] shape units"
        )

        mask = points_in_box(mesh_pts, box_min, box_max)
        if not mask.any():
            raise ValueError(f"'{subject_id}': bounding box cut removed every point.")

        cut_mesh = extract_polydata_vertex_subset(mesh, mask)

        if cfg.keep_largest_component:
            print("  keep largest component of the cut mesh")
            if cfg.min_bridge_width > 0:
                cut_mesh = filter_largest_component(cut_mesh, cfg.min_bridge_width, relative=cfg.min_bridge_relative)
            else:
                cut_mesh = filter_largest_component(cut_mesh)

        out_path = os.path.join(cfg.output_dir, os.path.basename(shape_path))
        save_vtp(cut_mesh, out_path)
        print(f"  Kept {cut_mesh.GetNumberOfPoints():,} points -> {out_path}")

    print("Landmark bounding box cut completed successfully!")
