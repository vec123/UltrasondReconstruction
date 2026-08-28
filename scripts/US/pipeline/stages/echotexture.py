"""Stage: attach per-vertex "echotexture" fields sampled from the ORIGINAL nrrd
volume - not from the extracted surface geometry.

nrrd_to_vtp's contour step reduces each volume's full grayscale signal to a
binary in/out decision at one threshold; curvature/roughness computed on the
resulting mesh is downstream of that step and never sees the original echo
pattern (placenta: homogeneous "cotyledon" texture; fetal bone: hyperechoic
with acoustic shadowing). This stage samples a small voxel neighborhood around
each surface vertex, back in the source volume, and attaches local
intensity/gradient statistics as point fields (see
`src.vtk.echotexture.sample_echotexture`) - a signal orthogonal to anything
geometric. It also attaches gradient-STRUCTURE-TENSOR shape descriptors (see
`src.vtk.echotexture.sample_structure_tensor`): a different question from
intensity/gradient MAGNITUDE - whether nearby edges point the same way, all
lie along one shared plane, or point every-which-way: `echo_planarity` is
high on a flat sheet-like boundary (e.g. a face/skull profile), `echo_linearity`
is high along a tube/rod's axis (e.g. an umbilical cord or a limb), and
`echo_sphericity` is high where gradients point every-which-way (speckle
noise, or a flat, featureless patch). `echo_energy` is the total local
gradient energy those three ratios are normalized by - near zero (flat,
edge-free region) makes the ratios unreliable.

Every field (echo_* and structure-tensor alike) also gets a companion
`*_variability` field (see `src.vtk.echotexture.compute_field_variability`):
the standard deviation of that field's OWN value among vertices within
`variability_radius_mm` of each vertex, on the mesh's own surface. A
uniformly bright region has high `echo_local_mean` but low
`echo_local_mean_variability`; a patchy texture (or the edge of a hyperechoic
shadow) scores high on both.

Must run BEFORE any alignment stage (align_to_reference / align_to_template):
those rescale/rotate/translate the mesh, after which vertex world coordinates
no longer line up with the source nrrd's voxel grid. Point downstream stages
at THIS stage's output_dir instead of nrrd_to_vtp's directly to carry the
echotexture fields forward (vtkTransformPolyDataFilter, used by every
alignment stage, passes point data through unchanged).

`crop` / `crop_threshold` / `crop_margin` MUST equal the values passed to
nrrd_to_vtp for the same run - cropping shifts the volume's origin, and the
input vtp's vertex coordinates already reflect that shifted frame if
nrrd_to_vtp cropped. `denoise` / `normalize` do not affect the coordinate
frame (only which signal gets measured) and may be set independently of
nrrd_to_vtp's own choice - left at their defaults, the volume sampled here is
untouched ("raw") intensity.
"""

import dataclasses
import glob
import os
from typing import Optional, Tuple

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from src.geometry import load_and_preprocess_volume
from src.ultrasound.echotexture import compute_field_variability, sample_echotexture, sample_structure_tensor
from src.vtk.fields import add_point_field
from src.vtk.io import load_vtp, save_vtp

from ..matching import match_by_subject_id


@dataclasses.dataclass
class Config:
    vtp_dir: str = ""  # nrrd_to_vtp's output_dir (pre-alignment meshes)
    nrrd_dir: str = ""  # the SAME source directory as nrrd_to_vtp.input_dir
    output_dir: str = ""

    # Must mirror nrrd_to_vtp's own volume.crop* settings - see module docstring.
    crop: bool = False
    crop_threshold: int = 0
    crop_margin: float = 0

    # What signal to measure; independent of the coordinate frame above.
    denoise: Optional[str] = None  # "median" | "curvature" | None
    denoise_radius: int = 5
    normalize: bool = False
    clip_percentiles: Tuple[float, float] = (1.0, 99.0)
    norm_out_max: float = 255.0

    neighborhood_radius_mm: float = 1.5  # half-width of the sampled voxel neighborhood
    interp_order: int = 1  # 0 = nearest voxel, 1 = trilinear

    # Integration-scale radius (mm) for the structure-tensor orientation-
    # coherence fields (echo_linearity/planarity/sphericity/energy) - see
    # module docstring and src.vtk.echotexture.sample_structure_tensor.
    structure_tensor_radius_mm: float = 1.5

    # Radius (mm, on the mesh's own un-aligned surface) each field's *_variability
    # companion is measured over - see module docstring. Must run before any
    # alignment stage rescales the mesh, or this stops meaning a physical distance.
    variability_radius_mm: float = 10.0
    # Caps the neighbor search to each vertex's nearest N, so a dense mesh
    # doesn't turn variability_radius_mm into an unbounded (memory-blowing)
    # neighbor count - see compute_field_variability's docstring.
    variability_max_neighbors: int = 64

    field_prefix: str = "echo_"


def process_one(vtp, nrrd_path, cfg: Config):
    volume, spacing, origin = load_and_preprocess_volume(
        nrrd_path,
        preprocess=True,
        denoise=cfg.denoise,
        denoise_radius=cfg.denoise_radius,
        normalize=cfg.normalize,
        clip_percentiles=cfg.clip_percentiles,
        out_max=cfg.norm_out_max,
        crop=cfg.crop,
        crop_threshold=cfg.crop_threshold,
        crop_margin=cfg.crop_margin,
    )

    points = vtk_to_numpy(vtp.GetPoints().GetData())
    fields = sample_echotexture(
        points, volume, spacing, origin,
        radius_mm=cfg.neighborhood_radius_mm, interp_order=cfg.interp_order,
    )
    fields.update(sample_structure_tensor(
        points, volume, spacing, origin,
        radius_mm=cfg.structure_tensor_radius_mm, interp_order=cfg.interp_order,
    ))

    n_out = int((fields["in_bounds"] < 0.5).sum())
    if n_out:
        print(
            f"  warning: {n_out} of {len(points)} vertices fall outside the volume's voxel "
            f"grid (clamped to the nearest edge voxel) - check that crop/crop_threshold/"
            f"crop_margin match the values nrrd_to_vtp used for '{os.path.basename(nrrd_path)}'."
        )

    for name, values in fields.items():
        add_point_field(vtp, values, field_name=f"{cfg.field_prefix}{name}")

    variability_fields = compute_field_variability(
        points,
        {name: values for name, values in fields.items() if name != "in_bounds"},
        radius_mm=cfg.variability_radius_mm,
        max_neighbors=cfg.variability_max_neighbors,
    )
    for name, values in variability_fields.items():
        add_point_field(vtp, values, field_name=f"{cfg.field_prefix}{name}_variability")

    return vtp


def run(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)

    vtp_files = sorted(glob.glob(os.path.join(cfg.vtp_dir, "*.vtp")))
    nrrd_files = sorted(glob.glob(os.path.join(cfg.nrrd_dir, "*.nrrd")))
    if not vtp_files:
        raise FileNotFoundError(f"No .vtp files found in '{cfg.vtp_dir}'.")
    if not nrrd_files:
        raise FileNotFoundError(f"No .nrrd files found in '{cfg.nrrd_dir}'.")

    common_ids, (vtp_map, nrrd_map) = match_by_subject_id([vtp_files, nrrd_files])
    if not common_ids:
        raise ValueError(f"No matching subject ids between '{cfg.vtp_dir}' and '{cfg.nrrd_dir}'.")
    print(f"Found {len(common_ids)} matched (vtp, nrrd) pairs.\n" + "-" * 40)

    output_paths = []
    for subject_id in common_ids:
        print(f"Processing: {subject_id}")
        vtp = load_vtp(vtp_map[subject_id])
        vtp = process_one(vtp, nrrd_map[subject_id], cfg)

        output_path = os.path.join(cfg.output_dir, f"{subject_id}.vtp")
        save_vtp(vtp, output_path)
        output_paths.append(output_path)

    return output_paths
