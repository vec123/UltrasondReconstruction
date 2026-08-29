"""Stage: attach a per-vertex roughness field to each mesh and filter by it.

Computes a local-surface-roughness scalar for every vertex (see
``compute_vertex_roughness``), attaches it as a point field named
``roughness``, then keeps only the cells whose roughness stays below a
threshold fraction of the field's range - vertices in noisy/bumpy regions
are dropped. Thresholding can fragment a single coherent mesh, so by default
the largest connected component is re-extracted afterward - set
``keep_largest_component: false`` when the input is already a legitimate
union of disjoint pieces (e.g. scan_consistency's merged multi-subject scan),
where "largest component" would discard every subject but one.

The roughness *threshold* itself is a fraction of each mesh's own roughness
range, so it is invariant to the isotropic similarity scale align_to_reference
/ align_to_template apply - the same vertices pass whether this stage runs
before or after alignment. ``min_bridge_width`` is not: it is an absolute
distance, tuned for the original nrrd/ply mesh scale. When this stage runs
downstream of alignment (input mesh already rescaled, per-subject, to match a
reference/template), point ``transforms_dir`` at that alignment stage's saved
transforms so ``min_bridge_width`` is converted into each subject's own scaled
units the same way ``landmark_distance_cut`` converts ``max_distance_mm``.

If ``unfiltered_dir`` is set, each mesh - with the computed ``roughness``
field attached, before thresholding - is also saved there, so the full
(unfiltered) roughness distribution stays inspectable alongside the filtered
result.

Was previously folded into the ``nrrd_to_vtp`` stage; split out so roughness
filtering can run standalone against any directory of .vtp meshes.
"""

import glob
import os

from src.config.US.schema import RoughnessFilterConfig as Config
from src.vtk.extract import threshold_vtp_by_point_field
from src.vtk.io import load_vtp, save_vtp
from src.vtk.surface_measures import compute_vertex_roughness
from src.vtk.process import filter_largest_component

from src.file_name_matching.matching import subject_id_from_path
from src.transforms.similarity import load_subject_transform


def process_one(vtp, cfg: Config, min_bridge_width):
    print("compute vertex roughness")
    compute_vertex_roughness(
        vtp,
        n_rings=cfg.n_rings,
        scale_invariant=cfg.scale_invariant,
        field_name="roughness",
        method=cfg.method,
        neighbors=cfg.neighbors,
        max_neighbors=cfg.max_neighbors,
    )

    print(f"threshold roughness: keep below {cfg.threshold:.0%} of the field range")
    filtered = threshold_vtp_by_point_field(vtp, "roughness", cfg.threshold, relative=True, keep="below")

    if cfg.keep_largest_component:
        print("keep largest component of the thresholded mesh")
        if min_bridge_width > 0:
            filtered = filter_largest_component(filtered, min_bridge_width, relative=cfg.min_bridge_relative)
        else:
            filtered = filter_largest_component(filtered)

    return filtered


def run(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.unfiltered_dir:
        os.makedirs(cfg.unfiltered_dir, exist_ok=True)

    vtp_files = sorted(glob.glob(os.path.join(cfg.input_dir, "*.vtp")))
    if not vtp_files:
        raise FileNotFoundError(f"No .vtp files found in '{cfg.input_dir}'.")

    print(f"Found {len(vtp_files)} .vtp files to process.\n" + "-" * 40)

    output_paths = []
    for vtp_path in vtp_files:
        base_name = os.path.basename(vtp_path)
        print(f"Processing: {base_name}")

        min_bridge_width = cfg.min_bridge_width
        if cfg.transforms_dir and cfg.keep_largest_component and not cfg.min_bridge_relative:
            subject_id = subject_id_from_path(vtp_path)
            scale = load_subject_transform(cfg.transforms_dir, subject_id).scale
            min_bridge_width = cfg.min_bridge_width * scale
            print(f"  min_bridge_width {cfg.min_bridge_width:g} mm -> {min_bridge_width:.4f} shape units (scale {scale:.4f})")

        vtp = load_vtp(vtp_path)
        filtered = process_one(vtp, cfg, min_bridge_width)

        if cfg.unfiltered_dir:
            unfiltered_path = os.path.join(cfg.unfiltered_dir, base_name)
            save_vtp(vtp, unfiltered_path)

        output_path = os.path.join(cfg.output_dir, base_name)
        save_vtp(filtered, output_path)
        output_paths.append(output_path)

    return output_paths
