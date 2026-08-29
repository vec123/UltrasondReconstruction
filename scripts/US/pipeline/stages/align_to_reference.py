"""Stage: align every subject's mesh + landmarks onto one reference subject.

Picks a reference subject (the first common id, unless one is given), fits a
similarity transform from each other subject's landmarks onto the reference
subject's landmarks, and applies it to that subject's landmarks and mesh.

The fitted transform is saved per subject (see pipeline/transforms.py) so it
can be composed with later stages and, ultimately, inverted to map a
processed result back onto the subject's original mesh frame.
"""

import glob
import os

from src.config.US.schema import AlignToReferenceConfig as Config
from src.vtk.io import glob_meshes, load_mesh, save_vtp

from src.landmarks.vtk_landmarks import landmarks_csv_to_polydata
from src.file_name_matching.matching import match_by_subject_id
from src.transforms.similarity import SimilarityTransform, compose_with_previous, save_subject_transform


def run(cfg: Config):
    csv_files = glob.glob(os.path.join(cfg.landmarks_csv_dir, "*.csv"))
    mesh_files = glob_meshes(cfg.vtp_shapes_dir)

    common_ids, (csv_map, vtp_map) = match_by_subject_id([csv_files, mesh_files])
    if not common_ids:
        raise ValueError(
            f"No matching ID pairs found between '{cfg.landmarks_csv_dir}' and '{cfg.vtp_shapes_dir}'."
        )
    print(f"Found {len(common_ids)} matched pairs across input directories.")

    # Resolve subdirectory defaults and write them back onto cfg, so that
    # `align_to_reference.aligned_shapes_vtp_dir` etc. resolve to the real
    # path when a later pipeline stage references this stage's config.
    cfg.aligned_landmarks_dir = cfg.aligned_landmarks_dir or os.path.join(cfg.output_dir, "aligned_landmarks")
    cfg.aligned_shapes_vtp_dir = cfg.aligned_shapes_vtp_dir or os.path.join(cfg.output_dir, "aligned_shapes_vtp")
    cfg.transforms_dir = cfg.transforms_dir or os.path.join(cfg.output_dir, "transforms")

    os.makedirs(cfg.aligned_landmarks_dir, exist_ok=True)
    os.makedirs(cfg.aligned_shapes_vtp_dir, exist_ok=True)
    os.makedirs(cfg.raw_landmarks_vtp_dir, exist_ok=True)
    os.makedirs(cfg.transforms_dir, exist_ok=True)

    reference_id = cfg.reference_subject_id
    if reference_id is None or reference_id not in common_ids:
        reference_id = common_ids[0]
    print(f"Using reference subject: {reference_id}")

    _, ref_df = landmarks_csv_to_polydata(csv_map[reference_id], lps_to_ras=cfg.lps_to_ras)

    for subject_id in common_ids:
        csv_path = csv_map[subject_id]
        vtp_path = vtp_map[subject_id]

        # 1. Read landmarks CSV and save the raw (unaligned) landmark VTP.
        raw_lmk_vtp_path = os.path.join(cfg.raw_landmarks_vtp_dir, f"{subject_id}_landmarks.vtp")
        lmk_poly, cur_df = landmarks_csv_to_polydata(csv_path, raw_lmk_vtp_path, lps_to_ras=cfg.lps_to_ras)

        # 2. Match landmark points by label column.
        merged_df = cur_df.merge(ref_df, on="label", suffixes=("_src", "_dst"))
        if len(merged_df) < 3:
            print(f"Skipping {subject_id}: insufficient matching landmarks.")
            continue

        src_pts = merged_df[["x_src", "y_src", "z_src"]].values.astype(float)
        dst_pts = merged_df[["x_dst", "y_dst", "z_dst"]].values.astype(float)

        # 3. Fit the similarity transform and (if chained) compose it with
        # whatever brought this subject's data into src_pts's frame.
        local_transform = SimilarityTransform.from_correspondences(
            src_pts, dst_pts, with_scale=cfg.with_scale, stage="align_to_reference"
        )
        transform = compose_with_previous(local_transform, subject_id, cfg.previous_transforms_dir)
        save_subject_transform(
            cfg.transforms_dir, subject_id, transform, labels=merged_df["label"].astype(str).values
        )

        # 4. Transform landmark VTP and save.
        aligned_lmk_poly = local_transform.apply_polydata(lmk_poly)
        aligned_lmk_vtp_path = os.path.join(cfg.aligned_landmarks_dir, f"{subject_id}_landmarks.vtp")
        save_vtp(aligned_lmk_poly, aligned_lmk_vtp_path)

        # 5. Transform landmark DataFrame coordinates and save as .csv.
        pts_raw = cur_df[["x", "y", "z"]].values.astype(float)
        pts_aligned = local_transform.apply_points(pts_raw)

        aligned_df = cur_df.copy()
        aligned_df[["x", "y", "z"]] = pts_aligned
        aligned_lmk_csv_path = os.path.join(cfg.aligned_landmarks_dir, f"{subject_id}_landmarks.csv")
        aligned_df.to_csv(aligned_lmk_csv_path, index=False)

        # 6. Load surface mesh (.vtp or .ply), apply transform, and save.
        mesh_poly = load_mesh(vtp_path)
        aligned_mesh_poly = local_transform.apply_polydata(mesh_poly)
        aligned_mesh_out_path = os.path.join(cfg.aligned_shapes_vtp_dir, f"{subject_id}.vtp")
        save_vtp(aligned_mesh_poly, aligned_mesh_out_path)

        print(f"Processed and aligned: {subject_id}")
