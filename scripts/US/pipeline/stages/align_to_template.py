"""Stage: align every subject's mesh onto a fixed template's landmark space.

The template is a VTP carrying `heatmap_{landmark_name}` point fields (as
produced by the landmark_geodesics stage on a hand-picked reference shape);
target landmark positions are the argmax of each heatmap field. Each subject's
own landmark CSV is matched to those positions by label, a similarity
transform is fit, and applied to that subject's mesh and landmarks.

The transform saved per subject is composed with whatever transform the
previous stage (align_to_reference) already saved for that subject, so it
always maps directly from the very first mesh frame into template space -
see pipeline/transforms.py.
"""

import glob
import os
import shutil

import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from src.config.US import AlignToTemplateConfig as Config
from src.vtk.io import load_vtp, save_vtp

from ..landmarks import points_to_labeled_polydata
from ..matching import match_by_subject_id
from ..transforms import RigidTransform, compose_with_previous, save_subject_transform


def extract_landmarks_from_heatmap_vtp(target_vtp_path, heatmap_prefix=""):
    """
    Extracts 3D landmark locations from a target VTP containing point fields
    named `heatmap_{landmark_name}`, by finding the point index with the
    maximum scalar intensity for each field.
    """
    polydata = load_vtp(target_vtp_path)
    point_data = polydata.GetPointData()

    target_landmarks = []
    for i in range(point_data.GetNumberOfArrays()):
        array_name = point_data.GetArrayName(i)
        if array_name and array_name.startswith(heatmap_prefix):
            landmark_label = array_name[len(heatmap_prefix):]
            heatmap_vals = vtk_to_numpy(point_data.GetArray(array_name))

            max_idx = int(np.argmax(heatmap_vals))
            point_coord = polydata.GetPoint(max_idx)

            target_landmarks.append({
                "label": landmark_label,
                "x": point_coord[0],
                "y": point_coord[1],
                "z": point_coord[2],
            })

    if not target_landmarks:
        raise ValueError(f"No point fields starting with '{heatmap_prefix}' found in {target_vtp_path}")

    df_target = pd.DataFrame(target_landmarks)
    print(f"Extracted {len(df_target)} target landmarks from heatmap point fields in {os.path.basename(target_vtp_path)}:")
    print(df_target[["label", "x", "y", "z"]])
    return df_target


def run(cfg: Config):
    shape_files = glob.glob(os.path.join(cfg.vtp_shapes_dir, "*.vtp"))
    csv_files = glob.glob(os.path.join(cfg.landmarks_csv_dir, "*.csv"))

    # Resolve subdirectory defaults and write them back onto cfg, so that
    # `align_to_template.out_shapes_vtp_dir` etc. resolve to the real path
    # when a later pipeline stage references this stage's config.
    cfg.out_shapes_vtp_dir = cfg.out_shapes_vtp_dir or os.path.join(cfg.output_dir, "shapes_vtp")
    cfg.out_landmarks_csv_dir = cfg.out_landmarks_csv_dir or os.path.join(cfg.output_dir, "landmarks_csv")
    cfg.out_landmarks_vtp_dir = cfg.out_landmarks_vtp_dir or os.path.join(cfg.output_dir, "landmarks_vtp")
    cfg.transforms_dir = cfg.transforms_dir or os.path.join(cfg.output_dir, "transforms")

    os.makedirs(cfg.out_shapes_vtp_dir, exist_ok=True)
    os.makedirs(cfg.out_landmarks_csv_dir, exist_ok=True)
    os.makedirs(cfg.out_landmarks_vtp_dir, exist_ok=True)
    os.makedirs(cfg.transforms_dir, exist_ok=True)

    # Copy the template VTP into the output directory for reference.
    target_template_out_path = os.path.join(cfg.output_dir, os.path.basename(cfg.template_vtp))
    shutil.copy2(cfg.template_vtp, target_template_out_path)
    print(f"Saved reference template to: {target_template_out_path}")

    target_lmk_df = extract_landmarks_from_heatmap_vtp(cfg.template_vtp, heatmap_prefix=cfg.heatmap_prefix)

    common_ids, (vtp_map, csv_map) = match_by_subject_id(
        [shape_files, csv_files], strip_suffixes=("_landmarks",)
    )
    if not common_ids:
        raise ValueError("No matching subject IDs found between provided VTP shape files and CSV landmark files.")

    print(f"\nFound {len(common_ids)} subjects to align to template space.")

    append_filter = vtk.vtkAppendPolyData()

    for subject_id in common_ids:
        vtp_path = vtp_map[subject_id]
        csv_path = csv_map[subject_id]

        src_df = pd.read_csv(csv_path).copy()
        if cfg.lps_to_ras:
            src_df["x"] = -src_df["x"]
            src_df["y"] = -src_df["y"]

        merged_df = src_df.merge(target_lmk_df, on="label", suffixes=("_src", "_dst"))
        if len(merged_df) < 3:
            print(f"Skipping {subject_id}: Insufficient matching landmarks ({len(merged_df)} found, minimum 3 required).")
            continue

        src_pts = merged_df[["x_src", "y_src", "z_src"]].values.astype(float)
        dst_pts = merged_df[["x_dst", "y_dst", "z_dst"]].values.astype(float)

        # Fit this stage's own transform, then compose with align_to_reference's
        # saved transform so the manifest maps all the way back to the first mesh.
        local_transform = RigidTransform.from_correspondences(
            src_pts, dst_pts, with_scale=cfg.with_scale, stage="align_to_template"
        )
        transform = compose_with_previous(local_transform, subject_id, cfg.previous_transforms_dir)
        save_subject_transform(
            cfg.transforms_dir, subject_id, transform, labels=merged_df["label"].astype(str).values
        )

        # Transform shape mesh and save.
        mesh_poly = load_vtp(vtp_path)
        aligned_mesh_poly = local_transform.apply_polydata(mesh_poly)
        out_vtp_path = os.path.join(cfg.out_shapes_vtp_dir, f"{subject_id}.vtp")
        save_vtp(aligned_mesh_poly, out_vtp_path)
        append_filter.AddInputData(aligned_mesh_poly)

        # Transform and save landmarks as .csv and .vtp.
        src_coords = src_df[["x", "y", "z"]].values.astype(float)
        aligned_coords = local_transform.apply_points(src_coords)

        aligned_csv_df = src_df.copy()
        aligned_csv_df[["x", "y", "z"]] = aligned_coords
        out_csv_path = os.path.join(cfg.out_landmarks_csv_dir, f"{subject_id}.csv")
        aligned_csv_df.to_csv(out_csv_path, index=False)

        labels = aligned_csv_df["label"].astype(str) if "label" in aligned_csv_df.columns else None
        lmk_polydata = points_to_labeled_polydata(aligned_coords, labels=labels)
        out_lmk_vtp_path = os.path.join(cfg.out_landmarks_vtp_dir, f"{subject_id}_landmarks.vtp")
        save_vtp(lmk_polydata, out_lmk_vtp_path)

        print(f"Successfully aligned shape & landmarks for subject: {subject_id}")

    if append_filter.GetNumberOfInputConnections(0) > 0:
        append_filter.Update()
        merged_vtp_path = os.path.join(cfg.output_dir, "merged_targets.vtp")
        save_vtp(append_filter.GetOutput(), merged_vtp_path)
        print(f"\nSaved combined merged VTP to: {merged_vtp_path}")
