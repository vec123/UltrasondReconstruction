"""Stage: rigidly (+ scale) aligns one mesh into another's coordinate frame,
via shared named landmarks - see src.landmarks.vtk_landmarks.fit_landmark_alignment.

`source_vtp` must carry its own `heatmap_{label}` point fields (e.g.
Templates/baby_mean.vtp, Templates/fetus_mean.vtp): the argmax vertex of each
field gives that label's position in source_vtp's own frame. `landmark_path`
is a named landmark CSV (x, y, z, label columns) already living in the target
frame - e.g. one subject's file from align_to_template's out_landmarks_csv_dir,
which shares align_to_template's own aligned frame with every other subject
(and with anything built from them, like scan_fusion's fused_scan.vtp).

The whole of source_vtp (geometry + every existing point field) is carried
through the fitted similarity transform and saved to output_path - unlike
fit_landmark_alignment's other caller (vtp_to_nrrd_overlay), which only needs
the moved points, this stage's whole job is producing a mesh other stages can
read back (e.g. as resnet_lddmm_train's source).
"""

from src.config.US.schema import SimilarityTransformConfig as Config
from src.landmarks.vtk_landmarks import fit_landmark_alignment
from src.vtk.io import load_vtp, save_vtp
from vtk.util.numpy_support import vtk_to_numpy


def run(cfg: Config):
    source = load_vtp(cfg.source_vtp)
    source_points = vtk_to_numpy(source.GetPoints().GetData())

    transform = fit_landmark_alignment(
        source_points,
        template_vtp=cfg.source_vtp,
        landmark_path=cfg.landmark_path,
        landmarks_frame=cfg.landmarks_frame,
        with_scale=cfg.with_scale,
        heatmap_prefix=cfg.heatmap_prefix,
    )

    aligned = transform.apply_polydata(source)
    save_vtp(aligned, cfg.output_path)
    print(
        f"Aligned '{cfg.source_vtp}' to '{cfg.landmark_path}' landmarks "
        f"(scale={transform.scale:.4f}) -> {cfg.output_path}"
    )
