"""Stage: smooth mesh geometry with a non-shrinking windowed-sinc filter.

Same shape as the roughness_filter stage - a folder of .vtp meshes in, a
folder of processed .vtp meshes out - but instead of removing vertices, this
one moves them: each mesh is smoothed to reduce staircase/speckle noise left
over from volume-to-surface extraction, without the shrinkage plain
Laplacian smoothing would cause.

Point count and connectivity are unchanged, so this can slot in anywhere a
.vtp directory is expected. It DOES move vertex positions, though - any
per-vertex field an earlier stage computed from those positions (e.g.
nrrd_to_vtp's "True vertices" / "True normals" fields) will no longer match
the smoothed geometry.
"""

import glob
import os

from src.config.US import SmoothingFilterConfig as Config
from src.vtk.io import load_vtp, save_vtp
from src.vtk.process import smooth_vtp


def process_one(vtp_path, cfg: Config):
    vtp = load_vtp(vtp_path)

    print("smooth mesh vtp")
    vtp = smooth_vtp(
        vtp,
        iterations=cfg.iterations,
        pass_band=cfg.pass_band,
        boundary_smoothing=cfg.boundary_smoothing,
        feature_edge_smoothing=cfg.feature_edge_smoothing,
    )

    return vtp


def run(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    vtp_files = sorted(glob.glob(os.path.join(cfg.input_dir, "*.vtp")))
    if not vtp_files:
        raise FileNotFoundError(f"No .vtp files found in '{cfg.input_dir}'.")

    print(f"Found {len(vtp_files)} .vtp files to process.\n" + "-" * 40)

    output_paths = []
    for vtp_path in vtp_files:
        base_name = os.path.basename(vtp_path)
        print(f"Processing: {base_name}")

        vtp = process_one(vtp_path, cfg)

        output_path = os.path.join(cfg.output_dir, base_name)
        save_vtp(vtp, output_path)
        output_paths.append(output_path)

    return output_paths
