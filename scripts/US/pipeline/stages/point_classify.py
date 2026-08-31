"""Stage: unsupervised clustering of mesh points using point-data FEATURE
fields already attached by an earlier stage (by default, echotexture's
echo_* fields) as each point's feature vector - an exploratory first attempt
at separating tissue types (face, placenta, hand/limb, noise, ...) with NO
labeled training data (see `src.vtk.classify` for the clustering itself and
what a cluster id does/doesn't mean).

Deliberately NOT wired into pipeline.yaml's `stages:`/`paths:`/`hyperparameters:`
- run it standalone against whatever directory (or single mesh file) you want
to try it on: `python scripts/run_stage.py --config configs/06_point_classify.yaml`.
It only READS an existing mesh (or directory of meshes) and WRITES its own
new output_dir + one new point field, so nothing else in the pipeline
references or depends on it. EASY TO REMOVE entirely: delete this file,
src/vtk/classify.py, configs/06_point_classify.yaml, and the "point_classify"
import + STAGE_REGISTRY entry in pipeline/registry.py.

Fits ONE clustering model across every input subject's points pooled
together, so a cluster id means the same thing from one subject to the
next - see src.vtk.classify.fit_predict_clusters.
"""

import os

import numpy as np

from src.config.US.schema import PointClassifyConfig as Config
from src.vtk.classify import describe_clusters, fit_predict_clusters
from src.vtk.extract import extract_vtp_point_fields
from src.vtk.fields import add_point_field
from src.vtk.io import glob_meshes, load_mesh, save_vtp


def _resolve_feature_fields(cfg: Config, first_mesh):
    if cfg.feature_fields:
        return list(cfg.feature_fields)

    point_data = first_mesh.GetPointData()
    names = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]
    detected = sorted(
        name for name in names
        if name and name.startswith(cfg.feature_prefix) and name not in cfg.exclude_fields
    )
    if not detected:
        raise ValueError(
            f"No point fields starting with '{cfg.feature_prefix}' found (after excluding "
            f"{cfg.exclude_fields}) on the first mesh in '{cfg.input_dir}'; set feature_fields "
            f"explicitly if the fields use a different naming convention."
        )
    return detected


def run(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)

    # input_dir may name a single mesh file directly (e.g. one specific
    # scan_consistency merged-threshold output living alongside sibling
    # threshold variants in the same directory) instead of a directory to glob.
    if os.path.isfile(cfg.input_dir):
        mesh_files = [cfg.input_dir]
    else:
        mesh_files = sorted(glob_meshes(cfg.input_dir))
    if not mesh_files:
        raise FileNotFoundError(f"No mesh files found at '{cfg.input_dir}'.")

    print(f"Loading {len(mesh_files)} mesh(es) from '{cfg.input_dir}'...")
    meshes = [load_mesh(f) for f in mesh_files]

    feature_fields = _resolve_feature_fields(cfg, meshes[0])
    print(f"Clustering on {len(feature_fields)} feature field(s): {feature_fields}")

    per_mesh_features = []
    for path, mesh in zip(mesh_files, meshes):
        field_values = extract_vtp_point_fields(mesh, feature_fields)
        missing = [name for name, values in field_values.items() if values is None]
        if missing:
            raise ValueError(f"'{path}' is missing point field(s) {missing} required for clustering.")
        per_mesh_features.append(np.stack([field_values[name] for name in feature_fields], axis=1))

    split_sizes = [f.shape[0] for f in per_mesh_features]
    X = np.concatenate(per_mesh_features, axis=0)

    print(f"Fitting {cfg.algorithm} (k={cfg.n_clusters}) over {X.shape[0]:,} pooled points...")
    labels, centers, scaler = fit_predict_clusters(
        X, n_clusters=cfg.n_clusters, algorithm=cfg.algorithm,
        random_state=cfg.random_state, n_init=cfg.n_init, batch_size=cfg.batch_size,
    )

    print("Cluster summary (means in each feature's own original units):")
    for line in describe_clusters(labels, centers, scaler, feature_fields):
        print(f"  {line}")

    output_paths = []
    offset = 0
    for path, mesh, n in zip(mesh_files, meshes, split_sizes):
        mesh_labels = labels[offset:offset + n].astype(np.float32)
        offset += n

        add_point_field(mesh, mesh_labels, field_name=cfg.class_field_name)

        output_path = os.path.join(cfg.output_dir, os.path.basename(path))
        save_vtp(mesh, output_path)
        output_paths.append(output_path)
        print(f"  saved: {output_path}")

    return output_paths
