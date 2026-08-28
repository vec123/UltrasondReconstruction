"""Runner that exports the mean shape, its landmarks, and per-vertex model fields.

Everything written here is a property of the shape model itself rather than of
any fit, so it is the reference picture: what the mean face looks like, where
the landmarks sit on it, and - through the heatmaps - where the training
population actually varies and which modes are responsible.

The variance field is the useful one. For a vertex $v$ the model gives its
displacement as $d_v = \\sum_k \\alpha_k \\phi_k(v)$ with $\\alpha \\sim N(0, \\Lambda)$,
so $\\operatorname{Cov}(d_v) = \\sum_k \\lambda_k \\phi_k(v)\\phi_k(v)^\\top$ and the
total variance at that vertex is its trace, $\\sum_k \\lambda_k \\|\\phi_k(v)\\|^2$.
Its square root is a length, in model units, directly comparable to the
distance thresholds used elsewhere.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

from src.config.schema import MeanShapeConfig
from src.geometry.geometry_np import compute_mean_curvature
from src.geometry.heatmap import landmark_heatmaps
from src.geometry.vtk import add_point_field, create_polydata, save_vtp_mesh
from src.model.load import load_shape_model, load_shape_model_triangulation


@dataclass
class MeanShapeResult:
    vertices: np.ndarray                              # (N, 3) mean shape
    landmarks: Dict[str, dict] = field(default_factory=dict)
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    written_files: List[Path] = field(default_factory=list)


class MeanShapeRunner:
    """Writes the mean shape with heatmaps, plus the landmark table."""

    def __init__(self, config: MeanShapeConfig):
        self.config = config

        self.mean: Optional[np.ndarray] = None            # (N, 3)
        self.faces: Optional[np.ndarray] = None
        self.eigenfunctions: Optional[np.ndarray] = None  # (3N, K)
        self.eigenvalues: Optional[np.ndarray] = None     # (K,)
        self.landmark_names: List[str] = []
        self.landmark_indices: np.ndarray = np.empty(0, dtype=np.int64)
        self.fields: Dict[str, np.ndarray] = {}
        self.written_files: List[Path] = []

    def run(self) -> MeanShapeResult:
        self.load_model()
        self.compute_fields()
        self.write_mean_shape()
        self.write_landmarks()

        return MeanShapeResult(
            vertices=self.mean,
            landmarks=self.landmark_table(),
            fields=self.fields,
            written_files=self.written_files,
        )

    # -- inputs ------------------------------------------------------------

    def load_model(self):
        cfg = self.config
        if not cfg.shape_model_mat.exists():
            raise FileNotFoundError(f"shape_model_mat does not exist: {cfg.shape_model_mat}")

        mean, eigenfunctions, eigenvalues, landmarks = load_shape_model(
            cfg.shape_model_mat, normalize=cfg.model.normalize,
            atlas_options=cfg.model.atlas.to_options(), verbose=cfg.verbose,
        )
        self.mean = np.asarray(mean, dtype=np.float64).reshape(-1, 3)
        self.eigenfunctions = np.asarray(eigenfunctions, dtype=np.float64)
        self.eigenvalues = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
        self.faces = load_shape_model_triangulation(cfg.shape_model_mat)

        # load_mat_ssm already strips the names and makes the indices 0-based;
        # the trailing spaces in the keys come from the .mat field names
        self.landmark_names = [str(n) for n in landmarks["names "]]
        self.landmark_indices = np.asarray(landmarks["verts "], dtype=np.int64).reshape(-1)

        self._log(f"model:     {len(self.mean)} vertices, "
                  f"{self.eigenfunctions.shape[1]} modes, "
                  f"{0 if self.faces is None else len(self.faces)} triangles")
        self._log(f"landmarks: {len(self.landmark_names)}")
        extent = self.mean.max(0) - self.mean.min(0)
        self._log(f"extent:    {np.array2string(extent, precision=5)}")

    # -- fields ------------------------------------------------------------

    def compute_fields(self):
        cfg = self.config.fields
        n_vertices, n_modes = len(self.mean), self.eigenfunctions.shape[1]

        # (N, K) squared magnitude of each mode's displacement at each vertex
        per_mode_sq = (self.eigenfunctions.reshape(n_vertices, 3, n_modes) ** 2).sum(axis=1)

        if cfg.total_std:
            variance = per_mode_sq @ self.eigenvalues
            self._add("total_std", np.sqrt(variance))

        count = max(0, min(int(cfg.mode_count), n_modes))
        for k in range(count):
            sigma = float(np.sqrt(self.eigenvalues[k]))
            self._add(f"mode_{k:02d}_std", sigma * np.sqrt(per_mode_sq[:, k]))
            if cfg.mode_vectors:
                displacement = self.eigenfunctions.reshape(n_vertices, 3, n_modes)[:, :, k]
                self._add(f"mode_{k:02d}_vector", sigma * displacement)

        if cfg.curvature:
            self._add_curvature()

        if cfg.landmark_distance and len(self.landmark_indices):
            distance, nearest = cKDTree(self.mean[self.landmark_indices]).query(self.mean)
            self._add("landmark_distance", distance)
            self._add("nearest_landmark", nearest.astype(np.int32))

        if (cfg.landmark_heatmaps or cfg.combined_heatmap) and len(self.landmark_indices):
            self._add_landmark_heatmaps()

    def _add_landmark_heatmaps(self):
        cfg = self.config.fields
        per_landmark, combined = landmark_heatmaps(
            self.mean, self.faces, self.landmark_names, self.landmark_indices,
            want_per_landmark=cfg.landmark_heatmaps, want_combined=cfg.combined_heatmap,
            combined_mode=cfg.combined_mode, heat_sigma=cfg.heat_sigma,
            heat_normalize=cfg.heat_normalize, log=self._log,
        )

        if cfg.landmark_heatmaps:
            for name, values in per_landmark.items():
                self.fields[f"heatmap_{self._safe(name)}"] = values
            self._log(f"  {'heatmap_<landmark>':<22} {len(per_landmark)} fields, "
                      f"{'normalised to peak 1' if cfg.heat_normalize else 'unnormalised'}")

        if cfg.combined_heatmap:
            self._add("heatmap_landmarks", combined)
            self._log(f"  {'':22} combined by '{cfg.combined_mode}'")

    @staticmethod
    def _safe(name: str) -> str:
        """Landmark names become array names, so keep them free of whitespace."""
        return "_".join(str(name).split())

    def _add_curvature(self):
        """Mean curvature of the mean surface; optional, so a failure is not fatal."""
        try:
            curvature = compute_mean_curvature(self.mean, self.faces)
        except Exception as exc:                      # noqa: BLE001 - diagnostic field only
            self._log(f"  curvature skipped: {type(exc).__name__}: {exc}")
            return
        self._add("curvature", np.asarray(curvature, dtype=np.float64).reshape(-1))

    def _add(self, name: str, values: np.ndarray):
        values = np.asarray(values)
        if values.shape[0] != len(self.mean):
            raise ValueError(
                f"field '{name}' has {values.shape[0]} rows, expected {len(self.mean)}"
            )
        self.fields[name] = values

        flat = values.reshape(len(values), -1)
        if flat.shape[1] == 1:
            self._log(f"  {name:<22} {flat.min():+.4e} .. {flat.max():+.4e}")
        else:
            magnitude = np.linalg.norm(flat, axis=1)
            self._log(f"  {name:<22} |v| {magnitude.min():.4e} .. {magnitude.max():.4e}")

    # -- outputs -----------------------------------------------------------

    def write_mean_shape(self):
        self._log("fields:")
        polydata = create_polydata(self.mean, self.faces)
        for name, values in self.fields.items():
            add_point_field(polydata, np.ascontiguousarray(values), name)

        path = self.config.outputs.path("mean_shape")
        save_vtp_mesh(polydata, str(path))
        self.written_files.append(path)

    def landmark_table(self) -> Dict[str, dict]:
        table = {}
        total_std = self.fields.get("total_std")
        for name, index in zip(self.landmark_names, self.landmark_indices):
            entry = {"index": int(index), "position": self.mean[index]}
            if total_std is not None:
                entry["total_std"] = float(total_std[index])
            table[name] = entry
        return table

    def write_landmarks(self):
        if not self.landmark_names:
            self._log("no landmarks in the model, skipping the table")
            return

        table = self.landmark_table()
        columns = ["name", "vertex_index", "x", "y", "z"]
        if "total_std" in next(iter(table.values())):
            columns.append("total_std")

        path = self.config.outputs.path("landmarks_csv")
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for name, entry in table.items():
                row = [name, entry["index"], *(f"{v:.10g}" for v in entry["position"])]
                if "total_std" in entry:
                    row.append(f"{entry['total_std']:.10g}")
                writer.writerow(row)
        self.written_files.append(path)
        self._log(f"landmark table written to {path.name}")

        # the same points as a cloud, so they can be opened next to the surface
        points = np.array([entry["position"] for entry in table.values()])
        polydata = create_polydata(points)
        add_point_field(polydata, np.arange(len(points), dtype=np.int32), "landmark_id")
        path = self.config.outputs.path("landmarks_vtp")
        save_vtp_mesh(polydata, str(path))
        self.written_files.append(path)

    def _log(self, *args):
        if self.config.verbose:
            print(*args)
