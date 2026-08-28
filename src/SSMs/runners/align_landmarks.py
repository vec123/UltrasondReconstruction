"""Runner for the landmark-driven rigid alignment of the mean face model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio
import vtk
from vtk.util import numpy_support

from src.config.schema import AlignLandmarksConfig
from src.data.load import load_landmarks


def get_rigid_transform(source_pts, target_pts, allow_scale: bool = True):
    """Computes Scaling, Rotation, and Translation (Kabsch Algorithm)."""
    src_mean = np.mean(source_pts, axis=0)
    tgt_mean = np.mean(target_pts, axis=0)

    src_centered = source_pts - src_mean
    tgt_centered = target_pts - tgt_mean

    if allow_scale:
        src_scale = np.sqrt(np.sum(src_centered**2) / len(source_pts))
        tgt_scale = np.sqrt(np.sum(tgt_centered**2) / len(target_pts))
        scale = tgt_scale / src_scale
    else:
        scale = 1.0

    H = src_centered.T @ (tgt_centered / scale)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    T = tgt_mean - (scale * (R @ src_mean))
    return R, T, scale


def save_vtp(vertices, filename, connectivity):
    """Writes vertices + triangle connectivity as a binary .vtp file."""
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(vertices.astype(np.float32), deep=True))
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    cells = vtk.vtkCellArray()
    cells.SetData(3, numpy_support.numpy_to_vtk(connectivity.flatten(), array_type=vtk.VTK_ID_TYPE))
    polydata.SetPolys(cells)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(filename))
    writer.SetInputData(polydata)
    writer.Write()


@dataclass
class AlignLandmarksResult:
    R: np.ndarray
    T: np.ndarray
    s: float
    mean_vertices: np.ndarray
    aligned_vertices: np.ndarray
    trilist: np.ndarray
    written_files: list = field(default_factory=list)


class AlignLandmarksRunner:
    """Aligns the mean shape of the .mat model onto a handful of target landmarks."""

    def __init__(self, config: AlignLandmarksConfig):
        self.config = config

    def run(self) -> AlignLandmarksResult:
        cfg = self.config

        mean_vertices, trilist, lmks_vertsIND, landmark_names = self.load_model()
        target_landmarks = self.load_targets()

        self._log("landmark_names: ", landmark_names)

        src_pts, tgt_pts, used = self.match_landmarks(
            mean_vertices, lmks_vertsIND, landmark_names, target_landmarks
        )
        self._log(f"matched {len(used)} landmark(s): {used}")

        R, T, s = get_rigid_transform(src_pts, tgt_pts, allow_scale=cfg.allow_scale)
        aligned_vertices = (s * (R @ mean_vertices.T)).T + T

        written = self.save_results(mean_vertices, aligned_vertices, trilist, R, T, s)

        return AlignLandmarksResult(
            R=R,
            T=T,
            s=s,
            mean_vertices=mean_vertices,
            aligned_vertices=aligned_vertices,
            trilist=trilist,
            written_files=written,
        )

    # -- steps -------------------------------------------------------------

    def load_model(self):
        path = self.config.shape_model_mat
        if not path.is_file():
            raise FileNotFoundError(f"shape_model_mat does not exist: {path}")

        mat_data = sio.loadmat(str(path), squeeze_me=True)
        FaceModel = mat_data["BabyFaceModel"]

        refShape = FaceModel["refShape"].item()
        meanDeformation = FaceModel["meanDeformation"].item()
        triang = FaceModel["triang"].item()
        lmks_vertsIND = FaceModel["landmark_verts"].item() - 1
        landmark_names = FaceModel["landmark_names"].item()

        trilist = (triang - 1).astype(np.int64, order="C")
        mean_vertices = (
            meanDeformation + refShape.flatten(order="F")
        ).reshape((3, -1), order="F").T

        return mean_vertices, trilist, lmks_vertsIND, landmark_names

    def load_targets(self) -> dict:
        cfg = self.config
        if cfg.target_landmarks_json is not None:
            if not cfg.target_landmarks_json.is_file():
                raise FileNotFoundError(
                    f"target_landmarks_json does not exist: {cfg.target_landmarks_json}"
                )
            landmarks, _, _ = load_landmarks(str(cfg.target_landmarks_json))
            return landmarks
        return {k: np.asarray(v, dtype=np.float32) for k, v in cfg.target_landmarks.items()}

    def match_landmarks(self, mean_vertices, lmks_vertsIND, landmark_names, target_landmarks):
        src_pts_list = []
        tgt_pts_list = []
        used = []

        for i, name in enumerate(landmark_names):
            key = str(name).strip()
            if key not in target_landmarks:
                continue
            src_pts_list.append(mean_vertices[lmks_vertsIND[i]])
            tgt_pts_list.append(target_landmarks[key])
            used.append(key)

        if len(used) < 3:
            known = sorted(str(n).strip() for n in landmark_names)
            raise ValueError(
                f"need at least 3 matching landmarks, matched {len(used)}: {used}. "
                f"Model landmark names: {known}"
            )

        return np.array(src_pts_list), np.array(tgt_pts_list), used

    def save_results(self, mean_vertices, aligned_vertices, trilist, R, T, s) -> list:
        out = self.config.output
        out.dir.mkdir(parents=True, exist_ok=True)

        tf_path = out.path("transform")
        np.savez(str(tf_path), R=R, T=T, s=s)
        self._log(f"Rigid transformation saved to {tf_path}")

        original_path = out.path("original_mesh")
        aligned_path = out.path("aligned_mesh")
        save_vtp(mean_vertices, original_path, trilist)
        save_vtp(aligned_vertices, aligned_path, trilist)
        self._log(f"Meshes saved to {original_path} and {aligned_path}")

        return [tf_path, original_path, aligned_path]

    def _log(self, *args):
        if self.config.verbose:
            print(*args)
