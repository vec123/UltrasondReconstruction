"""recon_target_distance on a model's result folder, in each frame.

    python -m pytest tests/test_evaluate.py
    python tests/test_evaluate.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.config import ReconTargetDistanceConfig as Config  # noqa: E402
from src.evaluation.stages import recon_target_distance  # noqa: E402
from src.evaluation.stages.distance_distribution_overlay import read_distance_field  # noqa: E402
from src.shared.results import Result, similarity_record, write_results  # noqa: E402
from src.shared.transforms.similarity import SimilarityTransform  # noqa: E402
from src.shared.vtk.create import create_polydata  # noqa: E402
from src.shared.vtk.fields import add_point_field  # noqa: E402
from src.shared.vtk.io import save_vtp  # noqa: E402

SHIFT = np.array([0.06, 0.0, 0.08])  # target = recon + SHIFT, in the recon frame: 0.1 from each recon point
SCALE = 2.0
LANDMARKS = {"prn": 0, "enL": 4, "enR": 20, "pg": 62}  # label -> recon vertex, not in one plane


def _grid():
    axis = np.arange(4.0)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def _recon_to_scan():
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    matrix = np.eye(4)
    matrix[:3, :3] = SCALE * rotation
    matrix[:3, 3] = [10.0, -5.0, 3.0]
    return SimilarityTransform(matrix)


def _result_folder(root):
    """A model output folder: recon with heatmap fields, target, result.json, and the scan's landmark CSV."""
    folder = Path(root, "fit")
    folder.mkdir()
    points = _grid()
    recon = create_polydata(points)
    for label, vertex in LANDMARKS.items():
        heatmap = np.zeros(len(points))
        heatmap[vertex] = 1.0
        add_point_field(recon, heatmap, field_name=f"heatmap_{label}")
    save_vtp(recon, str(folder / "recon.vtp"))
    save_vtp(create_polydata(points + SHIFT), str(folder / "target.vtp"))
    write_results(folder, {"fit": Result(recon=str(folder / "recon.vtp"), target=str(folder / "target.vtp"),
                                         recon_to_scan=similarity_record(_recon_to_scan()))})

    scan_points = _recon_to_scan().apply_points(points[list(LANDMARKS.values())])
    landmarks = Path(root, "landmarks.csv")
    pd.DataFrame({"label": list(LANDMARKS), "x": scan_points[:, 0], "y": scan_points[:, 1],
                  "z": scan_points[:, 2]}).to_csv(landmarks, index=False)
    return folder, landmarks


def _distances(out):
    return (read_distance_field(str(out / "recon_w_distance.vtp"), "recon_dist_2_target"),
            read_distance_field(str(out / "target_w_distance.vtp"), "target_dist_2_recon"))


def test_frames():
    with tempfile.TemporaryDirectory() as root:
        folder, landmarks = _result_folder(root)
        for frame, scale in (("recon", 1.0), ("scan", SCALE), ("landmarks", SCALE)):
            out = Path(root, frame)
            recon_target_distance.run(Config(result_dir=str(folder), frame=frame, landmarks_path=str(landmarks),
                                             output_dir=str(out)))
            for distances in _distances(out):
                assert np.allclose(distances, scale * np.linalg.norm(SHIFT), atol=1e-5), (frame, distances[:3])
            summary = json.loads((out / "distance_summary.json").read_text())
            assert summary["recon_path"] == str(folder / "recon.vtp")
            assert summary["units"] == recon_target_distance.DEFAULT_UNITS[frame]
            landmark = json.loads((out / "landmark_distances.json").read_text())
            assert sorted(landmark["landmarks"]) == sorted(LANDMARKS)
            if frame != "recon":  # the CSV is in the scan frame
                assert landmark["summary"]["max"] < 1e-4, (frame, landmark["summary"])


def test_without_landmarks():
    with tempfile.TemporaryDirectory() as root:
        folder, _ = _result_folder(root)
        out = Path(root, "out")
        recon_target_distance.run(Config(result_dir=str(folder), layout="result", layout_kwargs={"result": "fit"},
                                         frame="scan", units="mm", output_dir=str(out)))
        assert not (out / "landmark_distances.json").exists()
        assert json.loads((out / "distance_summary.json").read_text())["landmark_distances"] is None


def test_wrong_settings():
    with tempfile.TemporaryDirectory() as root:
        folder, landmarks = _result_folder(root)
        recon, target = str(folder / "recon.vtp"), str(folder / "target.vtp")
        for cfg, message in [
            (Config(result_dir=str(folder), recon_path=recon, frame="scan"), "not both"),
            (Config(recon_path=recon, target_path=target, frame="scan"), "gives none"),
            (Config(recon_path=recon, target_path=target), "requires landmarks_path"),
            (Config(result_dir=str(folder), frame="model"), "frame must be one of"),
        ]:
            try:
                recon_target_distance.run(Config(**{**cfg.__dict__, "output_dir": str(Path(root, "out"))}))
            except ValueError as error:
                assert message in str(error), (message, error)
            else:
                raise AssertionError(f"{cfg} did not raise")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
