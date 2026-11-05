"""The code_fit stage (src/shared/fitting/stage.py) on a small synthetic GPMM.

    python -m pytest tests/test_code_fit.py
    python tests/test_code_fit.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.pipeline.registry as registry  # noqa: E402  (the stages and the model kinds)
from src.evaluation.loader.layouts import ResultLayout  # noqa: E402
from src.models.gpmm.model.model import StatisticalShapeModel  # noqa: E402
from src.models.gpmm.shape_model import GpmmShapeModel  # noqa: E402
from src.shared.core.pipeline import engine  # noqa: E402
from src.shared.fitting.stage import Config, DataConfig, FrameConfig, model_to_scan, run  # noqa: E402
from src.shared.shape_model import CATEGORY, MODEL_KINDS  # noqa: E402
from src.shared.transforms.similarity import SimilarityTransform  # noqa: E402
from src.shared.vtk.create import create_polydata  # noqa: E402
from src.shared.vtk.io import save_vtp  # noqa: E402

N = 12
THETA = np.array([1.2, -0.8, 0.5])


def _grid():
    u, v = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    faces = [(i * N + j, i * N + j + 1, (i + 1) * N + j + 1) for i in range(N - 1) for j in range(N - 1)]
    faces += [(i * N + j, (i + 1) * N + j + 1, (i + 1) * N + j) for i in range(N - 1) for j in range(N - 1)]
    return u.ravel(), v.ravel(), np.array(faces)


def _model():
    """A curved patch of half-width 0.01 (model units) with 3 smooth, orthonormal modes;
    one sigma moves a vertex by about 3 mm in the scan, so the data defines the code."""
    u, v, faces = _grid()
    mean = np.stack([u, v, 0.3 * (u ** 2 + v ** 2)], axis=1) * 0.01
    modes = np.stack([np.stack([0 * u, 0 * u, f], axis=1).ravel() for f in (u, v, u * v)], axis=1)
    modes, _ = np.linalg.qr(modes)
    return StatisticalShapeModel(mean.ravel(), modes, np.array([1.0, 0.5, 0.25]) * 1e-4, {}, faces=faces)


class SyntheticGpmm(GpmmShapeModel):
    @classmethod
    def load(cls, device="cpu"):
        return cls(_model(), device)


MODEL_KINDS.register(CATEGORY, "synthetic_gpmm", f"{__name__}:SyntheticGpmm")


def _scene(root):
    """target.vtp (the model at THETA, in scan mm) and similarity.json (model -> scan)."""
    matrix = np.eye(4)
    matrix[:3, :3] = 4000.0 * np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    matrix[:3, 3] = [10.0, -5.0, 30.0]
    to_scan = SimilarityTransform(matrix)
    to_scan.to_json(f"{root}/similarity.json")
    shape = SyntheticGpmm.load()
    with torch.no_grad():
        model_frame = shape.decode(torch.tensor(THETA[None], dtype=torch.float32))[0].double().numpy()
    save_vtp(create_polydata(to_scan.apply_points(model_frame), shape.template.faces), f"{root}/target.vtp")
    save_vtp(create_polydata(to_scan.apply_points(shape.template.points), shape.template.faces), f"{root}/aligned.vtp")
    return to_scan


def _config(root, **changes):
    values = dict(target=f"{root}/target.vtp", output_dir=f"{root}/fit", model={"kind": "synthetic_gpmm"},
                  frame=FrameConfig(similarity=f"{root}/similarity.json"),
                  data=DataConfig(sigma_mm=0.5, points=N * N, fit_points=N * N), prior_weight=0.0, steps=400,
                  lr=0.05, device="cpu")
    values.update(changes)
    return Config(**values)


def test_fit_finds_the_code():
    with tempfile.TemporaryDirectory() as root:
        _scene(root)
        stage = Path(root, "code_fit.yaml")
        stage.write_text(yaml.safe_dump({
            "stage": "code_fit",
            "paths": {"target": f"{root}/target.vtp", "output_dir": f"{root}/fit"},
            "hyperparameters": {"model": {"kind": "synthetic_gpmm"}, "frame": {"similarity": f"{root}/similarity.json"},
                                "data": {"sigma_mm": 0.5, "points": N * N, "fit_points": N * N},
                                "prior_weight": 0.0, "steps": 400, "lr": 0.05, "device": "cpu"}}))
        engine.run_main(stage, registry.STAGE_REGISTRY)
        theta = np.array(json.loads(Path(root, "fit", "code.json").read_text())["theta"])
        summary = json.loads(Path(root, "fit", "summary.json").read_text())
        assert np.abs(theta - THETA).max() < 0.05, theta
        assert summary["target_to_fit_mm"]["mean"] < 0.05, summary
        pair = ResultLayout().load(Path(root, "fit"), "case")
        assert pair.target_path == Path(root, "target.vtp")
        assert np.allclose(pair.recon_to_surface.matrix, np.eye(4))
        assert (Path(root, "fit", "manifest.json")).is_file()


def test_prior_pulls_to_the_mean():
    with tempfile.TemporaryDirectory() as root:
        _scene(root)
        run(_config(root, data=DataConfig(sigma_mm=1e4, points=N * N, fit_points=N * N), prior_weight=1.0))
        theta = np.array(json.loads(Path(root, "fit", "code.json").read_text())["theta"])
        assert np.abs(theta).max() < 0.05, theta


def test_aligned_template_frame():
    with tempfile.TemporaryDirectory() as root:
        to_scan = _scene(root)
        found = model_to_scan(FrameConfig(aligned_template=f"{root}/aligned.vtp"), SyntheticGpmm.load())
        assert np.allclose(found.matrix, to_scan.matrix, atol=1e-6 * 4000)
        for frame in (FrameConfig(), FrameConfig(aligned_template="a", similarity="b")):
            try:
                model_to_scan(frame, SyntheticGpmm.load())
            except ValueError as error:
                assert "exactly one" in str(error)
            else:
                raise AssertionError(f"{frame} did not raise")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
