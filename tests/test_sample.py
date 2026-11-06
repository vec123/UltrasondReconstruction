"""The sample stage (src/shared/sampling) on a small linear model with a full prior covariance.

    python -m pytest tests/test_sample.py
    python tests/test_sample.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from vtk.util.numpy_support import vtk_to_numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pipeline.registry import STAGE_REGISTRY  # noqa: E402
from src.shared.core.pipeline import engine  # noqa: E402
from src.shared.sampling.draws import standard_draws  # noqa: E402
from src.shared.shape_model import CATEGORY, MODEL_KINDS, CodePrior, ShapeModel, Template  # noqa: E402
from src.shared.vtk.io import load_vtp  # noqa: E402

COVARIANCE = np.array([[2.0, 0.5], [0.5, 1.0]])
PRIOR_MEAN = np.array([1.0, -1.0])


class LinearModel(ShapeModel):
    """x = mean + basis z on 4 points; prior theta = z ~ N(PRIOR_MEAN, COVARIANCE)."""

    def __init__(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.5]], dtype=np.float32)
        self.mean = torch.as_tensor(points)
        self.basis = torch.randn(4, 3, 2, generator=torch.Generator().manual_seed(0))
        self.template = Template(points.astype(float), np.array([[0, 1, 2], [1, 3, 2]]))
        self.prior = CodePrior(torch.zeros(2), torch.eye(2), torch.as_tensor(PRIOR_MEAN, dtype=torch.float32),
                               torch.as_tensor(np.linalg.inv(COVARIANCE), dtype=torch.float32), "test")

    @classmethod
    def load(cls, device="cpu"):
        return cls()

    def decode(self, z, idx=None):
        return self.mean + torch.einsum("nck,tk->tnc", self.basis, z)


MODEL_KINDS.register(CATEGORY, "linear_test", f"{__name__}:LinearModel")


def _run(root, name, **settings):
    config = Path(root, f"{name}.yaml")
    config.write_text(yaml.safe_dump({"stage": "sample", "model": {"kind": "linear_test"},
                                      "output_dir": str(Path(root, name)), **settings}))
    engine.run_case(config, STAGE_REGISTRY, {})
    return Path(root, name)


def _points(path):
    return vtk_to_numpy(load_vtp(str(path)).GetPoints().GetData()).astype(float)


def test_prior_modes_file():
    with tempfile.TemporaryDirectory() as root:
        out = _run(root, "prior", codes={"source": "prior", "n": 5, "seed": 3})
        c = np.load(out / "coefficients.npz")
        assert np.array_equal(c["sigmas"], standard_draws(5, 2, "random", 3))
        assert np.allclose(c["theta"], PRIOR_MEAN + c["sigmas"] @ np.linalg.cholesky(COVARIANCE).T, atol=1e-5)
        assert np.allclose(c["z"], c["theta"]) and str(c["prior"]) == "test"
        model = LinearModel()
        for i in range(5):
            expected = model.decode(torch.as_tensor(c["z"][i:i + 1]))[0].numpy()
            assert np.allclose(_points(out / "members" / f"sample_{i:05d}.vtp"), expected, atol=1e-5)
        assert np.allclose(_points(out / "template.vtp"),
                           model.decode(torch.as_tensor(PRIOR_MEAN[None], dtype=torch.float32))[0].numpy(), atol=1e-5)

        modes = _run(root, "modes", codes={"source": "modes", "parameters": 2, "steps": [-1.0, 1.0]})
        assert np.array_equal(np.load(modes / "coefficients.npz")["sigmas"], [[-1, 0], [1, 0], [0, -1], [0, 1]])

        again = _run(root, "file", codes={"source": "file", "file": str(out / "coefficients.npz")})
        assert np.array_equal(np.load(again / "coefficients.npz")["z"], c["z"])

        subset = _run(root, "subset", codes={"source": "prior", "n": 2}, subset=3)
        assert len(np.load(subset / "coefficients.npz")["vertex_idx"]) == 3
        assert load_vtp(str(subset / "members" / "sample_00001.vtp")).GetNumberOfPolys() == 0


def test_wrong_settings():
    with tempfile.TemporaryDirectory() as root:
        _run(root, "out", codes={"source": "prior", "n": 1})
        for name, settings, error in [("out", {"codes": {"n": 1}}, FileExistsError),
                                      ("k", {"codes": {"parameters": 3}}, ValueError),
                                      ("s", {"codes": {"source": "grid"}}, ValueError)]:
            try:
                _run(root, name, **settings)
            except error:
                pass
            else:
                raise AssertionError(f"{settings} did not raise {error.__name__}")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
