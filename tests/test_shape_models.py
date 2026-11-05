"""The ShapeModel interface (src/shared/shape_model.py): code priors, the GPMM kind, load_model.

    python -m pytest tests/test_shape_models.py
    python tests/test_shape_models.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.gpmm.model.model import StatisticalShapeModel  # noqa: E402
from src.models.gpmm.shape_model import GpmmShapeModel  # noqa: E402
from src.shared.core.registry import LazyRegistry  # noqa: E402
from src.shared.shape_model import CATEGORY, CodePrior, load_model  # noqa: E402


def _gpmm(n=5, k=3, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((3 * n, k)))
    landmarks = {"verts ": [0, 4], "names ": ["prn ", "n "]}
    return StatisticalShapeModel(rng.standard_normal(3 * n), q, np.array([4.0, 1.0, 0.25]), landmarks)


def test_standard_prior():
    prior = CodePrior.standard(3)
    theta = torch.randn(4, 3)
    assert torch.equal(prior.code(theta), theta)
    assert torch.allclose(prior.energy(theta), 0.5 * (theta ** 2).sum(-1))
    assert prior.sample(1000, torch.Generator().manual_seed(0)).std(0).sub(1).abs().max() < 0.1


def test_prior_with_basis_and_precision():
    prior = CodePrior(torch.tensor([1.0, 2.0]), torch.tensor([[2.0], [0.0]]), torch.tensor([0.5]),
                      torch.tensor([[4.0]]))
    theta = torch.tensor([[1.5]])
    assert torch.equal(prior.code(theta), torch.tensor([[4.0, 2.0]]))
    assert torch.allclose(prior.energy(theta), torch.tensor([2.0]))       # 1/2 * 4 * (1.5 - 0.5)^2
    draws = prior.sample(20000, torch.Generator().manual_seed(0))
    assert abs(draws.mean() - 0.5) < 0.01 and abs(draws.std() - 0.5) < 0.01


def test_gpmm_decode():
    model = _gpmm()
    shape_model = GpmmShapeModel(model)
    s = torch.randn(2, 3)
    expected = (model.mean[None] + (s.numpy() * np.sqrt(model.eigenvalues)) @ model.eigenfunctions.T).reshape(2, -1, 3)
    assert np.allclose(shape_model.decode(s).numpy(), expected, atol=1e-5)
    idx = torch.tensor([[0, 3], [1, 1]])
    assert torch.allclose(shape_model.decode(s, idx), shape_model.decode(s)[torch.arange(2)[:, None], idx])
    assert shape_model.template.landmarks == {"prn": 0, "n": 4}
    assert shape_model.prior.size == 3


def test_load_model():
    registry = LazyRegistry()
    registry.register(CATEGORY, "gpmm", "src.models.gpmm.shape_model:GpmmShapeModel")
    for spec, message in [({"path": "x"}, "needs 'kind:'"), ({"kind": "gpmm", "file": "x"}, "Unknown key(s) ['file']")]:
        try:
            load_model(spec, registry)
        except ValueError as error:
            assert message in str(error), (message, error)
        else:
            raise AssertionError(f"{spec} did not raise")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
