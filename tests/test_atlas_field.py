"""The atlas field (fields/composite/atlas.py): the zero code is exactly the template.

    python -m pytest tests/test_atlas_field.py
    python tests/test_atlas_field.py
"""

import dataclasses
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.models.lddmm.resnet_flow.registrations  # noqa: E402,F401  (fills the Registry)
from src.models.lddmm.resnet_flow.conditioning.film import ConcatConditioning  # noqa: E402
from src.models.lddmm.resnet_flow.config import ExperimentCfg, FieldCfg  # noqa: E402
from src.models.lddmm.resnet_flow.fields.blocks import VelocityBlock  # noqa: E402
from src.models.lddmm.resnet_flow.fields.composite.atlas import AtlasField  # noqa: E402
from src.models.lddmm.resnet_flow.fields.composite.modes import network_of  # noqa: E402
from src.models.lddmm.resnet_flow.fields.composite.symmetry import SymmetricField  # noqa: E402
from src.models.lddmm.resnet_flow.fields.fourier import FourierTimeVaryingField  # noqa: E402
from src.models.lddmm.resnet_flow.runner.model import build_flow  # noqa: E402

N_Z = 8
STEPS = 4
KINDS = ("time_varying", "time_varying_fourier", "stationary_fourier")
VARIANTS = [(mode, rank) for mode in ("linear", "nonlinear") for rank in ("full", 3)]


def _cfg(kind="time_varying_fourier", atlas=None, code=None, **field):
    """A small ExperimentCfg: a 4-step field of one 16-wide block, an 8-entry code."""
    field = {"kind": kind, "num_steps": STEPS, "mlp": [16], "fourier_n_e": 2, **field}
    if atlas is not None:
        field["atlas"] = atlas
    return ExperimentCfg.from_dict({
        "source": "source", "target": "target", "output_dir": "out", "field": field,
        "code": code if code is not None else {"kind": "auto_decoder", "n_z": N_Z},
    })


def _randomise(flow, std=0.3):
    """Give every block a non-zero output layer: at the zero init every v is 0, a test of nothing."""
    torch.manual_seed(0)
    for module in flow.modules():
        if isinstance(module, VelocityBlock):
            torch.nn.init.normal_(module.proj.weight, std=std)
    return flow


def _points(batch=2, n=50):
    torch.manual_seed(1)
    return torch.rand(batch, n, 3)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode,rank", VARIANTS)
def test_zero_code_is_identity(kind, mode, rank):
    flow = _randomise(build_flow(_cfg(kind, {"mode": mode, "rank": rank})))
    x = _points()
    zero = torch.zeros(x.shape[0], N_Z)
    assert isinstance(flow.field, AtlasField)
    assert torch.equal(flow(x, zero).end, x)
    assert torch.equal(flow.inverse(x, zero).end, x)
    # A non-zero code does move the points, so the identity above is not a dead field.
    assert not torch.allclose(flow(x, torch.randn(x.shape[0], N_Z)).end, x)


@pytest.mark.parametrize("rank", ("full", 3))
def test_linear_velocity_is_linear_in_code(rank):
    field = _randomise(build_flow(_cfg(atlas={"mode": "linear", "rank": rank}))).field
    x = _points()
    a, b = torch.randn(2, N_Z), torch.randn(2, N_Z)
    combined = field(x, step=1, code=2.0 * a - 0.5 * b)
    assert torch.allclose(combined, 2.0 * field(x, step=1, code=a) - 0.5 * field(x, step=1, code=b), atol=1e-5)


@pytest.mark.parametrize("rank", ("full", 3))
def test_nonlinear_velocity_is_not_linear_in_code(rank):
    field = _randomise(build_flow(_cfg(atlas={"mode": "nonlinear", "rank": rank})), std=1.0).field
    x = _points()
    a = torch.randn(2, N_Z)
    assert not torch.allclose(field(x, step=1, code=3.0 * a), 3.0 * field(x, step=1, code=a), atol=1e-4)
    assert torch.equal(field(x, step=1, code=torch.zeros(2, N_Z)), torch.zeros_like(x))


def test_rank_sets_basis_width_and_code_map():
    full = build_flow(_cfg(atlas={"mode": "linear", "rank": "full"})).field
    low = build_flow(_cfg(atlas={"mode": "linear", "rank": 3})).field
    assert full.network.blocks[0].proj.weight.shape[0] == 3 * N_Z
    assert low.network.blocks[0].proj.weight.shape[0] == 3 * 3
    assert not any(k.startswith("mix.") for k in full.state_dict())
    assert low.state_dict()["mix.weight"].shape == (3, N_Z)
    # The linear basis network reads no code; the nonlinear one does.
    assert full.network.conditioning.dim == 0
    assert build_flow(_cfg(atlas={"mode": "nonlinear", "rank": 3})).field.network.conditioning.dim == N_Z


def test_without_atlas_the_model_is_unchanged():
    cfg = _cfg()
    assert cfg.field.atlas is None
    field = build_flow(cfg).field
    # The same keys and shapes as the field built the way it was before field.atlas.
    before = FourierTimeVaryingField(num_blocks=STEPS, mlp=(16,), conditioning=ConcatConditioning(N_Z), n_e=2)
    assert type(field) is FourierTimeVaryingField
    assert {k: v.shape for k, v in field.state_dict().items()} == \
           {k: v.shape for k, v in before.state_dict().items()}


def test_field_config_round_trips():
    # config.json is dataclasses.asdict of the config; apply parses it back.
    field = _cfg(atlas={"mode": "nonlinear", "rank": 3}).field
    assert FieldCfg.from_dict(dataclasses.asdict(field)).atlas == {"mode": "nonlinear", "rank": 3}
    assert FieldCfg.from_dict(dataclasses.asdict(_cfg().field)).atlas is None
    assert _cfg(atlas={"mode": "linear"}).field.atlas == {"mode": "linear", "rank": "full"}


def test_checkpoint_loads_into_rebuilt_flow():
    cfg = _cfg(atlas={"mode": "linear", "rank": 3})
    trained = _randomise(build_flow(cfg))
    rebuilt = build_flow(cfg)
    rebuilt.load_state_dict(trained.state_dict())
    x, z = _points(), torch.randn(2, N_Z)
    assert torch.equal(rebuilt(x, z).end, trained(x, z).end)


def test_symmetric_atlas_field_is_zero_at_zero_code_and_reaches_the_encoding():
    cfg = _cfg(atlas={"mode": "linear", "rank": 3})
    cfg = dataclasses.replace(cfg, field=dataclasses.replace(cfg.field, symmetry={"mode": "symmetric"}))
    flow = _randomise(build_flow(cfg))
    assert isinstance(flow.field, SymmetricField) and isinstance(flow.field.network, AtlasField)
    flow.field.plane.place(torch.tensor([1.0, 0.0, 0.0]), 0.5)
    x = _points()
    assert torch.equal(flow(x, torch.zeros(2, N_Z)).end, x)
    # FourierAnnealing finds the encoding through every wrapper (runner/callbacks.py).
    assert hasattr(network_of(flow.field), "encoding")


@pytest.mark.parametrize("atlas,match", [
    ({"mode": "affine"}, "mode must be one of"),
    ({"mode": "linear", "rank": 0}, "positive int"),
    ({"mode": "linear", "rank": True}, "positive int"),
    ({"mode": "linear", "scale": 1}, "scale"),
    ("linear", "must be a mapping"),
])
def test_parser_rejects_bad_atlas(atlas, match):
    with pytest.raises(ValueError, match=match):
        _cfg(atlas=atlas)


def test_parser_rejects_unsupported_kind_and_combinations():
    with pytest.raises(ValueError, match="supported for kinds"):
        _cfg("stationary", {"mode": "linear"})
    with pytest.raises(ValueError, match="field.prior"):
        _cfg(atlas={"mode": "linear"}, prior={"run": "some/run"})


def test_builder_rejects_bad_code():
    with pytest.raises(ValueError, match="rank < n_z"):
        build_flow(_cfg(atlas={"mode": "linear", "rank": N_Z}))
    with pytest.raises(ValueError, match="needs a code"):
        build_flow(_cfg(atlas={"mode": "linear"}, code={"kind": "none"}))
    with pytest.raises(ValueError, match="position_aware"):
        build_flow(_cfg(atlas={"mode": "linear"}, code={"kind": "auto_decoder", "n_z": N_Z,
                                                        "position_aware": True}))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
