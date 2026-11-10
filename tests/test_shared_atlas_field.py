"""The shared atlas flow (fields/composite/shared.py): template -> atlas -> member.

    python -m pytest tests/test_shared_atlas_field.py
    python tests/test_shared_atlas_field.py
"""

import dataclasses
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.models.lddmm.resnet_flow.registrations  # noqa: E402,F401  (fills the Registry)
from src.models.lddmm.resnet_flow.config import ExperimentCfg, FieldCfg  # noqa: E402
from src.models.lddmm.resnet_flow.fields.blocks import VelocityBlock  # noqa: E402
from src.models.lddmm.resnet_flow.fields.composite.atlas import AtlasField  # noqa: E402
from src.models.lddmm.resnet_flow.fields.composite.modes import network_of, shared_field_of  # noqa: E402
from src.models.lddmm.resnet_flow.fields.composite.shared import SharedAtlasField  # noqa: E402
from src.models.lddmm.resnet_flow.runner.loss_terms import priced_parts  # noqa: E402
from src.models.lddmm.resnet_flow.runner.model import build_flow  # noqa: E402

N_Z = 8
STEPS = 8
SHARED_STEPS = 2
KINDS = ("time_varying", "time_varying_fourier", "stationary_fourier")


def _cfg(kind="time_varying_fourier", shared=None, mode="cohort", **field):
    """A small cohort ExperimentCfg: an 8-step field, 2 of them shared, a linear full-rank atlas field."""
    field = {"kind": kind, "num_steps": STEPS, "mlp": [16], "fourier_n_e": 2,
             "atlas": {"mode": "linear", "rank": "full"}, **field}
    if shared is not False:
        field["shared"] = shared if shared is not None else {"steps": SHARED_STEPS}
    return ExperimentCfg.from_dict({
        "source": "source", "target": "target", "output_dir": "out", "field": field,
        "code": {"kind": "auto_decoder", "n_z": N_Z}, "train": {"mode": mode},
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
def test_zero_code_ends_at_the_atlas(kind):
    flow = _randomise(build_flow(_cfg(kind)))
    x = _points()
    trajectory = flow(x, torch.zeros(2, N_Z))
    atlas = trajectory.points[SHARED_STEPS]
    assert not torch.allclose(atlas, x), "the shared flow does not move: a test of nothing"
    # The member flow of the zero code does not move: every later step stays at the atlas.
    for k in range(SHARED_STEPS, STEPS + 1):
        assert torch.equal(trajectory.points[k], atlas)


def test_shared_steps_do_not_depend_on_the_code():
    flow = _randomise(build_flow(_cfg()))
    x = _points()
    a, b = flow(x, torch.randn(2, N_Z)), flow(x, torch.randn(2, N_Z))
    assert torch.equal(a.points[:SHARED_STEPS + 1], b.points[:SHARED_STEPS + 1])
    assert not torch.allclose(a.end, b.end), "the member steps must read the code"


def test_shared_part_is_priced_apart():
    flow = _randomise(build_flow(_cfg()))
    trajectory = flow(_points(), torch.randn(2, N_Z))
    shared = trajectory.parts["shared"]
    assert shared.shape == trajectory.velocities.shape
    assert torch.equal(shared[:SHARED_STEPS], trajectory.velocities[:SHARED_STEPS])
    assert torch.count_nonzero(shared[SHARED_STEPS:]) == 0


def test_structure_blocks_and_wrappers():
    field = build_flow(_cfg()).field
    assert isinstance(field, SharedAtlasField) and isinstance(field.network, AtlasField)
    assert len(field.shared.blocks) == SHARED_STEPS
    assert len(field.network.network.blocks) == STEPS - SHARED_STEPS
    assert field.shared.conditioning.dim == 0
    assert field.shared.blocks[0].proj.weight.shape[0] == 3
    # FourierAnnealing finds both encodings (runner/callbacks.py).
    member, shared = network_of(field), shared_field_of(field)
    assert member is field.network.network and shared is field.shared
    assert hasattr(member, "encoding") and hasattr(shared, "encoding") and member is not shared
    assert shared_field_of(build_flow(_cfg(shared=False)).field) is None


def test_shared_options_set_the_shared_field():
    cfg = _cfg(shared={"steps": 3, "mlp": [24, 24], "fourier_n_e": 1, "fourier_anneal": {"start": 1, "end": 5}},
               fourier_anneal={"start": 10, "end": 20})
    field = build_flow(cfg).field
    assert len(field.shared.blocks) == 3 and field.shared.blocks[0].lift.out_features == 24
    assert len(field.shared.blocks[0].hidden) == 1
    assert float(field.shared.encoding.alpha) == 0.0      # annealed: starts with every band off
    assert cfg.field.shared["fourier_anneal"] == {"start": 1, "end": 5}


def test_defaults_inherit_the_field_options():
    shared = _cfg(fourier_anneal={"start": 10, "end": 20}).field.shared
    assert shared == {"steps": SHARED_STEPS, "mlp": [16], "fourier_n_e": 2,
                      "fourier_anneal": {"start": 10, "end": 20}, "discount": 1.0}
    assert _cfg(shared={"steps": 2, "fourier_anneal": None},
                fourier_anneal={"start": 10, "end": 20}).field.shared["fourier_anneal"] is None


def test_inverse_returns_to_the_start():
    flow = _randomise(build_flow(_cfg()), std=0.05)
    x, z = _points(), torch.randn(2, N_Z)
    back = flow.inverse(flow(x, z).end, z).end
    assert torch.allclose(back, x, atol=1e-3)


def test_checkpoint_loads_into_rebuilt_flow():
    cfg = _cfg(shared={"steps": 2, "discount": "once"})
    trained = _randomise(build_flow(cfg))
    rebuilt = build_flow(cfg)
    rebuilt.load_state_dict(trained.state_dict())
    x, z = _points(), torch.randn(2, N_Z)
    assert torch.equal(rebuilt(x, z).end, trained(x, z).end)


def test_field_config_round_trips():
    field = _cfg(shared={"steps": 3, "discount": "once"}).field
    assert FieldCfg.from_dict(dataclasses.asdict(field)).shared == field.shared
    assert FieldCfg.from_dict(dataclasses.asdict(_cfg(shared=False).field)).shared is None


def test_discount_once_is_one_over_the_cohort_size():
    parts = {name: discount for name, _, discount in priced_parts(_cfg(shared={"steps": 2, "discount": "once"}), 281)}
    assert parts == {"shared_kinetic": 1 / 281}
    assert priced_parts(_cfg(shared={"steps": 2, "discount": 0.5}))[0][2] == 0.5
    assert priced_parts(_cfg())[0] == ("shared_kinetic", "shared", 1.0)
    with pytest.raises(ValueError, match="cohort members"):
        priced_parts(_cfg(shared={"steps": 2, "discount": "once"}))


@pytest.mark.parametrize("field,match", [
    ({"shared": {"steps": 2}, "atlas": None}, "needs field.atlas"),
    ({"shared": {"steps": STEPS}}, "1 <= steps <"),
    ({"shared": {"steps": 0}}, "1 <= steps <"),
    ({"shared": {"steps": True}}, "1 <= steps <"),
    ({"shared": {"steps": 2, "discount": -1}}, "discount"),
    ({"shared": {"steps": 2, "discount": "twice"}}, "discount"),
    ({"shared": {"steps": 2, "width": 3}}, "width"),
    ({"shared": "2"}, "must be a mapping"),
    ({"shared": {"steps": 2}, "symmetry": {"mode": "symmetric", "pairs": [["a", "b"]]}}, "symmetry"),
])
def test_parser_rejects_bad_shared(field, match):
    shared = field.pop("shared")
    with pytest.raises(ValueError, match=match):
        _cfg(shared=shared, **field)


def test_parser_rejects_anneal_on_a_kind_without_bands():
    with pytest.raises(ValueError, match="fourier_anneal"):
        _cfg("time_varying", shared={"steps": 2, "fourier_anneal": {"start": 1, "end": 2}})


def test_builder_rejects_a_pair_run():
    with pytest.raises(ValueError, match="train.mode cohort"):
        build_flow(_cfg(mode="pair"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
