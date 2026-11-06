"""The point_fields stage and its field kinds (src/shared/point_fields).

    python -m pytest tests/test_point_fields.py
    python tests/test_point_fields.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from vtk.util.numpy_support import vtk_to_numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pipeline.registry import STAGE_REGISTRY  # noqa: E402
from src.shared.core.pipeline import engine  # noqa: E402
from src.shared.point_fields.kinds import build_field  # noqa: E402
from src.shared.vtk.create import create_polydata  # noqa: E402
from src.shared.vtk.fields import add_point_field  # noqa: E402
from src.shared.vtk.io import load_vtp, save_vtp  # noqa: E402

N = 12  # grid points per side
LANDMARK = 5 * N + 5  # the vertex of heatmap_prn


def _plane(path):
    """An N x N grid in z = 0 with faces, an upward 'normal' array, 'support' and one heatmap field."""
    axis = np.arange(N, dtype=float)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    points = np.column_stack([points, np.zeros(len(points))])
    index = np.arange(N * N).reshape(N, N)
    faces = np.concatenate([np.stack([index[:-1, :-1], index[1:, :-1], index[1:, 1:]], -1).reshape(-1, 3),
                            np.stack([index[:-1, :-1], index[1:, 1:], index[:-1, 1:]], -1).reshape(-1, 3)])
    mesh = create_polydata(points, faces)
    add_point_field(mesh, np.tile([0.0, 0.0, 1.0], (len(points), 1)).astype(np.float32), field_name="normal")
    add_point_field(mesh, np.full(len(points), 2.0, dtype=np.float32), field_name="support")
    heatmap = np.zeros(len(points))
    heatmap[LANDMARK] = 1.0
    add_point_field(mesh, heatmap, field_name="heatmap_prn")
    save_vtp(mesh, str(path))


def _field(mesh, name):
    array = mesh.GetPointData().GetArray(name)
    assert array is not None, f"no '{name}' array"
    return vtk_to_numpy(array)


def test_stage():
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        _plane(root / "plane.vtp")
        config = root / "fields.yaml"
        config.write_text(yaml.safe_dump({
            "stage": "point_fields", "input": str(root / "plane.vtp"), "output": str(root / "out.vtp"),
            "fields": [{"kind": "normals", "field": "face_normal"},
                       {"kind": "edge", "k": 8, "ramp_spacings": 2.0},
                       {"kind": "landmark_weight", "include": ["prn"], "decay_scale": 2.0, "max_weight": 3.0,
                        "min_weight": 0.1}]}))
        engine.run_case(config, STAGE_REGISTRY, {})

        mesh = load_vtp(str(root / "out.vtp"))
        assert np.allclose(np.abs(_field(mesh, "face_normal")[:, 2]), 1.0)
        edge = _field(mesh, "edge").reshape(N, N)
        assert edge[0].all() and edge[-1].all() and edge[:, 0].all() and not edge[3:-3, 3:-3].any()
        assert np.allclose(_field(mesh, "support_edge"), 2.0 * _field(mesh, "edge_weight"))
        weight = _field(mesh, "landmark_weight")
        assert np.argmax(weight) == LANDMARK and np.isclose(weight.max(), 3.0) and weight.min() >= 0.1
        assert np.allclose(_field(mesh, "support"), 2.0)  # the arrays it had stay
        # With output:, the input does not change.
        assert load_vtp(str(root / "plane.vtp")).GetPointData().GetArray("edge") is None


def test_wrong_entries():
    for entry, message in [({"weighting": "unit"}, "needs 'kind"), ({"kind": "colour"}, "colour"),
                           ({"kind": "edge", "gap": 1}, "gap"),
                           ({"kind": "landmark_weight", "include": ["prn"], "vtp": "x.vtp"}, "input and output")]:
        try:
            build_field(entry, "mesh.vtp", "test")
        except Exception as error:
            assert message in str(error), (message, error)
        else:
            raise AssertionError(f"{entry} did not raise")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
