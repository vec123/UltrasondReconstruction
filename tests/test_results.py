"""result.json records (src/shared/results.py) and the evaluation's ``result`` layout.

    python -m pytest tests/test_results.py
    python tests/test_results.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.loader.layouts import ResultLayout  # noqa: E402
from src.evaluation.loader.pair_loader import detect_layout  # noqa: E402
from src.shared.results import Result, read_results, similarity_record, write_results  # noqa: E402
from src.shared.transforms.similarity import SimilarityTransform  # noqa: E402
from src.shared.vtk.create import create_polydata  # noqa: E402
from src.shared.vtk.io import save_vtp  # noqa: E402


def _mesh(path, shift=0.0):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.5]], dtype=float) + shift
    save_vtp(create_polydata(points, np.array([[0, 1, 2], [1, 3, 2]])), str(path))


def _transform():
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    matrix = np.eye(4)
    matrix[:3, :3] = 2.5 * rotation
    matrix[:3, 3] = [1, 2, 3]
    return SimilarityTransform(matrix)


def test_round_trip():
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as elsewhere:
        folder = Path(root)
        write_results(folder, {"fit": Result(str(folder / "recon.vtp"), f"{elsewhere}/target.vtp",
                                              recon_to_scan=similarity_record(_transform()),
                                              model={"kind": "gpmm", "path": "m.npz"})})
        raw = (folder / "result.json").read_text()
        assert '"recon": "recon.vtp"' in raw and f'"target": "{elsewhere}/target.vtp"' in raw
        result = read_results(folder)["fit"]
        assert result.recon == str(folder / "recon.vtp") and result.landmarks is None
        assert np.allclose(result.transform().matrix, _transform().matrix, atol=1e-12)
        assert Result("a", "b").transform() is None


def test_result_layout():
    with tempfile.TemporaryDirectory() as root:
        folder = Path(root)
        _mesh(folder / "rigid_alignment.vtp")
        _mesh(folder / "map.vtp", shift=0.1)
        _mesh(folder / "refined.vtp", shift=0.2)
        assert detect_layout(folder) == "direct_ssm_fit"
        write_results(folder, {name: Result(str(folder / f"{name}.vtp"), str(folder / "rigid_alignment.vtp"),
                                            recon_to_scan=similarity_record(_transform()))
                               for name in ("map", "refined")})
        assert detect_layout(folder) == "result"
        pair = ResultLayout(result="map").load(folder, "case")
        assert np.allclose(pair.recon.points[0], [0.1, 0.1, 0.1]) and pair.target_path == folder / "rigid_alignment.vtp"
        assert np.allclose(pair.recon_to_surface.matrix, _transform().matrix, atol=1e-12)
        for layout, message in [(ResultLayout(), "name one"), (ResultLayout(result="x"), "has no 'x'")]:
            try:
                layout.load(folder, "case")
            except ValueError as error:
                assert message in str(error), (message, error)
            else:
                raise AssertionError(f"{layout} did not raise")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
