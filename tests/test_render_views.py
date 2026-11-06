"""The render_views stage on synthetic face meshes with heatmap landmarks.

    python -m pytest tests/test_render_views.py
    python tests/test_render_views.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pipeline.registry import STAGE_REGISTRY  # noqa: E402
from src.shared.core.pipeline import engine  # noqa: E402
from src.shared.vtk.create import create_polydata  # noqa: E402
from src.shared.vtk.fields import add_point_field  # noqa: E402
from src.shared.vtk.io import save_vtp  # noqa: E402

PANEL = 100
HEADER = 40  # render.write_grid's header height
LANDMARKS = {"exR": (-1, 1), "exL": (1, 1), "n": (0, 1.5), "pg": (0, -1.5), "prn": (0, 0)}


def _face(path):
    """A bump in z = 0 with heatmap fields at the five landmarks that set the cameras, and a 'dist' field."""
    axis = np.linspace(-2, 2, 17)
    x, y = (a.ravel() for a in np.meshgrid(axis, axis, indexing="ij"))
    points = np.column_stack([x, y, np.exp(-(x ** 2 + y ** 2))])
    index = np.arange(17 * 17).reshape(17, 17)
    faces = np.concatenate([np.stack([index[:-1, :-1], index[1:, :-1], index[1:, 1:]], -1).reshape(-1, 3),
                            np.stack([index[:-1, :-1], index[1:, 1:], index[:-1, 1:]], -1).reshape(-1, 3)])
    mesh = create_polydata(points, faces)
    for label, (lx, ly) in LANDMARKS.items():
        heatmap = np.zeros(len(points))
        heatmap[np.argmin((x - lx) ** 2 + (y - ly) ** 2)] = 1.0
        add_point_field(mesh, heatmap, field_name=f"heatmap_{label}")
    add_point_field(mesh, np.abs(x), field_name="dist")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_vtp(mesh, str(path))


def _run(root, case=None, **settings):
    config = root / "views.yaml"
    config.write_text(yaml.safe_dump({
        "stage": "render_views", "variables": {"ROOT": str(root)}, "output_dir": "${ROOT}/views",
        "panel_px": PANEL, "color": {"field": "dist", "max": 2.0},
        "columns": [{"title": "A", "mesh": "${ROOT}/a/${SUBJECT_ID}/${WEEK}/recon.vtp"},
                    {"title": "B", "mesh": "${ROOT}/b/${CASE_KEY}.vtp", "views": ["frontal"]}],
        **settings}))
    engine.run_case(config, STAGE_REGISTRY, case or {})
    return root / "views"


def _size(path):
    with Image.open(path) as image:
        return image.size


def test_per_case_and_grid():
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        for session in ("1/20 semanas", "2/20 semanas"):
            _face(root / "a" / session / "recon.vtp")
        _face(root / "b" / "1_20_semanas.vtp")  # case 2 has no B: "(missing)"

        views = _run(root)
        assert sorted(p.name for p in views.glob("*.png")) == ["1_20_semanas.png", "2_20_semanas.png"]
        assert _size(views / "2_20_semanas.png") == (2 * PANEL, HEADER + 3 * PANEL)  # frontal, left, right
        with Image.open(views / "1_20_semanas.png") as image:
            pixels = np.asarray(image)[HEADER:HEADER + PANEL, :PANEL].reshape(-1, 3)
        assert len(np.unique(pixels, axis=0)) > 10  # a coloured mesh, not an empty panel

        _run(root, layout="grid")
        assert _size(views / "views.png") == (2 * PANEL, HEADER + 2 * PANEL)  # one row per case

        # One case, every case variable given (a pipeline run): named by case_name.
        for png in views.glob("*.png"):
            png.unlink()
        _run(root, case={"SUBJECT_ID": "1", "WEEK": "20 semanas"}, case_name="1_20_semanas")
        assert [p.name for p in views.glob("*.png")] == ["1_20_semanas.png"]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
