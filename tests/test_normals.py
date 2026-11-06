"""Vertex normals from faces (geometry.mesh_vertex_normals, surface_measures._vertex_normals) at any scale.

    python -m pytest tests/test_normals.py
    python tests/test_normals.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared.geometry.geometry import mesh_vertex_normals  # noqa: E402
from src.shared.vtk.surface_measures import _vertex_normals  # noqa: E402

SCALES = (1e-3, 1.0, 1e3)  # mm, and frames such as a model's unit frame
NORMALS = (mesh_vertex_normals, _vertex_normals)


def curved_grid(n=12, size_mm=60.0):
    """A bumpy n x n grid with an 85 mm diagonal: (verts, faces, sliver face, the vertex in no face)."""
    u = np.linspace(0.0, size_mm, n)
    x, y = np.meshgrid(u, u)
    z = 5.0 * np.sin(x / 10.0) * np.cos(y / 13.0)
    grid = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    faces = [[i * n + j, i * n + j + 1, (i + 1) * n + j] for i in range(n - 1) for j in range(n - 1)]
    faces += [[i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j] for i in range(n - 1) for j in range(n - 1)]
    # A near-degenerate face on the edge from vertex 0 to 1: its tip is 1e-10 mm off that edge.
    tip = (grid[0] + grid[1]) / 2 + [0.0, 0.0, 1e-10]
    verts = np.vstack([grid, tip, [[30.0, 30.0, 40.0]]])
    sliver = [0, len(grid), 1]
    return verts, np.array(faces + [sliver]), sliver, len(verts) - 1


def angles_deg(a, b):
    """Row-wise angles; atan2 stays exact near 0 degrees, where arccos loses ~1e-6 degrees."""
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a * b, axis=1)))


def test_normals_do_not_depend_on_scale():
    verts, faces, _, isolated = curved_grid()
    used = np.arange(len(verts)) != isolated
    for normals in NORMALS:
        reference = normals(verts, faces)
        for scale in SCALES:
            assert angles_deg(normals(verts * scale, faces)[used], reference[used]).max() < 1e-6, (normals, scale)


def test_normals_are_unit_or_zero():
    verts, faces, _, isolated = curved_grid()
    for normals in NORMALS:
        for scale in SCALES:
            result = normals(verts * scale, faces)
            assert np.isfinite(result).all()
            lengths = np.linalg.norm(result, axis=1)
            assert lengths[isolated] == 0.0
            assert np.allclose(np.delete(lengths, isolated), 1.0, atol=1e-12), (normals, scale)


def test_sliver_fades_out():
    verts, faces, sliver, _ = curved_grid()
    corners = [sliver[0], sliver[2]]
    for scale in SCALES:
        with_sliver = mesh_vertex_normals(verts * scale, faces)[corners]
        without = mesh_vertex_normals(verts * scale, faces[:-1])[corners]
        assert angles_deg(with_sliver, without).max() < 0.1, scale


def test_mm_mesh_as_before():
    """On a mesh of 85 mm diagonal, the normals of the old absolute 1e-6 mm^2 epsilon."""
    verts, faces, _, isolated = curved_grid()
    raw = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    face_n = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-6)
    before = np.zeros(verts.shape)
    for i in range(3):
        np.add.at(before, faces[:, i], face_n)
    used = np.arange(len(verts)) != isolated
    before = before[used] / np.linalg.norm(before[used], axis=1, keepdims=True)
    assert angles_deg(mesh_vertex_normals(verts, faces)[used], before).max() < 1e-3


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
