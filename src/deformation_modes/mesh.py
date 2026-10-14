"""TemplateMesh: one triangle mesh plus everything derived from it that modes are specified against."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

from src.deformation_modes.landmarks import landmark_vertices
from src.deformation_modes.symmetry import MidlinePlane, MirrorMap


class TemplateMesh:
    """Vertices, faces and point fields, with derived quantities computed once, on first use.

    Normals, landmark vertices, the midline plane, the mirror map and the
    geodesic-distance solver are all properties of the *same* shape, so they
    live together and are cached here instead of being recomputed (or
    recomputed inconsistently) by each caller. Treat the arrays as read-only:
    a moved shape is a new TemplateMesh.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray,
                 point_fields: Optional[Mapping[str, np.ndarray]] = None,
                 landmarks: Optional[Mapping[str, int]] = None):
        """Wrap a mesh.

        Args:
            vertices: (N, 3).
            faces: (F, 3) triangle vertex indices.
            point_fields: named per-vertex arrays (heatmaps, curvature, weights, ...).
            landmarks: {label: vertex index}; by default read from the
                ``heatmap_<label>`` point fields.

        Raises:
            ValueError: a point field or landmark does not address the vertices.
        """
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.point_fields: Dict[str, np.ndarray] = {}
        for name, values in (point_fields or {}).items():
            values = np.asarray(values)
            if len(values) != len(self.vertices):
                raise ValueError(f"point field {name!r} has {len(values)} entries for "
                                 f"{len(self.vertices)} vertices")
            self.point_fields[name] = values
        self.landmarks: Dict[str, int] = dict(landmarks if landmarks is not None
                                              else landmark_vertices(self.point_fields))
        for name, index in self.landmarks.items():
            if not 0 <= index < len(self.vertices):
                raise ValueError(f"landmark {name!r} at vertex {index} is out of range")

        self._normals: Optional[np.ndarray] = None
        self._midline: Optional[MidlinePlane] = None
        self._mirror: Optional[MirrorMap] = None
        self._geodesics = None

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def centroid(self) -> np.ndarray:
        return self.vertices.mean(axis=0)

    @property
    def radius(self) -> float:
        """Bounding radius about the centroid."""
        return float(np.linalg.norm(self.vertices - self.centroid, axis=1).max())

    # -- derived geometry ----------------------------------------------------

    @property
    def normals(self) -> np.ndarray:
        """(N, 3) unit vertex normals (zero where the incident faces cancel)."""
        if self._normals is None:
            from src.geometry.geometry import mesh_vertex_normals
            self._normals = mesh_vertex_normals(self.vertices, self.faces)
        return self._normals

    def landmark_position(self, name: str) -> np.ndarray:
        """(3,) position of landmark ``name``.

        Raises:
            KeyError: no such landmark, listing the available ones.
        """
        try:
            return self.vertices[self.landmarks[name]]
        except KeyError:
            raise KeyError(f"no landmark {name!r}; available: {sorted(self.landmarks)}") from None

    def landmark_positions(self) -> Dict[str, np.ndarray]:
        return {name: self.vertices[index] for name, index in self.landmarks.items()}

    @property
    def midline(self) -> MidlinePlane:
        """The facial midline plane, fitted to this mesh's landmarks."""
        if self._midline is None:
            self._midline = MidlinePlane.fit(self.landmark_positions())
        return self._midline

    @property
    def mirror(self) -> MirrorMap:
        """Vertex -> mirror vertex across :attr:`midline`."""
        if self._mirror is None:
            self._mirror = MirrorMap(self.vertices, self.midline)
        return self._mirror

    def geodesic_distance(self, seeds) -> np.ndarray:
        """(N,) geodesic distance from the nearest of ``seeds`` (heat method).

        The solver factorises once per mesh (about a second at 31k vertices);
        every later query is a pair of sparse back-substitutions.
        """
        if self._geodesics is None:
            import potpourri3d as pp3d
            self._geodesics = pp3d.MeshHeatMethodDistanceSolver(self.vertices, self.faces)
        seeds = np.atleast_1d(np.asarray(seeds, dtype=np.int64))
        if len(seeds) == 1:
            return self._geodesics.compute_distance(int(seeds[0]))
        return self._geodesics.compute_distance_multisource(seeds.tolist())

    def geodesic_ball(self, seed: int, radius: float) -> np.ndarray:
        """Vertices within geodesic ``radius`` of ``seed`` - a region that never leaks across a fold."""
        return np.flatnonzero(self.geodesic_distance(seed) <= radius)

    def flipped_faces(self, deformed: np.ndarray) -> int:
        """How many triangles a deformation turns over (face normal reversed) - a folding check."""
        from src.geometry.geometry import face_normals
        before = face_normals(self.vertices, self.faces, normalized=False)
        after = face_normals(np.asarray(deformed, float), self.faces, normalized=False)
        return int(((before * after).sum(-1) < 0).sum())


def load_template_mesh(path) -> TemplateMesh:
    """Read a .vtp (triangles + point fields) into a :class:`TemplateMesh`."""
    from src.vtk.extract import extract_all_vtp_point_fields, extract_vtp_points_cells
    from src.vtk.io import load_vtp

    polydata = load_vtp(str(Path(path)))
    vertices, faces = extract_vtp_points_cells(polydata)
    return TemplateMesh(vertices, faces, extract_all_vtp_point_fields(polydata))
