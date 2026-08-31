"""Build a ``target_weight`` point field on a TARGET scan, combining an
existing per-point RELIABILITY field with proximity to annotated landmarks,
measured GEODESICALLY along the scan's own mesh.

Distinct from ``landmark_weight_field.py`` in three ways:

  * Landmark POSITIONS come from the annotated ``landmarks:`` CSVs
    (``landmarks.load_annotations``/``aggregate`` -- the per-label mean across
    every file naming that label), not the template's ``heatmap_<label>``
    fields. The CSVs and the scan are already in the same (world) frame, so no
    resolution step beyond that mean is needed.

  * Distance is GEODESIC, along the scan's own face connectivity, not
    Euclidean. A merged multi-sweep ultrasound scan can fold two points close
    in straight-line space across a gap the actual surface does not cross
    (opposite sides of the nose, say) -- Euclidean proximity would leak
    importance across that gap; geodesic proximity cannot.

  * The landmark-proximity term is MULTIPLIED by an existing reliability
    field (``consistency_score`` by default) rather than being the whole
    answer: "consistent AND close is high, consistent AND far is lower,
    inconsistent stays low regardless of proximity" -- a point near a
    landmark does not borrow the landmark's importance if it is not itself
    trustworthy.

The ``include``-parsing and per-landmark-gaussian-combination machinery is
shared with the source-side sibling via ``landmark_fields.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from src.resnet_lddmm.resnet_flow.config import reject_unknown_keys
from src.resnet_lddmm.resnet_flow.io import Shape
from src.resnet_lddmm.resnet_flow.landmarks import aggregate, load_annotations
from src.weight_field.landmark_fields import (
    AXES,
    DEFAULT_DECAY_SCALE,
    DEFAULT_MAX_WEIGHT,
    LandmarkEntry,
    axis_bounds,
    combine_peaked_gaussians,
    mahalanobis_norm,
    parse_include_entries,
)

DEFAULT_CONSISTENCY_FIELD = "consistency_score"
DEFAULT_OUTPUT_FIELD = "target_weight"

_ALLOWED_KEYS = {"vtp", "landmarks", "include", "decay_scale", "max_weight",
                 "consistency_field", "min_weight", "output_field", "output"}


@dataclass
class TargetWeightFieldCfg:
    """One run of ``scripts/weight_fields/write_target_weight_field.py``.

    Attributes:
        vtp: target scan carrying face connectivity and ``consistency_field``
            (and the one this run writes ``output_field`` onto).
        landmarks: folder of Slicer fiducial CSVs (see
            ``landmarks.load_annotations``) -- landmark positions are the
            per-label MEAN across every file that names that label, in the
            scan's own (world) frame.
        entries: landmarks to use, each with its own ``decay_scale``/
            ``max_weight`` (see ``landmark_fields.LandmarkEntry``). Whatever
            label is NOT here is simply never consulted.
        consistency_field: name of the existing per-point reliability field to
            multiply the landmark-proximity term by.
        min_weight: floor applied to the FINAL
            ``consistency_field * proximity`` product, so nothing is ever
            weighted to exactly zero.
        output_field: name of the point field this run writes.
        output: None = overwrite ``vtp`` in place.
    """

    vtp: str
    landmarks: str
    entries: List[LandmarkEntry]
    consistency_field: str = DEFAULT_CONSISTENCY_FIELD
    min_weight: float = 0.0
    output_field: str = DEFAULT_OUTPUT_FIELD
    output: Optional[str] = None

    @property
    def include(self) -> List[str]:
        """Labels only, in ``entries`` order -- for display and lookup."""
        return [entry.label for entry in self.entries]

    @classmethod
    def from_dict(cls, d: dict) -> "TargetWeightFieldCfg":
        d = dict(d)
        reject_unknown_keys(d, _ALLOWED_KEYS, what="target weight field config")

        if not d.get("vtp"):
            raise ValueError("target weight field config needs a non-empty 'vtp'")
        if not d.get("landmarks"):
            raise ValueError("target weight field config needs a non-empty 'landmarks' folder")

        raw_include = d.get("include")
        if not raw_include:
            raise ValueError(
                "target weight field config needs a non-empty 'include' list "
                "of landmark labels"
            )

        default_decay_scale = float(d.get("decay_scale", DEFAULT_DECAY_SCALE))
        if default_decay_scale <= 0:
            raise ValueError(f"decay_scale must be > 0, got {default_decay_scale}")

        default_max_weight = float(d.get("max_weight", DEFAULT_MAX_WEIGHT))
        if default_max_weight <= 0:
            raise ValueError(f"max_weight must be > 0, got {default_max_weight}")

        entries = parse_include_entries(raw_include, default_decay_scale, default_max_weight)

        min_weight = float(d.get("min_weight", cls.min_weight))
        if min_weight < 0.0:
            raise ValueError(f"min_weight must be >= 0, got {min_weight}")

        return cls(
            vtp=d["vtp"],
            landmarks=d["landmarks"],
            entries=entries,
            consistency_field=d.get("consistency_field", cls.consistency_field),
            min_weight=min_weight,
            output_field=d.get("output_field", cls.output_field),
            output=d.get("output"),
        )


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    """Unique undirected edges (as ``[E, 2]`` vertex-index pairs) from ``faces``.

    Topology only -- no weights yet, so this is invariant to which landmark
    (and therefore which ``axis_bounds``) ends up using it, and is computed
    exactly ONCE regardless of how many distinct anisotropies ``entries``
    needs (see ``geodesic_distance_to_landmarks``).
    """
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], axis=0)
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def _directed_weighted_graph(points: np.ndarray, edges: np.ndarray, bounds):
    """Directed, ``mahalanobis_norm``-weighted sparse adjacency for one set of ``bounds``.

    Isotropic ``bounds`` (every axis's neg == pos) make traversing an edge
    cost the same either way, reducing exactly to the old plain-Euclidean/
    ``decay_scale`` graph. A DIRECTIONAL bound (e.g. a larger ``pos`` on "y")
    makes crossing that edge cheaper one way than the other -- e.g. stepping
    upward costs less than stepping downward -- so the two directions of the
    SAME edge get two independently computed weights, and the graph is no
    longer symmetric. Every edge from ``_mesh_edges`` still gets both
    directions, so reachability is unaffected; only the cost of each is.
    """
    p_i = points[edges[:, 0]]
    p_j = points[edges[:, 1]]
    fwd = mahalanobis_norm(torch.tensor(p_j - p_i, dtype=torch.float32), bounds).numpy()
    bwd = mahalanobis_norm(torch.tensor(p_i - p_j, dtype=torch.float32), bounds).numpy()

    n = points.shape[0]
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([fwd, bwd])
    return coo_matrix((data, (row, col)), shape=(n, n)).tocsr()


def _nearest_vertex(points: np.ndarray, position: np.ndarray) -> int:
    """Index of the mesh vertex closest to an arbitrary 3D ``position``.

    CSV annotations are free-floating 3D points, not necessarily exactly on
    the mesh -- geodesic distance needs an actual mesh vertex to start from.
    """
    return int(np.argmin(np.linalg.norm(points - position[None, :], axis=1)))


def geodesic_distance_to_landmarks(shape: Shape, entries: List[LandmarkEntry],
                                   landmark_positions: np.ndarray) -> torch.Tensor:
    """``[N, L]`` NORMALISED geodesic distance, along ``shape``'s own mesh,
    from every vertex to each of ``entries``' landmarks.

    Each landmark position is first snapped to its nearest mesh VERTEX (see
    ``_nearest_vertex``), then that vertex is the Dijkstra source. Distance is
    accumulated edge-by-edge through ``mahalanobis_norm`` (see
    ``_directed_weighted_graph``), so the result is already in "how many
    decay_scales away" units -- ``combine_peaked_gaussians`` no longer divides
    by anything. A vertex unreachable from a given landmark (a disconnected
    mesh component) comes back as ``+inf``, which later decays to a weight of
    exactly 0 -- not NaN.

    Landmarks are grouped by IDENTICAL ``axis_bounds`` so that sharing a
    ``decay_scale``/``decay_axes`` (the common case) reuses one graph build
    and one multi-source Dijkstra call; only a genuinely distinct anisotropy
    pays for its own graph.

    Args:
        shape: target Shape with ``.points`` [1,N,3] and ``.faces`` [F,3].
        entries: L ``LandmarkEntry``, in the order ``landmark_positions`` is
            indexed.
        landmark_positions: ``[L, 3]`` array, same order as ``entries``.

    Returns:
        ``[N, L]`` float32 tensor.
    """
    points = shape.points[0].detach().cpu().numpy()
    faces = shape.faces.detach().cpu().numpy()
    edges = _mesh_edges(faces)

    source_vertices = [_nearest_vertex(points, landmark_positions[i])
                       for i in range(len(entries))]

    groups: Dict[Tuple, List[int]] = {}
    for i, entry in enumerate(entries):
        bounds = axis_bounds(entry)
        key = tuple(bounds[axis] for axis in AXES)
        groups.setdefault(key, []).append(i)

    columns = [None] * len(entries)
    for key, idxs in groups.items():
        bounds = dict(zip(AXES, key))
        graph = _directed_weighted_graph(points, edges, bounds)
        sub_sources = [source_vertices[i] for i in idxs]
        dist = dijkstra(graph, directed=True, indices=sub_sources)  # [len(idxs), N]
        for k, i in enumerate(idxs):
            columns[i] = dist[k]

    distances = np.stack(columns, axis=1)  # [N, L]
    return torch.tensor(distances, dtype=torch.float32)


def compute_target_weight_field(shape: Shape, cfg: TargetWeightFieldCfg) -> torch.Tensor:
    """Per-vertex weight: ``consistency_field`` * (max over included
    landmarks of that landmark's own peaked gaussian GEODESIC decay), floored.

    Multiplicative, not additive: a point that is close to a landmark but
    itself unreliable does not inherit the landmark's importance, and a point
    that is highly reliable but far from every included landmark decays
    toward ``min_weight`` regardless of how trustworthy it is.

    Args:
        shape: the target Shape carrying ``cfg.consistency_field`` and face
            connectivity.
        cfg: see ``TargetWeightFieldCfg``.

    Returns:
        ``[N]`` tensor, one weight per vertex.

    Raises:
        ValueError: ``cfg.consistency_field`` is missing or the wrong length,
            or an included label has no annotated position in ``cfg.landmarks``.
    """
    point_fields = shape.point_fields or {}
    if cfg.consistency_field not in point_fields:
        raise ValueError(
            f"consistency_field {cfg.consistency_field!r} not found on {cfg.vtp!r}; "
            f"available fields: {sorted(point_fields)}"
        )
    consistency = torch.tensor(point_fields[cfg.consistency_field], dtype=torch.float32)
    n_vertices = shape.points.shape[1]
    if consistency.shape[0] != n_vertices:
        raise ValueError(
            f"consistency_field {cfg.consistency_field!r} has {consistency.shape[0]} "
            f"entries but the mesh has {n_vertices} vertices"
        )

    annotations = load_annotations(cfg.landmarks)
    labels, mean_positions, _ = aggregate(annotations)
    position_by_label = dict(zip(labels, mean_positions))

    missing = [entry.label for entry in cfg.entries if entry.label not in position_by_label]
    if missing:
        raise ValueError(
            f"landmark(s) {missing} not found in {cfg.landmarks!r}; "
            f"available: {sorted(labels)}"
        )
    landmark_positions = np.stack([position_by_label[entry.label] for entry in cfg.entries])

    distance = geodesic_distance_to_landmarks(shape, cfg.entries, landmark_positions)  # [N, L]
    proximity = combine_peaked_gaussians(distance, cfg.entries)  # [N]

    weight = consistency * proximity
    return weight.clamp_min(cfg.min_weight)
