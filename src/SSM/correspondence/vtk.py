
from pathlib import Path

import numpy as np
import vtk

from src.SSM.correspondence.base import Correspondences
from src.vtk.io import load_vtp, save_vtp
from src.vtk.fields import add_cell_field, add_point_field
from src.vtk.create import create_polydata_w_lines


def save_matching_lines(src_verts, tgt_verts, matches, filename="matching_field.vtp"):
    """
    matches: (N, 3) array of (src_idx, tgt_idx, distance) rows, the format
    Correspondences.as_array() returns (a Correspondences works too).

    Writes one line per match, plus two fields to color/filter them in
    ParaView: cell data 'distance' (the match distance, so bad correspondences
    stand out) and point data 'endpoint' (0 at the source/model end, 1 at the
    target/scan end, so the two ends of a line can be told apart or glyphed
    separately).
    """
    matches = np.asarray(matches)

    src_np = src_verts.cpu().numpy() if hasattr(src_verts, 'cpu') else src_verts
    tgt_np = tgt_verts.cpu().numpy() if hasattr(tgt_verts, 'cpu') else tgt_verts

    src_idx = matches[:, 0].astype(int)
    tgt_idx = matches[:, 1].astype(int)
    distance = matches[:, 2]

    # Extract only the points that have valid matches
    src_matched = src_np[src_idx]
    tgt_matched = tgt_np[tgt_idx]

    # Create a line for each match
    # Paraview expects points and cells
    all_points = np.vstack([src_matched, tgt_matched])
    n_matches = len(matches)

    # Line indices: [0, n, 1, n+1, 2, n+2, ...]
    lines = np.column_stack([np.arange(n_matches), np.arange(n_matches) + n_matches])

    vtp = create_polydata_w_lines(all_points, lines)
    add_cell_field(vtp, distance, "distance")
    endpoint = np.concatenate([np.zeros(n_matches), np.ones(n_matches)])
    add_point_field(vtp, endpoint, "endpoint")
    save_vtp(vtp, filename)


def prefixed_path(path, prefix: str) -> Path:
    """``path`` with ``prefix`` put before its file name, in the same folder."""
    path = Path(path)
    return path.with_name(prefix + path.name)


def _as_correspondences(matches) -> Correspondences:
    if isinstance(matches, Correspondences):
        return matches
    return Correspondences.from_triples(np.asarray(matches).reshape(-1, 3))


def _copy_polydata(mesh) -> vtk.vtkPolyData:
    """A deep copy of a vtkPolyData, or the contents of a .vtp path, with all its fields."""
    source = load_vtp(str(mesh)) if isinstance(mesh, (str, Path)) else mesh
    copy = vtk.vtkPolyData()
    copy.DeepCopy(source)
    return copy


def _best_partner(own_idx, distance, weight, n_points):
    """For each point of one side: the index of the pair it keeps, or -1.

    A point with several pairs (many-to-one matching, Sinkhorn) keeps the pair
    with the largest weight, and of those the closest.
    """
    order = np.lexsort((distance, -weight))
    own_sorted = own_idx[order]
    _, first = np.unique(own_sorted, return_index=True)
    best = np.full(n_points, -1, dtype=np.int64)
    best[own_sorted[first]] = order[first]
    return best


def correspondence_point_data(own_idx, partner_idx, distance, weight, n_points, partner_name):
    """The correspondence data of one side, as ``{field name: (n_points,) array}``.

    ``partner_name`` is the partner index (-1 where there is none), and
    ``correspondence_weight`` the weight of that same pair (0 where there is none).
    """
    best = _best_partner(own_idx, distance, weight, n_points)
    has = best >= 0
    partner = np.full(n_points, -1, dtype=np.int64)
    partner[has] = partner_idx[best[has]]
    pair_weight = np.zeros(n_points)
    pair_weight[has] = weight[best[has]]
    return {
        partner_name: partner,
        "correspondence_weight": pair_weight,
    }


def save_correspondence_field(recon, target, matches, target_path, recon_path=None):
    """Writes the correspondence data onto a copy of the target, and of the recon.

    recon, target: vtkPolyData, or a .vtp path. Each is copied with all its
        fields, so the inputs are not changed.
    matches: Correspondences with source = recon and target = target, or the
        (N, 3) (recon_idx, target_idx, distance) rows of as_array().
    target_path: where the target copy goes. Give the target's own path to
        overwrite it, or prefixed_path(target, prefix) to keep it.
    recon_path: the same for the recon; None does not write the recon.

    Two fields on each side:

    correspondence_id      the identity of the point, as a recon vertex index.
                           On the recon it is the vertex's own index; on the
                           target it is the index of its recon partner, NaN
                           where there is none. Color both with the same range,
                           0 to n_recon - 1, and corresponding points get the
                           same color. It is the active scalar field.
    correspondence_weight  the mask: 0 where a point has no partner, else that
                           pair's weight (1 for the heuristic methods, the
                           transported mass for Sinkhorn).

    Returns the written paths.
    """
    matches = _as_correspondences(matches)
    recon_poly = _copy_polydata(recon)
    target_poly = _copy_polydata(target)
    n_recon, n_target = recon_poly.GetNumberOfPoints(), target_poly.GetNumberOfPoints()

    recon_data = correspondence_point_data(matches.source_idx, matches.target_idx, matches.distance,
                                           matches.weight, n_recon, "target_point_id")
    target_data = correspondence_point_data(matches.target_idx, matches.source_idx, matches.distance,
                                            matches.weight, n_target, "recon_vertex_id")

    target_id = target_data["recon_vertex_id"].astype(np.float64)
    target_id[target_id < 0] = np.nan
    sides = [(target_poly, target_data["correspondence_weight"], target_id, target_path)]
    if recon_path is not None:
        sides.append((recon_poly, recon_data["correspondence_weight"],
                      np.arange(n_recon, dtype=np.float64), recon_path))

    written = []
    for poly, weight, identity, path in sides:
        add_point_field(poly, weight, "correspondence_weight")
        add_point_field(poly, identity, "correspondence_id")
        save_vtp(poly, str(path))
        written.append(Path(path))
    return written
