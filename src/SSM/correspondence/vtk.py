
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from src.SSM.correspondence.correspondence import Correspondences
from src.vtk.io import load_vtp, save_vtp
from src.vtk.fields import add_cell_field, add_point_field, add_point_fields
from src.vtk.create import create_polydata_w_lines

#: colour of a point without a partner, so missing coverage is visible
UNMATCHED_RGB = 0.5


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


def correspondence_colormap(points):
    """RGB in [0, 1] from each point's position in its own bounding box.

    A smooth, deterministic colormap: nearby points get similar colors. Used
    to color two meshes so that corresponding points share a color -- bad
    matches then show up as a visible color mismatch between the two renders.
    """
    points = points.detach().cpu().numpy() if hasattr(points, 'detach') else np.asarray(points)

    low = points.min(axis=0)
    span = points.max(axis=0) - low
    span = np.where(span > 1e-12, span, 1.0)

    return (points - low) / span


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
    """For each point of one side: the pair it keeps, or -1, and its number of pairs.

    A point with several pairs (many-to-one matching, Sinkhorn) keeps the pair
    with the largest weight, and of those the closest.
    """
    order = np.lexsort((distance, -weight))
    own_sorted = own_idx[order]
    _, first = np.unique(own_sorted, return_index=True)
    best = np.full(n_points, -1, dtype=np.int64)
    best[own_sorted[first]] = order[first]
    count = np.bincount(own_idx, minlength=n_points).astype(np.int64)
    return best, count


def correspondence_point_data(own_idx, partner_idx, distance, weight, n_points, partner_name):
    """The correspondence data of one side, as ``{field name: (n_points,) array}``.

    ``partner_name`` is the partner index (-1 where there is none). The
    ``correspondence_distance`` and ``correspondence_weight`` belong to that
    same pair (NaN and 0 where there is none), and ``correspondence_count`` is
    the number of pairs the point is in.
    """
    best, count = _best_partner(own_idx, distance, weight, n_points)
    has = best >= 0
    partner = np.full(n_points, -1, dtype=np.int64)
    partner[has] = partner_idx[best[has]]
    pair_distance = np.full(n_points, np.nan)
    pair_distance[has] = distance[best[has]]
    pair_weight = np.zeros(n_points)
    pair_weight[has] = weight[best[has]]
    return {
        partner_name: partner,
        "correspondence_distance": pair_distance,
        "correspondence_weight": pair_weight,
        "correspondence_count": count,
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

    Target fields: ``recon_vertex_id`` and the per-pair data of
    correspondence_point_data(), plus ``correspondence_rgb``: the colour of the
    recon partner, gray where there is none. The recon gets the same data with
    ``target_point_id``, and its own correspondence_colormap() as
    ``correspondence_rgb``. Equal colours then mark corresponding points.

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

    recon_points = vtk_to_numpy(recon_poly.GetPoints().GetData())
    recon_rgb = correspondence_colormap(recon_points)
    target_rgb = np.full((n_target, 3), UNMATCHED_RGB)
    has = target_data["recon_vertex_id"] >= 0
    target_rgb[has] = recon_rgb[target_data["recon_vertex_id"][has]]

    add_point_fields(target_poly, target_data)
    add_point_field(target_poly, target_rgb, "correspondence_rgb")
    save_vtp(target_poly, str(target_path))
    written = [Path(target_path)]

    if recon_path is not None:
        add_point_fields(recon_poly, recon_data)
        add_point_field(recon_poly, recon_rgb, "correspondence_rgb")
        save_vtp(recon_poly, str(recon_path))
        written.append(Path(recon_path))
    return written
