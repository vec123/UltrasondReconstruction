
import numpy as np
from src.vtk.io import save_vtp
from src.vtk.fields import add_cell_field, add_point_field
from src.vtk.create import create_polydata_w_lines

def save_matching_field(src_verts, tgt_verts, matches, filename="matching_field.vtp"):
    """
    matches: (N, 3) array of (src_idx, tgt_idx, distance) rows, the format
    Correspondences.as_array() returns.

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
