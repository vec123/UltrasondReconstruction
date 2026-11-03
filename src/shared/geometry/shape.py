"""A mesh as torch tensors, with its point fields, and its reader."""

import os
from dataclasses import dataclass
from typing import Optional

import torch

from src.shared.vtk.extract import extract_all_vtp_point_fields, extract_vtp_point_fields, extract_vtp_points_cells
from src.shared.vtk.io import load_obj, load_vtp


@dataclass
class Shape:
    """A shape with points, faces, and optional weights and normals.

    Attributes:
        points: [1, N, 3] point cloud in normalized coordinates.
        faces: [F, 3] triangular face indices.
        weights: [1, N] per-point weights (e.g., area) or None (caller uses uniform).
        normals: [1, N, 3] per-point normals or None (caller computes if needed).
        point_fields: {name: numpy array} of every OTHER point array the file carried
            -- heatmaps, mode scores, labels -- or None. Passenger data: the model
            never reads it, it is kept so the exporters can put it back on the
            deformed shape, where it is still valid because a deformation moves the
            points without reordering or resampling them. numpy rather than torch,
            and uncast, so an int label array stays int.
    """

    points: torch.Tensor  # [1, N, 3]
    faces: torch.Tensor  # [F, 3]
    weights: Optional[torch.Tensor] = None  # [1, N] or None
    normals: Optional[torch.Tensor] = None  # [1, N, 3] or None
    point_fields: Optional[dict] = None  # {name: [N] or [N, C] numpy} or None
    # [1, N] or None: the weight the OTHER cloud's points take at their nearest point of
    # this one (data term source_to_target.inherit_target_field: <field name>).
    match_weights: Optional[torch.Tensor] = None


def load_shape(path):
    """Load a shape from a VTP or OBJ file.

    Extracts points, faces, and optional point fields (area, normal).
    Missing fields return None (caller handles uniform fallback).

    Args:
        path: path to .vtp or .obj file

    Returns:
        Shape with points [1,N,3], faces [F,3], weights/normals or None
    """
    # Detect format by extension
    _, ext = os.path.splitext(path.lower())

    if ext == ".vtp":
        poly = load_vtp(path)
    elif ext == ".obj":
        poly = load_obj(path)
    else:
        raise ValueError(f"Unsupported format: {ext}. Supported: .vtp, .obj")

    points_np, faces_np = extract_vtp_points_cells(poly)
    fields = extract_vtp_point_fields(poly, ["area", "normal"])

    # Reshape points to [1, N, 3]
    # The run's precision, not a hardcoded one: everything downstream derives its
    # dtype from these tensors, so this is where cfg.dtype actually takes effect.
    points = torch.tensor(points_np, dtype=torch.get_default_dtype()).unsqueeze(0)

    # Reshape faces to [F, 3]
    faces = torch.tensor(faces_np, dtype=torch.int64)

    # Extract weights from "area" field (or None if absent)
    weights = None
    if fields["area"] is not None:
        weights = torch.tensor(fields["area"],
                               dtype=torch.get_default_dtype()).unsqueeze(0)

    # Extract normals (or None if absent)
    normals = None
    if fields["normal"] is not None:
        normals = torch.tensor(fields["normal"],
                               dtype=torch.get_default_dtype()).unsqueeze(0)

    # Everything else the file carried, kept verbatim so the exporters can put it
    # back on the deformed shape. "area"/"normal" are in here too and that is
    # deliberate: they are still point data, and a template exported without them
    # would have lost arrays its input had.
    point_fields = extract_all_vtp_point_fields(poly) if ext == ".vtp" else {}

    return Shape(points=points, faces=faces, weights=weights, normals=normals,
                 point_fields=point_fields)
