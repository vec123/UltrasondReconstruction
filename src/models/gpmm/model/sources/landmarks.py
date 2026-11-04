"""Landmark annotation file: name -> mean-shape vertex index.

The BabyFM .mat carries its landmarks inline (``landmark_verts`` /
``landmark_names``); a Deformetrica atlas carries none at all - see the module
docstring of :mod:`src.model.deformetrica`. This is the file format that fills
that gap: a small JSON object mapping landmark name to a 0-based index into the
model's mean shape, written interactively by ``scripts/place_landmarks.py`` and
read back by :func:`src.model.deformetrica.load_atlas_ssm`, so an atlas model
attaches to every landmark-driven field - ``mean_shape.py``'s
``landmark_distance`` / ``landmark_heatmaps``, ``align_landmarks.py``, the NICP
landmark fit - exactly the way the .mat model's built-in landmarks already do.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Mapping


def read_landmark_map(path) -> "OrderedDict[str, int]":
    """name -> vertex index, in file order. Empty when ``path`` does not exist."""
    path = Path(path)
    if not path.is_file():
        return OrderedDict()

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object of name -> vertex index, got {type(data).__name__}")

    return OrderedDict((str(name), int(index)) for name, index in data.items())


def write_landmark_map(path, landmarks: Mapping[str, int]) -> None:
    """Writes name -> vertex index as JSON, in the order ``landmarks`` iterates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({str(name): int(index) for name, index in landmarks.items()}, handle, indent=2)
        handle.write("\n")
