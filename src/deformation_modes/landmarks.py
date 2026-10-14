"""Landmark vertices of a template, read from its ``heatmap_<label>`` point fields.

Same convention as ``src.landmarks.vtk_landmarks.heatmap_landmark_vertex_indices``
and ``src.resnet_lddmm.resnet_flow.landmarks``: a template does not store a
landmark list, it carries one heatmap per label, and the landmark is that
heatmap's peak (argmax) vertex. This variant works on point fields already in
memory instead of re-reading a file.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

HEATMAP_PREFIX = "heatmap_"
#: The superposition of every label's heatmap - not itself a landmark.
COMBINED_HEATMAP = "heatmap_landmarks"


def landmark_vertices(point_fields: Mapping[str, np.ndarray],
                      prefix: str = HEATMAP_PREFIX) -> Dict[str, int]:
    """{label: vertex index} - the peak vertex of each ``<prefix><label>`` scalar field.

    The combined ``heatmap_landmarks`` field and non-scalar arrays are skipped.
    An empty dict means the mesh carries no landmark heatmaps.
    """
    found: Dict[str, int] = {}
    for name, values in point_fields.items():
        if not name.startswith(prefix) or name == COMBINED_HEATMAP:
            continue
        values = np.asarray(values)
        if values.ndim != 1:
            continue
        found[name[len(prefix):]] = int(np.argmax(values))
    return found
