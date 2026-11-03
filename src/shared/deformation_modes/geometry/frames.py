"""Orthonormal local frames at surface points: tangent, bitangent, normal.

A frame is a (3, 3) rotation whose *columns* are the axes, the convention
IMODAL's implicit order-1 modules use for their ``R_i``: a strain prescribed
as ``R diag(s) R^T`` stretches by ``s[0]`` along the tangent, ``s[1]`` along
the bitangent and ``s[2]`` along the normal.
"""

from __future__ import annotations

import numpy as np

FRAME_AXES = ("tangent", "bitangent", "normal")


def local_frames(normals: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Frames whose normal is the surface normal and whose tangent follows ``reference``.

    Args:
        normals: (n, 3) unit surface normals.
        reference: (3,) or (n, 3) direction the tangent should follow (e.g. a
            direction rule's output); it is projected onto each tangent plane.

    Returns:
        (n, 3, 3) right-handed frames, columns (tangent, bitangent, normal) with
        bitangent = normal x tangent. Where ``reference`` is (nearly) parallel
        to the normal, the tangent falls back to the world axis least aligned
        with that normal, so every frame is defined.
    """
    normals = np.asarray(normals, float)
    normals = normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    reference = np.broadcast_to(np.asarray(reference, float), normals.shape)

    tangent = reference - (reference * normals).sum(-1, keepdims=True) * normals
    length = np.linalg.norm(tangent, axis=-1)
    degenerate = length < 1e-6 * np.maximum(np.linalg.norm(reference, axis=-1), 1e-12)
    if degenerate.any():
        fallback = np.eye(3)[np.argmin(np.abs(normals[degenerate]), axis=-1)]
        n = normals[degenerate]
        tangent[degenerate] = fallback - (fallback * n).sum(-1, keepdims=True) * n
        length = np.linalg.norm(tangent, axis=-1)
    tangent /= length[:, None]
    bitangent = np.cross(normals, tangent)
    return np.stack([tangent, bitangent, normals], axis=-1)
