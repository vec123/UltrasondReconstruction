"""From a :class:`~.spec.Mode` to its generated field: resolve, sample, constrain, solve.

The one place a mode's meaning is turned into numbers, so the editor's preview
and (later) training produce the same field from the same spec.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from src.deformation_modes.frames import local_frames
from src.deformation_modes.kernel_field import (
    KernelField, anchors, displacement, rigid, sliding, strain, thin,
)
from src.deformation_modes.mesh import TemplateMesh
from src.deformation_modes.spec import ConstraintSet, Mode, ModeSet, Selection

Resolver = Callable[[Selection], np.ndarray]


def mode_signature(modeset: ModeSet, mode: Mode) -> str:
    """A hash of everything the mode's field depends on: itself, its axis, its sets, the defaults.

    Two specs with the same signature produce the same field, so a compiled
    field stays valid exactly as long as its signature does.
    """
    from src.deformation_modes.io import modeset_to_dict
    whole = modeset_to_dict(modeset)
    used = {
        "mode": next(m for m in whole["modes"] if m["name"] == mode.name),
        "axis": next((a for a in whole["axes"] if a["name"] == mode.axis), None),
        "sets": [c for c in whole["constraint_sets"] if c["name"] in mode.constraints],
        "defaults": whole["defaults"],
    }
    return hashlib.sha1(json.dumps(used, sort_keys=True).encode()).hexdigest()


def is_symmetric(mesh: TemplateMesh, indices: np.ndarray, tolerance: float = 0.95) -> bool:
    """Whether a vertex set is (nearly) closed under mirroring."""
    if len(indices) == 0:
        return False
    try:
        mirrored = mesh.mirror.index[indices]
    except ValueError:                                  # no bilateral landmarks: no midline
        return False
    return float(np.isin(mirrored, indices).mean()) >= tolerance


def sample(mesh: TemplateMesh, indices: np.ndarray, spacing: float) -> np.ndarray:
    """``thin`` at ``spacing``, symmetrically whenever the set itself is symmetric."""
    return thin(indices, mesh.vertices, spacing,
                mirror=mesh.mirror if is_symmetric(mesh, indices) else None)


@dataclass
class SitePreview:
    """A mode's sampled sites and what they prescribe - cheap, no solve."""

    sites: np.ndarray                       # vertex indices (sampled)
    vectors: Optional[np.ndarray] = None    # displacement: (n, 3)
    frames: Optional[np.ndarray] = None     # strain: (n, 3, 3)


def mode_sites(modeset: ModeSet, mode: Mode, mesh: TemplateMesh,
               resolve: Optional[Resolver] = None) -> SitePreview:
    """Sample a mode's sites and evaluate its axis there.

    Raises:
        ValueError: the mode has no sites or no axis.
        KeyError: its axis does not exist.
    """
    resolve = resolve or (lambda selection: selection.resolve(mesh))
    all_sites = resolve(mode.sites)
    if len(all_sites) == 0:
        raise ValueError(f"mode {mode.name!r} has no sites - select some first")
    if not mode.axis:
        raise ValueError(f"mode {mode.name!r} has no axis")
    sigma = mode.sigma or modeset.sigma
    sites = sample(mesh, all_sites, modeset.spacing * sigma)
    directions = modeset.axis(mode.axis).rule.evaluate(mesh, sites)
    if mode.kind == "displacement":
        return SitePreview(sites, vectors=directions * mode.magnitude)
    return SitePreview(sites, frames=local_frames(mesh.normals[sites], directions))


@dataclass
class CompiledMode:
    name: str
    signature: str
    field: np.ndarray                         # (N, 3) on the mesh vertices
    kernel: KernelField
    preview: SitePreview
    constraint_points: Dict[str, np.ndarray]  # set name -> sampled vertex indices
    warnings: List[str] = field(default_factory=list)
    seconds: float = 0.0


def compile_mode(modeset: ModeSet, name: str, mesh: TemplateMesh,
                 resolve: Optional[Resolver] = None) -> CompiledMode:
    """Solve mode ``name`` and evaluate its field on every vertex of ``mesh``.

    Constraint-set vertices that coincide with the mode's own sites are left
    out of that set (a vertex cannot both move and stay put), and a set that
    comes within sigma of the sites is reported, since the two then fight over
    the same smooth field.

    Raises:
        ValueError / KeyError: see :func:`mode_sites`, or an unknown constraint set.
    """
    started = time.perf_counter()
    resolve = resolve or (lambda selection: selection.resolve(mesh))
    mode = modeset.mode(name)
    sigma = mode.sigma or modeset.sigma
    spacing = modeset.spacing * sigma
    preview = mode_sites(modeset, mode, mesh, resolve)
    points = mesh.vertices[preview.sites]

    if mode.kind == "displacement":
        constraints = [displacement(points, preview.vectors, label=f"{name}: sites")]
    else:
        constraints = [strain(points, preview.frames, mode.stretch, label=f"{name}: sites")]

    all_sites = resolve(mode.sites)
    warnings: List[str] = []
    constraint_points: Dict[str, np.ndarray] = {}
    for set_name in mode.constraints:
        cset: ConstraintSet = modeset.constraint_set(set_name)
        region = resolve(cset.selection)
        overlap = np.intersect1d(region, all_sites)
        region = np.setdiff1d(region, all_sites)
        if len(overlap):
            warnings.append(f"{set_name}: {len(overlap)} vertices are also sites - left out of the set")
        if len(region) == 0:
            warnings.append(f"{set_name}: empty - ignored")
            continue
        near = _closer_than(mesh.vertices[region], mesh.vertices[preview.sites], sigma)
        if near > 0.02:
            warnings.append(f"{set_name}: {near:.0%} of it lies within sigma of the sites - "
                            f"the two may conflict (overshoot); move it away or lower sigma")
        chosen = sample(mesh, region, spacing)
        constraint_points[set_name] = chosen
        label = f"{name}: {set_name}"
        if cset.kind == "anchor":
            constraints.append(anchors(mesh.vertices[chosen], cset.weight, label))
        elif cset.kind == "slide":
            constraints.append(sliding(mesh.vertices[chosen], mesh.normals[chosen], cset.weight, label))
        else:
            constraints.append(rigid(mesh.vertices[chosen], cset.weight, label))

    kernel = KernelField(sigma, constraints, nu=modeset.nu)
    values = kernel(mesh.vertices)
    if kernel.max_violation > 0.05 * max(1.0, abs(mode.magnitude)):
        warnings.append(f"constraints met only to {kernel.max_violation:.2g} - "
                        f"they conflict somewhere; check the warnings above")
    return CompiledMode(name, mode_signature(modeset, mode), values, kernel, preview,
                        constraint_points, warnings, time.perf_counter() - started)


def _closer_than(points: np.ndarray, targets: np.ndarray, distance: float) -> float:
    """Fraction of ``points`` within ``distance`` of any of ``targets``."""
    from scipy.spatial import cKDTree
    nearest, _ = cKDTree(targets).query(points)
    return float((nearest < distance).mean()) if len(points) else 0.0
