"""Carrying designed modes onto a posed copy of the template they were drawn on.

Training does not flow the template file itself but a copy of it that has been
scaled, rotated and moved onto the subject (``similarity_transform``) and then
normalised: same vertices, same triangles, another frame. So each mode is
compiled on the template its spec was authored on, the similarity between the
two vertex sets is fitted (exactly - they are the same points), and the
mode's kernel expansion is carried across. Nothing is re-solved in the new
frame, so the field training uses is the field the editor showed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from src.deformation_modes.compile import compile_mode
from src.deformation_modes.io import SpecError, load_modeset
from src.deformation_modes.kernel_field import KernelExpansion
from src.deformation_modes.mesh import load_template_mesh


@dataclass(frozen=True)
class Similarity:
    """``x' = scale * rotation @ x + translation``, and how well it maps one point set onto another."""

    scale: float
    rotation: np.ndarray       # (3, 3), proper (det +1)
    translation: np.ndarray    # (3,)
    rms: float                 # residual of the fit, in target units

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(points, float) @ self.rotation.T + self.translation


def fit_similarity(source: np.ndarray, target: np.ndarray) -> Similarity:
    """Least-squares similarity taking ``source[i]`` to ``target[i]`` (Umeyama 1991)."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"need two (n, 3) point sets in correspondence, got {source.shape} "
                         f"and {target.shape}")
    mu_s, mu_t = source.mean(0), target.mean(0)
    xs, xt = source - mu_s, target - mu_t
    u, d, vt = np.linalg.svd(xt.T @ xs / len(source))
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt)) or 1.0])
    rotation = u @ flip @ vt
    scale = float(np.trace(np.diag(d) @ flip) / (xs ** 2).sum(1).mean())
    translation = mu_t - scale * rotation @ mu_s
    rms = float(np.sqrt(((scale * xs @ rotation.T - xt) ** 2).sum(1).mean()))
    return Similarity(scale, rotation, translation, rms)


@dataclass(frozen=True)
class TransportedMode:
    """One mode's field on the posed mesh, normalised so its peak there is ``unit``."""

    name: str
    expansion: KernelExpansion
    raw_peak: float            # peak |v| on the posed mesh before normalising
    warnings: Tuple[str, ...]


def transport_modes(spec: Path, template: Path, names: Sequence[str], vertices: np.ndarray,
                    unit: float = 1.0, tolerance: float = 1e-4
                    ) -> Tuple[List[TransportedMode], Similarity]:
    """Compile modes ``names`` of ``spec`` on ``template`` and carry them onto ``vertices``.

    Args:
        spec: the mode spec (YAML written by the editor).
        template: the mesh the spec was authored on.
        names: which modes, in order.
        vertices: (N, 3) the posed copy - the template's vertices, in the same
            order, under some similarity transform.
        unit: each field is scaled so its peak over ``vertices`` is this, so a
            mode's amplitude reads as peak speed in the units ``unit`` stands for.
        tolerance: largest acceptable fit residual, as a fraction of the posed
            mesh's size.

    Returns:
        (one TransportedMode per name, the fitted similarity template -> posed)

    Raises:
        SpecError: the spec does not belong to ``template``, or names an unknown mode.
        ValueError: ``vertices`` is not a similarity copy of the template.
    """
    mesh = load_template_mesh(template)
    modeset = load_modeset(spec, mesh)
    vertices = np.asarray(vertices, float)
    if len(vertices) != mesh.n_vertices:
        raise ValueError(f"the posed mesh has {len(vertices)} vertices, the template "
                         f"{Path(template).name} {mesh.n_vertices}: modes can only be carried "
                         f"onto a posed copy of the template they were drawn on")
    similarity = fit_similarity(mesh.vertices, vertices)
    size = float(np.linalg.norm(vertices - vertices.mean(0), axis=1).max())
    if similarity.rms > tolerance * size:
        raise ValueError(f"the posed mesh is not a similarity copy of {Path(template).name}: "
                         f"fit residual {similarity.rms:.3g} vs size {size:.3g}. Modes are "
                         f"carried by a scale + rotation + shift only, not onto a reshaped "
                         f"mesh (e.g. an SSM fit).")
    unknown = [name for name in names if not modeset.has("modes", name)]
    if unknown:
        raise SpecError(f"{Path(spec).name} has no mode(s) {unknown}; it has "
                        f"{[m.name for m in modeset.modes]}")

    transported = []
    for name in names:
        compiled = compile_mode(modeset, name, mesh)
        field = compiled.kernel.expansion().transported(similarity.scale, similarity.rotation,
                                                         similarity.translation)
        peak = float(np.linalg.norm(field(vertices), axis=1).max())
        if peak <= 0:
            raise ValueError(f"mode {name!r} compiles to a zero field - nothing to move")
        transported.append(TransportedMode(name, field.scaled(unit / peak), peak,
                                           tuple(compiled.warnings)))
    return transported, similarity
