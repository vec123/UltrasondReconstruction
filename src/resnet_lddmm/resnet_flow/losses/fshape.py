"""Functional-varifold ("fshape") data term, windowed to one region of the face.

Chamfer, currents and plain varifolds all compare GEOMETRY only, so they cannot
tell which template tissue ended up where: a nose that stays narrow while the
cheek beside it bulges into the target's wide nostril wing scores as well as a
nose that widens. The fshape framework (Charlier, Charon & Trouve, 2017)
attaches a signal to each shape and compares geometry and signal jointly. Here
the template's signal is its own curvature, computed once and TRANSPORTED with
its tissue: crease-labelled template tissue lying on flat target surface is
penalised directly, which a geometric term never sees.

Each triangle i is a weighted Dirac (centre c_i, unit normal n_i, mass m_i,
signal f_i); two surfaces are compared through

    <S, T> = sum_ij m_i m_j  exp(-|c_i - c_j|^2 / s^2) (n_i . n_j)^2  exp(-(f_i - f_j)^2 / s_f^2)

and ``D = <S,S> - 2<S,T> + <T,T>`` -- a squared RKHS norm (the kernel is a
product of positive-definite kernels), so ``D >= 0`` with equality for identical
fshapes. ``(n . n')^2`` is the UNORIENTED varifold kernel: it does not care how
either mesh is wound. With ``signal: none`` the third factor is dropped and this
is a plain varifold -- the same term, for comparison.

WINDOWED, because the target is a partial frontal scan (~56% of the template's
area): a global varifold would pull every template region the scan never saw.
Both surfaces are weighted by the same smooth ball in SPACE (SmoothBallMask,
centred on the template's own nose landmarks), so the term compares the two
surfaces inside one region and nothing else -- which also keeps it cheap. Each
scale is normalised by its own <T,T>, so the value is a relative distance and
the scales in ``sigmas`` count equally.

Triangles are a fixed, area-proportional random subset (``n_faces`` per mesh),
each carrying ``kappa`` x its area so the subset represents the whole
windowed surface -- a Monte Carlo approximation that is accurate while the
subset's spacing stays well below the smallest kernel width.
"""

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.resnet_lddmm.resnet_flow.fields.local import SmoothBallMask
from src.resnet_lddmm.resnet_flow.losses.terms import LossContext
from src.resnet_lddmm.resnet_flow.surface_signals import curvature_signal


def face_geometry(vertices: Tensor, faces: Tensor):
    """Per-triangle centre, unit normal and area.

    Args:
        vertices: [..., V, 3]
        faces: [F, 3] long

    Returns:
        (centres [..., F, 3], unit normals [..., F, 3], areas [..., F]). A
        degenerate triangle gets a ZERO normal and area, so it drops out of every
        kernel sum instead of turning it into NaN.
    """
    v0, v1, v2 = (vertices[..., faces[:, k], :] for k in range(3))
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    norm = cross.norm(dim=-1, keepdim=True)
    return (v0 + v1 + v2) / 3.0, cross / norm.clamp_min(1e-12), 0.5 * norm[..., 0]


def pair_terms(c1: Tensor, n1: Tensor, c2: Tensor, n2: Tensor, f1: Optional[Tensor] = None,
               f2: Optional[Tensor] = None, sigma_f: Optional[float] = None):
    """The scale-independent parts of the kernel between two triangle sets.

    Squared distances go through ``|a|^2 + |b|^2 - 2 a.b`` -- one matrix product
    instead of an [F1, F2, 3] difference tensor. Callers pass centres relative
    to the window centre, which keeps them small enough for float32.

    Returns:
        (squared distances [F1, F2], (n1 . n2)^2 [F1, F2], signal exponent
        -(f1 - f2)^2 / sigma_f^2 [F1, F2] or None)
    """
    sq = ((c1 * c1).sum(-1)[:, None] + (c2 * c2).sum(-1)[None, :] - 2.0 * c1 @ c2.T).clamp_min(0.0)
    normal = (n1 @ n2.T).pow(2)
    signal = None if f1 is None else -(f1[:, None] - f2[None, :]).pow(2) / sigma_f ** 2
    return sq, normal, signal


def scale_inner(m1: Tensor, m2: Tensor, sq: Tensor, normal: Tensor, signal: Optional[Tensor],
                sigma: float) -> Tensor:
    """<S, T> at one position scale, from ``pair_terms``' output."""
    exponent = -sq / sigma ** 2 if signal is None else signal - sq / sigma ** 2
    return m1 @ (torch.exp(exponent) * normal) @ m2


def fvarifold_inner(c1: Tensor, n1: Tensor, m1: Tensor, c2: Tensor, n2: Tensor, m2: Tensor,
                    sigma: float, f1: Optional[Tensor] = None, f2: Optional[Tensor] = None,
                    sigma_f: Optional[float] = None) -> Tensor:
    """<S, T> between two sets of weighted, orientation-free, signal-carrying triangles.

    Args:
        c1, n1: [F1, 3] centres and unit normals; m1: [F1] masses
        c2, n2, m2: the same for the second surface, [F2, ...]
        sigma: position kernel width (same units as the centres)
        f1, f2: [F1] / [F2] signals, or None for a plain varifold
        sigma_f: signal kernel width, required with signals

    Returns:
        Scalar tensor.
    """
    return scale_inner(m1, m2, *pair_terms(c1, n1, c2, n2, f1, f2, sigma_f), sigma)


class FshapeDataTerm(nn.Module):
    """Relative fshape distance between the deformed template and the target, in a window.

    Flows its own fixed template triangle subset each step (like
    LandmarkGuidanceLoss flows its own points), so it is independent of the
    main data term's subsampling. The target side is precomputed. All
    tensors are resolved at build time by ``build_fshape_kwargs``.
    """

    def __init__(self, template_vertices: Tensor, template_faces: Tensor, template_kappa: float,
                 target_centers: Tensor, target_normals: Tensor, target_masses: Tensor,
                 center: Tensor, r_in: float, r_out: float, sigmas: Sequence[float],
                 template_signal: Optional[Tensor] = None, target_signal: Optional[Tensor] = None,
                 sigma_signal: Optional[float] = None):
        """Initialize with the resolved inputs (normalised frame throughout).

        Args:
            template_vertices: [V, 3] vertices of the template triangle subset
            template_faces: [F, 3] the subset, indexing template_vertices
            template_kappa: area factor making the subset represent the crop
            target_centers, target_normals: [G, 3]; target_masses: [G], window
                and kappa already applied
            center, r_in, r_out: the window (SmoothBallMask)
            sigmas: position kernel widths, normalised units
            template_signal, target_signal: [F] / [G] per-triangle signals, or
                both None for a plain varifold
            sigma_signal: signal kernel width, required with signals

        Raises:
            ValueError: exactly one signal given, or signals without sigma_signal.
        """
        super().__init__()
        if (template_signal is None) != (target_signal is None):
            raise ValueError("give both signals (fshape) or neither (varifold)")
        if template_signal is not None and not sigma_signal:
            raise ValueError("signals need a positive sigma_signal")
        self.register_buffer("template_vertices", template_vertices)
        self.register_buffer("template_faces", template_faces.long())
        self.register_buffer("target_centers", target_centers)
        self.register_buffer("target_normals", target_normals)
        self.register_buffer("target_masses", target_masses)
        self.template_kappa = float(template_kappa)
        self.use_signal = template_signal is not None
        if self.use_signal:
            self.register_buffer("template_signal", template_signal)
            self.register_buffer("target_signal", target_signal)
        self.sigma_signal = sigma_signal
        self.sigmas = tuple(float(s) for s in sigmas)
        self.window = SmoothBallMask()
        self.window.place(center, r_in, r_out)
        with torch.no_grad():
            f_t = self.target_signal if self.use_signal else None
            t_rel = self.target_centers - self.window.center
            tt_terms = pair_terms(t_rel, self.target_normals, t_rel, self.target_normals,
                                  f_t, f_t, self.sigma_signal)
            self.register_buffer("target_self", torch.stack([
                scale_inner(self.target_masses, self.target_masses, *tt_terms, s) for s in self.sigmas]))

    def distance(self, vertices: Tensor) -> Tensor:
        """Mean over scales (and batch) of D / <T,T> for deformed template vertices.

        Args:
            vertices: [B, V, 3] the template subset's vertices, deformed

        Returns:
            Scalar tensor; 0 when the windowed fshapes coincide.
        """
        centers, normals, areas = face_geometry(vertices, self.template_faces)
        masses = self.template_kappa * areas * self.window(centers)[..., 0]
        rel = centers - self.window.center.to(centers.dtype)
        t_rel = self.target_centers - self.window.center
        f_s = self.template_signal if self.use_signal else None
        f_t = self.target_signal if self.use_signal else None
        total = 0.0
        for b in range(vertices.shape[0]):
            ss_terms = pair_terms(rel[b], normals[b], rel[b], normals[b], f_s, f_s, self.sigma_signal)
            st_terms = pair_terms(rel[b], normals[b], t_rel, self.target_normals, f_s, f_t, self.sigma_signal)
            for s, tt in zip(self.sigmas, self.target_self):
                ss = scale_inner(masses[b], masses[b], *ss_terms, s)
                st = scale_inner(masses[b], self.target_masses, *st_terms, s)
                total = total + (ss - 2.0 * st + tt) / tt
        return total / (vertices.shape[0] * len(self.sigmas))

    def forward(self, ctx: LossContext) -> Optional[Tensor]:
        """Flow the template subset and return its fshape distance to the target.

        Returns:
            Scalar tensor, or None when there is no flow.
        """
        if ctx.flow is None:
            return None
        vertices = self.template_vertices[None]
        if ctx.code is not None and ctx.code.shape[0] > 1:
            vertices = vertices.expand(ctx.code.shape[0], -1, -1)
        return self.distance(ctx.flow(vertices, ctx.code).end)


def _area_sample(areas: np.ndarray, n: int, rng: np.random.Generator):
    """Up to n triangle indices drawn proportionally to area, and the mass factor kappa."""
    if len(areas) <= n:
        return np.arange(len(areas)), 1.0
    idx = rng.choice(len(areas), size=n, replace=False, p=areas / areas.sum())
    return idx, float(areas.sum() / areas[idx].sum())


def _submesh(faces: np.ndarray):
    """(unique vertex indices, faces reindexed onto them)."""
    vertex_idx, inverse = np.unique(faces.ravel(), return_inverse=True)
    return vertex_idx, inverse.reshape(faces.shape)


def _triangle_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    e1, e2 = points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def build_fshape_kwargs(source_points: np.ndarray, source_faces: np.ndarray,
                        target_points: np.ndarray, target_faces: np.ndarray,
                        center: np.ndarray, r_in: float, r_out: float, margin: float,
                        sigmas: Sequence[float], n_faces: int, seed: int,
                        signal: str, sigma_signal: Optional[float], curvature_radius_mm: float,
                        scale: float, dtype=torch.float32, signal_normalize: str = "rank"):
    """Resolve every FshapeDataTerm input from the two NORMALISED meshes.

    Template: triangles whose centre lies within ``r_out + margin`` of the
    window centre (the margin lets tissue that moves INTO the window be
    counted), area-sampled to ``n_faces``. Target: triangles with a non-zero
    window weight, area-sampled likewise, masses = kappa x area x window.
    Curvature is computed in mm (points x ``scale``) with neighbours from the
    WHOLE mesh, then averaged per triangle. For ``mean_curvature`` the target's
    sign is flipped when the two meshes' mean normals disagree, so convex means
    the same thing on both. ``signal_normalize: rank`` then replaces each
    mesh's values by their percentile inside the window, since a noisy scan is
    systematically curvier than the smooth template.

    Returns:
        (kwargs for FshapeDataTerm, info dict for logging and sanity exports)
    """
    rng = np.random.default_rng(seed)
    window = SmoothBallMask()
    window.place(torch.as_tensor(center, dtype=torch.float32), r_in, r_out)

    src_centers = source_points[source_faces].mean(1)
    crop = np.linalg.norm(src_centers - center, axis=1) < r_out + margin
    src_crop = source_faces[crop]
    src_idx, kappa_s = _area_sample(_triangle_areas(source_points, src_crop), n_faces, rng)
    src_vertex_idx, src_faces_sub = _submesh(src_crop[src_idx])

    tgt_centers = target_points[target_faces].mean(1)
    with torch.no_grad():
        tgt_window = window(torch.as_tensor(tgt_centers, dtype=torch.float32))[:, 0].numpy()
    keep = tgt_window > 0
    tgt_keep, tgt_window = target_faces[keep], tgt_window[keep]
    tgt_areas = _triangle_areas(target_points, tgt_keep)
    tgt_idx, kappa_t = _area_sample(tgt_areas, n_faces, rng)
    tgt_vertex_idx, tgt_faces_sub = _submesh(tgt_keep[tgt_idx])

    t_vertices = torch.as_tensor(target_points[tgt_vertex_idx], dtype=dtype)
    t_centers, t_normals, t_areas = face_geometry(t_vertices, torch.as_tensor(tgt_faces_sub).long())
    t_masses = kappa_t * t_areas * torch.as_tensor(tgt_window[tgt_idx], dtype=dtype)

    kwargs = dict(
        template_vertices=torch.as_tensor(source_points[src_vertex_idx], dtype=dtype),
        template_faces=torch.as_tensor(src_faces_sub).long(),
        template_kappa=kappa_s,
        target_centers=t_centers, target_normals=t_normals, target_masses=t_masses,
        center=torch.as_tensor(center, dtype=dtype), r_in=r_in, r_out=r_out, sigmas=list(sigmas),
    )
    info = dict(template_faces=len(src_faces_sub), template_crop_faces=int(crop.sum()),
                target_faces=len(tgt_faces_sub), target_window_faces=int(keep.sum()),
                template_export=(source_points[src_vertex_idx], src_faces_sub, None),
                target_export=(target_points[tgt_vertex_idx], tgt_faces_sub, None),
                sign_flipped=False, missing=(0, 0), sigma_signal=None)

    if signal != "none":
        s_vertex, s_missing = curvature_signal(source_points * scale, source_faces, curvature_radius_mm,
                                               signal, query=src_vertex_idx)
        t_vertex, t_missing = curvature_signal(target_points * scale, target_faces, curvature_radius_mm,
                                               signal, query=tgt_vertex_idx)
        if signal == "mean_curvature":
            s_geom = face_geometry(kwargs["template_vertices"], kwargs["template_faces"])
            mean_s = (s_geom[1] * s_geom[2][:, None]).sum(0)
            mean_t = (t_normals * t_areas[:, None]).sum(0)
            if float(mean_s @ mean_t) < 0:
                t_vertex = -t_vertex
                info["sign_flipped"] = True
        s_face = s_vertex[src_faces_sub].mean(1)
        t_face = t_vertex[tgt_faces_sub].mean(1)
        raw_stats = (float(np.median(s_face)), float(np.quantile(s_face, .95)),
                     float(np.median(t_face)), float(np.quantile(t_face, .95)))
        if signal_normalize == "rank":
            # Each mesh's signal -> its percentile among ITS OWN in-window
            # triangles: "the most curved 5% of the region" then means the same
            # on a smooth template and a noisier scan. Template margin triangles
            # (window 0 at the start) are mapped through the same reference.
            s_centers = source_points[src_crop[src_idx]].mean(1)
            with torch.no_grad():
                inside = window(torch.as_tensor(s_centers, dtype=torch.float32))[:, 0].numpy() > 0
            s_face = _empirical_cdf(s_face[inside], s_face)
            t_face = _empirical_cdf(t_face, t_face)
        if sigma_signal is None:
            sigma_signal = 0.5 * float(np.std(np.concatenate([s_face, t_face])))
        kwargs.update(template_signal=torch.as_tensor(s_face, dtype=dtype),
                      target_signal=torch.as_tensor(t_face, dtype=dtype),
                      sigma_signal=float(sigma_signal))
        info.update(missing=(s_missing, t_missing), sigma_signal=float(sigma_signal),
                    template_export=(source_points[src_vertex_idx], src_faces_sub, s_vertex),
                    target_export=(target_points[tgt_vertex_idx], tgt_faces_sub, t_vertex),
                    signal_stats=raw_stats)
    return kwargs, info


def _empirical_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Fraction of ``reference`` <= each value, in (0, 1]."""
    return np.searchsorted(np.sort(reference), values, side="right") / len(reference)
