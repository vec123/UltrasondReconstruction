"""Resamples a Deformetrica atlas's template to a higher vertex count.

The fetus atlas's resolution is fixed by whatever mesh Deformetrica's original
estimation used as a template (1416 vertices here) - there is no slider for
that. This runner produces a second, independent atlas at roughly ``scale``
times the vertex count, so a config that wants more geometric detail (a finer
curvature field, more room between landmarks) can just point at it instead.

Two things carry over unchanged from the source atlas: the control points and
every subject's momenta. Nothing about them is resolution-specific - a
diffeomorphism's initial momentum lives on the control points, not on the
template's vertices, and :func:`src.model.deformetrica.shoot` already takes
*any* set of points to carry along the geodesic. So a denser template is not
an approximation of the model - it is the same control points and the same
184 momenta, evaluated (shot, then PCA'd) at more points, through the exact
loader every other config already uses.

The template surface is resampled to close-to-uniform vertex spacing: an
earlier version subdivided and then quadric-decimated back down to the target
count, but quadric decimation is deliberately feature-preserving - it keeps
vertices where curvature is high and thins them out on flat regions - which is
the opposite of uniform, and measurably so (edge-length coefficient of
variation 0.43 against the original template's own 0.23). Instead,
:func:`vtk.vtkAdaptiveSubdivisionFilter` is driven purely by a maximum edge
length: any triangle with an edge longer than the threshold gets split, so the
result has no edge longer than that threshold and no curvature-driven bias.
The right threshold for a given target vertex count isn't known in closed
form, so it is found by bisection - point count is monotonically decreasing in
the threshold, which is the only property bisection needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vtk

from src.config.schema import UpsampleAtlasConfig
from src.geometry.vtk import create_polydata, get_vtp_faces, get_vtp_vertices, save_vtk
from src.model.deformetrica import load_atlas, load_atlas_ssm


def _edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate([
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 1]],
        vertices[faces[:, 0]] - vertices[faces[:, 2]],
    ])
    return np.linalg.norm(edges, axis=1)


@dataclass
class UpsampleAtlasResult:
    source_vertices: int
    output_vertices: int
    output_faces: int
    output_dir: Path
    cache_warmed: bool


class UpsampleAtlasRunner:
    """Reads an atlas, resamples its template, writes a new atlas directory."""

    def __init__(self, config: UpsampleAtlasConfig):
        self.config = config

    def run(self) -> UpsampleAtlasResult:
        cfg = self.config
        if not cfg.source.is_dir():
            raise NotADirectoryError(f"source is not a directory: {cfg.source}")

        atlas = load_atlas(cfg.source)
        self._log(f"source: {cfg.source}")
        self._log(f"  {atlas.n_subjects} subjects, {atlas.n_vertices} vertices, "
                  f"{len(atlas.control_points)} control points, sigma {atlas.kernel_width:g}")

        template = create_polydata(atlas.template.astype(np.float32), atlas.faces)
        target_points = int(round(atlas.n_vertices * cfg.scale))
        dense = self._resample_uniform(template, target_points)

        vertices = get_vtp_vertices(dense).astype(np.float64)
        faces = get_vtp_faces(dense)
        source_lengths = _edge_lengths(atlas.template, atlas.faces)
        source_cv = source_lengths.std() / source_lengths.mean()
        lengths = _edge_lengths(vertices, faces)
        self._log(f"resampled: {len(vertices)} vertices ({len(vertices) / atlas.n_vertices:.2f}x), "
                  f"{len(faces)} triangles (target was {target_points})")
        self._log(f"  edge length: mean {lengths.mean():.3f}, cv {lengths.std() / lengths.mean():.3f} "
                  f"(source cv {source_cv:.3f} - lower is more uniform)")

        self._write_atlas(vertices, faces, atlas.control_points, atlas.momenta, atlas.kernel_width)
        self._log(f"wrote atlas to {cfg.output_dir}")

        cache_warmed = False
        if cfg.warm_cache:
            self._warm_cache()
            cache_warmed = True

        return UpsampleAtlasResult(
            source_vertices=atlas.n_vertices,
            output_vertices=len(vertices),
            output_faces=len(faces),
            output_dir=cfg.output_dir,
            cache_warmed=cache_warmed,
        )

    # -- resampling ----------------------------------------------------------

    def _resample_uniform(self, polydata, target_points: int, tol: float = 0.01):
        """A copy of ``polydata`` with about ``target_points`` vertices, spaced
        as evenly as an edge-length threshold can make them.

        vtkAdaptiveSubdivisionFilter takes the threshold, not the vertex count,
        so the threshold is found by bisection: point count is monotonically
        non-increasing as the threshold grows (a larger threshold only ever
        disqualifies edges from being split, never qualifies more of them), so
        the search always converges. The upper bracket is the mesh's own
        longest edge - at or above it nothing gets split, which reproduces the
        source point count exactly and is the one value guaranteed to be
        smaller than any target this is called with (``scale`` > 1 is enforced
        by the config). The lower bracket is found by repeatedly halving until
        it overshoots the target, standard exponential search.
        """
        source_points = polydata.GetNumberOfPoints()
        if target_points <= source_points:
            return polydata

        hi = float(_edge_lengths(
            get_vtp_vertices(polydata).astype(np.float64), get_vtp_faces(polydata),
        ).max())
        lo = hi / 2.0
        for _ in range(30):
            if self._subdivide(polydata, lo).GetNumberOfPoints() >= target_points:
                break
            hi = lo
            lo /= 2.0
        else:
            raise RuntimeError(
                f"edge length threshold search did not bracket {target_points} points "
                f"after 30 halvings - the target is likely unreasonably large"
            )

        tolerance = max(1, int(round(target_points * tol)))
        result = None
        for _ in range(40):
            mid = (lo + hi) / 2.0
            result = self._subdivide(polydata, mid)
            n = result.GetNumberOfPoints()
            if abs(n - target_points) <= tolerance:
                break
            if n >= target_points:
                lo = mid
            else:
                hi = mid
        return result

    @staticmethod
    def _subdivide(polydata, max_edge_length: float):
        subdivide = vtk.vtkAdaptiveSubdivisionFilter()
        subdivide.SetInputData(polydata)
        subdivide.SetMaximumEdgeLength(max_edge_length)
        # Only the edge-length criterion should drive this, so the other two
        # are set far out of reach rather than left at their (small) defaults.
        subdivide.SetMaximumTriangleArea(1.0e30)
        subdivide.SetMaximumNumberOfTriangles(10_000_000)
        subdivide.Update()

        # vtkAdaptiveSubdivisionFilter only ever emits triangles, but this
        # keeps that an invariant rather than an assumption - get_vtp_faces()
        # raises on anything else.
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputConnection(subdivide.GetOutputPort())
        triangulate.Update()
        return triangulate.GetOutput()

    # -- writing ---------------------------------------------------------------

    def _write_atlas(self, vertices, faces, control_points, momenta, kernel_width):
        out = self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)

        # Regenerating an existing output directory in place: its old shooting
        # cache is keyed by vertex count (see _read_cache in deformetrica.py),
        # so a resolution change already makes it get ignored - this just also
        # removes the now-unreachable file instead of leaving it on disk.
        for stale_cache in out.glob("ssm_shot_*.npz"):
            stale_cache.unlink()
            self._log(f"  removed stale cache {stale_cache.name}")

        template_path = out / "DeterministicAtlas__EstimatedParameters__Template.vtk"
        save_vtk(create_polydata(vertices.astype(np.float32), faces), str(template_path), binary=True)

        control_points_path = out / "DeterministicAtlas__EstimatedParameters__ControlPoints.txt"
        np.savetxt(control_points_path, control_points, fmt="%.17g")

        momenta_path = out / "DeterministicAtlas__EstimatedParameters__Momenta.txt"
        self._write_momenta(momenta_path, momenta)

        params_path = out / "params.json"
        params_path.write_text(json.dumps({"kwd": float(kernel_width)}, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _write_momenta(path: Path, momenta: np.ndarray):
        """The '<S> <M> <dim>' header plus one blank-line-separated block per
        subject - matches what src.model.deformetrica._read_momenta() parses,
        including the header it requires; the blank lines are only for
        readability, since the parser treats the remainder as one flat blob."""
        n_subjects, n_control_points, dim = momenta.shape
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"{n_subjects} {n_control_points} {dim}\n")
            for subject in momenta:
                np.savetxt(handle, subject, fmt="%.17g")
                handle.write("\n")

    # -- cache warming -----------------------------------------------------

    def _warm_cache(self):
        cfg = self.config
        self._log(f"  shooting {cfg.warm_cache_time_points} time points x every subject "
                  f"at the new resolution (this is the slow part, same as a fresh original atlas)")
        load_atlas_ssm(
            cfg.output_dir, normalize=False,
            time_points=cfg.warm_cache_time_points, cache=True, verbose=cfg.verbose,
        )

    def _log(self, *args):
        if self.config.verbose:
            print(*args)
