"""Similarity alignment of two meshes that have no correspondence.

Everything else in this project that solves a transform has something to hold
on to: ``gmm_2_template`` has vertex-wise correspondence, ``align_landmarks``
has named landmarks. Bringing the fetus template into the baby model's frame
has neither - 1416 points against 31027, no shared landmarks, and a scale
ratio of about 6000 - so the transform has to be found from the geometry alone.

That is done in three stages:

1. **Normalise.** Both meshes are centred on their centroid and scaled to unit
   RMS radius. Without this an ICP would start with every source point nearer
   to the target's centre than to its own counterpart, and the first
   correspondence set would be meaningless.

2. **Enumerate orientations.** The principal axes of each point cloud give a
   frame, but only up to a sign per axis and, when two extents are similar, up
   to a permutation - so the frames match in 24 proper rotations, not one.
   Every one is tried; a face is roughly an ovoid and several of them fit
   almost as well as the right one, which is exactly why guessing a single
   PCA alignment is unsafe.

3. **Refine and score.** The most promising candidates get a similarity ICP,
   and the winner is chosen on a *symmetric* residual - source to target and
   target to source. A one-sided residual would reward the transform for
   shrinking the source into a dense part of the target, which a similarity
   ICP is perfectly able to do. Candidates whose surfaces end up facing
   opposite ways are rejected on normal agreement, since an inside-out fit can
   otherwise score well on distance alone.

The result is an initial guess, not a registration: it places and scales the
fetus template on the baby mean so a nonrigid method has somewhere sane to
start. The residual it reports is the shape difference between a fetal and a
neonatal face, which no similarity transform can remove.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations, product
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import vtk
from scipy.spatial import cKDTree

from src.config.schema import AlignMeshesConfig, MeshSource
from src.geometry.vtk import (
    create_polydata, get_vtp_faces, get_vtp_vertices, load_vtp_mesh,
    save_vtp_mesh, transform_polydata,
)


@dataclass
class Candidate:
    """One starting orientation and what it converged to."""

    label: str
    R: np.ndarray
    t: np.ndarray
    s: float
    residual: float          # symmetric mean surface distance, target units
    normal_agreement: float  # mean n_src . n_tgt over matched pairs, in [-1, 1]

    @property
    def rejected(self) -> bool:
        return self.normal_agreement < 0.0


@dataclass
class AlignMeshesResult:
    R: np.ndarray
    t: np.ndarray
    s: float
    residual: float
    candidates: List[Candidate] = field(default_factory=list)
    landmark_distances: dict = field(default_factory=dict)
    written_files: List[Path] = field(default_factory=list)


class AlignMeshesRunner:
    """Solves x_target ~= s R x_source + t without correspondence."""

    def __init__(self, config: AlignMeshesConfig):
        self.config = config
        self.written_files: List[Path] = []

    def run(self) -> AlignMeshesResult:
        self.load_meshes()
        candidates = self.solve()
        best = candidates[0]

        self._log(f"\nchosen: {best.label}")
        self._log(f"  scale     {best.s:.6g}")
        self._log(f"  residual  {best.residual:.6g} (symmetric mean surface distance)")
        self._log(f"  normals   {best.normal_agreement:+.3f} agreement")

        landmarks = self.landmark_report(best)
        self.write_outputs(best)

        return AlignMeshesResult(
            R=best.R, t=best.t, s=best.s, residual=best.residual,
            candidates=candidates, landmark_distances=landmarks,
            written_files=self.written_files,
        )

    # -- inputs ------------------------------------------------------------

    def load_meshes(self):
        cfg = self.config
        self.source_vertices, self.source_faces, self.source_polydata = self._load(cfg.source)
        self.target_vertices, self.target_faces, self.target_polydata = self._load(cfg.target)

        self._log(f"source: {len(self.source_vertices)} points, "
                  f"extent {np.array2string(self._extent(self.source_vertices), precision=4)}")
        self._log(f"target: {len(self.target_vertices)} points, "
                  f"extent {np.array2string(self._extent(self.target_vertices), precision=4)}")

        self.target_tree = cKDTree(self.target_vertices)
        self.source_normals = self._normals(self.source_polydata)
        self.target_normals = self._normals(self.target_polydata)
        self.target_weights = self._weights()

    def _weights(self) -> np.ndarray:
        """Per-target-point weight for the ICP and the residual."""
        focus = self.config.focus
        weights = np.ones(len(self.target_vertices))
        if focus is None or (focus.field is None and focus.weight == 1.0):
            return weights

        if focus.field is not None:
            from vtk.util import numpy_support

            array = self.target_polydata.GetPointData().GetArray(focus.field)
            if array is None:
                available = [self.target_polydata.GetPointData().GetArrayName(i)
                             for i in range(self.target_polydata.GetPointData().GetNumberOfArrays())]
                raise ValueError(
                    f"focus.field '{focus.field}' is not on the target mesh. Available: {available}"
                )
            field = numpy_support.vtk_to_numpy(array).astype(np.float64).reshape(len(weights), -1)
            field = field[:, 0]
            span = field.max() - field.min()
            shape = (field - field.min()) / span if span > 0 else np.zeros_like(field)
            self._log(f"  focus: field '{focus.field}', weight up to {focus.weight:g}x")
        else:
            names = getattr(self, "_landmark_names", None)
            if not names or focus.landmark not in names:
                raise ValueError(
                    f"focus.landmark '{focus.landmark}' needs a target shape_model carrying it. "
                    f"Available: {names}"
                )
            centre = self._landmark_points[names.index(focus.landmark)]
            distance = np.linalg.norm(self.target_vertices - centre, axis=1)
            shape = np.exp(-0.5 * (distance / focus.radius) ** 2)
            self._log(f"  focus: '{focus.landmark}' +/- {focus.radius:g}, "
                      f"weight up to {focus.weight:g}x "
                      f"({int((distance <= focus.radius).sum())} of {len(distance)} points inside)")

        return 1.0 + (focus.weight - 1.0) * shape

    def _load(self, source: MeshSource):
        """The mesh, its topology, and - when a model is given - its landmarks.

        A mesh file wins on geometry because it carries the point fields with
        it; the model is still read for its landmarks, and the two are checked
        against each other rather than assumed to match.
        """
        model_vertices = None
        if source.shape_model is not None:
            from src.model.load import load_shape_model, load_shape_model_triangulation

            if not source.shape_model.exists():
                raise FileNotFoundError(f"shape_model does not exist: {source.shape_model}")
            mean, _, _, landmarks = load_shape_model(
                source.shape_model, normalize=source.normalize, verbose=False
            )
            model_vertices = np.asarray(mean, dtype=np.float64).reshape(-1, 3)
            model_faces = load_shape_model_triangulation(source.shape_model)

            # Kept as positions, not vertex indices: a cropped mesh has its own
            # numbering, and the landmarks have to survive that.
            indices = np.asarray(landmarks.get("verts ", []), dtype=np.int64).reshape(-1)
            if len(indices):
                self._landmark_names = [str(n).strip() for n in landmarks["names "]]
                self._landmark_points = model_vertices[indices]

        if source.mesh_vtp is None:
            return (model_vertices, model_faces,
                    create_polydata(model_vertices.astype(np.float32), model_faces))

        if not source.mesh_vtp.is_file():
            raise FileNotFoundError(f"mesh_vtp does not exist: {source.mesh_vtp}")
        polydata = load_vtp_mesh(str(source.mesh_vtp))
        if source.triangulate:
            polydata = self._triangulate(polydata, source)
        vertices = np.asarray(get_vtp_vertices(polydata), dtype=np.float64)

        fields = polydata.GetPointData()
        names = [fields.GetArrayName(i) for i in range(fields.GetNumberOfArrays())]
        if names:
            self._log(f"  {source.mesh_vtp.name} carries {len(names)} point field(s)")

        if model_vertices is not None:
            self._check_same_shape(source, vertices, model_vertices)

        try:
            faces = get_vtp_faces(polydata)
        except ValueError as exc:                # mixed cell sizes, triangulate: false
            self._log(f"  topology not read: {exc}")
            faces = None
        return vertices, faces, polydata

    def _triangulate(self, polydata, source: MeshSource):
        n_before = polydata.GetNumberOfPolys()
        filt = vtk.vtkTriangleFilter()
        filt.SetInputData(polydata)
        filt.PassVertsOff()
        filt.PassLinesOff()
        filt.Update()
        output = filt.GetOutput()
        if output.GetNumberOfPolys() != n_before:
            self._log(f"  {source.mesh_vtp.name}: {n_before} cells triangulated into "
                      f"{output.GetNumberOfPolys()}")
        return output

    def _check_same_shape(self, source: MeshSource, vertices, model_vertices):
        """The mesh and the model must be the same surface for landmarks to mean anything.

        Two cases are legitimate. The mesh is the model's mean as written by
        mean_shape.py, in which case it agrees vertex for vertex; or it is a
        cropped piece of it, in which case it has its own point count and a
        boundary of points the clip interpolated, so it can only be required to
        *lie on* the model's surface.
        """
        scale = float(np.linalg.norm(model_vertices - model_vertices.mean(0), axis=1).max())

        if len(vertices) == len(model_vertices):
            offset = float(np.linalg.norm(vertices - model_vertices, axis=1).max())
            if offset > 1e-4 * scale:
                raise ValueError(
                    f"{source.mesh_vtp.name} and the mean of {source.shape_model.name} differ "
                    f"by up to {offset:.6g} ({offset / scale:.2%} of the shape). They are meant "
                    f"to be the same mesh - regenerate it with scripts/mean_shape.py, whose "
                    f"output is stale whenever the model changes."
                )
            return

        distance, _ = cKDTree(model_vertices).query(vertices)
        stray = float(np.percentile(distance, 99))
        if stray > 0.02 * scale:
            raise ValueError(
                f"{source.mesh_vtp.name} has {len(vertices)} points against the model's "
                f"{len(model_vertices)}, and they do not lie on the model's surface "
                f"(99th percentile offset {stray:.6g}, {stray / scale:.2%} of the shape). "
                f"A cropped copy of the mean is expected here."
            )
        self._log(f"  a cropped piece of {source.shape_model.name}: {len(vertices)} of "
                  f"{len(model_vertices)} points, lying on its surface to "
                  f"{distance.max():.3g} ({distance.max() / scale:.2%})")

    @staticmethod
    def _extent(vertices: np.ndarray) -> np.ndarray:
        return vertices.max(axis=0) - vertices.min(axis=0)

    @staticmethod
    def _normals(polydata) -> Optional[np.ndarray]:
        """Per-vertex normals, or None for a mesh without triangles."""
        if polydata.GetNumberOfPolys() == 0:
            return None
        filt = vtk.vtkPolyDataNormals()
        filt.SetInputData(polydata)
        filt.ComputePointNormalsOn()
        filt.ComputeCellNormalsOff()
        filt.SplittingOff()
        filt.ConsistencyOn()
        filt.AutoOrientNormalsOn()
        filt.Update()

        from vtk.util import numpy_support
        array = filt.GetOutput().GetPointData().GetNormals()
        return None if array is None else numpy_support.vtk_to_numpy(array).astype(np.float64)

    # -- the solve ---------------------------------------------------------

    @staticmethod
    def _frame(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Centroid, principal axes (columns, descending) and RMS radius."""
        centroid = vertices.mean(axis=0)
        centered = vertices - centroid
        rms = float(np.sqrt((centered ** 2).sum(axis=1).mean()))
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        return centroid, right.T, rms

    def _orientations(self, source_axes: np.ndarray, target_axes: np.ndarray):
        """The 24 proper rotations that carry one principal frame onto the other.

        The axes come out of the SVD with an arbitrary sign, and their order is
        only meaningful when the extents along them are well separated - which
        is not the case for two face meshes that cover different parts of the
        face. Both freedoms are therefore enumerated rather than assumed.
        """
        for order in permutations(range(3)):
            permuted = source_axes[:, list(order)]
            for signs in product((1.0, -1.0), repeat=3):
                candidate = permuted * np.asarray(signs)
                R = target_axes @ candidate.T
                if np.linalg.det(R) > 0:                 # proper rotation only
                    yield f"axes {order} signs {''.join('+-'[s < 0] for s in signs)}", R

    def solve(self) -> List[Candidate]:
        cfg = self.config
        source_centroid, source_axes, source_rms = self._frame(self.source_vertices)
        target_centroid, target_axes, target_rms = self._frame(self.target_vertices)
        scale = target_rms / source_rms

        self._log(f"\ninitial scale from RMS radius: {scale:.6g}")

        # Screen every orientation cheaply, then refine only the best few: an
        # ICP per candidate is affordable here but pointless for the ones that
        # start facing the wrong way entirely.
        screened = []
        for label, R in self._orientations(source_axes, target_axes):
            t = target_centroid - scale * (R @ source_centroid)
            residual, agreement = self.score(R, t, scale)
            screened.append(Candidate(label, R, t, scale, residual, agreement))
        screened.sort(key=lambda c: c.residual)

        self._log(f"screened {len(screened)} orientations, "
                  f"refining the best {cfg.refine_count}")

        refined = []
        for candidate in screened[:cfg.refine_count]:
            refined.append(self.refine(candidate))

        # A surface fitted inside-out can still score well on distance alone,
        # so those are only kept if nothing else survives.
        refined.sort(key=lambda c: (c.rejected, c.residual))
        for candidate in refined:
            self._log(f"  {candidate.label:<28} residual {candidate.residual:.6g}  "
                      f"scale {candidate.s:.6g}  normals {candidate.normal_agreement:+.3f}"
                      f"{'  (rejected: inside out)' if candidate.rejected else ''}")
        return refined

    @staticmethod
    def _weighted_similarity(source, target, weights, allow_scale: bool):
        """Weighted Umeyama: the s, R, t minimising sum_i w_i |s R x_i + t - y_i|^2."""
        total = weights.sum()
        source_mean = (weights[:, None] * source).sum(axis=0) / total
        target_mean = (weights[:, None] * target).sum(axis=0) / total
        x, y = source - source_mean, target - target_mean

        covariance = (weights[:, None] * x).T @ y
        u, singular, vt = np.linalg.svd(covariance)
        correction = np.ones(3)
        if np.linalg.det(vt.T @ u.T) < 0:            # keep it a rotation
            correction[2] = -1.0
        R = vt.T @ np.diag(correction) @ u.T

        s = 1.0
        if allow_scale:
            variance = (weights * (x ** 2).sum(axis=1)).sum()
            s = float((singular * correction).sum() / variance) if variance > 0 else 1.0

        return R, target_mean - s * (R @ source_mean), s

    def refine(self, candidate: Candidate) -> Candidate:
        """Weighted similarity ICP from a starting orientation.

        Written out rather than handed to vtkIterativeClosestPointTransform
        because that one weights every correspondence equally, and the whole
        point of ``focus`` is that they are not equal.
        """
        cfg = self.config
        R, t, s = candidate.R, candidate.t, candidate.s
        previous = np.inf

        for _ in range(cfg.iterations):
            moved = s * (self.source_vertices @ R.T) + t
            distance, indices = self.target_tree.query(moved)

            weights = self.target_weights[indices]
            if cfg.trim_ratio > 0.0:
                # Drop the worst matches: the two surfaces do not cover quite
                # the same region, and an unmatched rim would otherwise drag
                # the transform outward.
                keep = distance <= np.quantile(distance, 1.0 - cfg.trim_ratio)
                weights = weights * keep

            if weights.sum() <= 0:
                break

            R, t, s = self._weighted_similarity(
                self.source_vertices, self.target_vertices[indices], weights, cfg.allow_scale
            )

            mean_distance = float((weights * distance).sum() / weights.sum())
            if abs(previous - mean_distance) < cfg.tolerance:
                break
            previous = mean_distance

        residual, agreement = self.score(R, t, s)
        return Candidate(candidate.label, R, t, s, residual, agreement)

    def score(self, R: np.ndarray, t: np.ndarray, s: float) -> Tuple[float, float]:
        """Symmetric mean surface distance, and how far the normals agree.

        Symmetric because a one-sided distance is minimised by shrinking the
        source into the densest part of the target, which a similarity ICP will
        happily do if allowed to.
        """
        moved = s * (self.source_vertices @ R.T) + t

        forward, indices = self.target_tree.query(moved)
        backward, _ = cKDTree(moved).query(self.target_vertices)

        # Weighted the same way the fit is, so the score ranks candidates by
        # what the fit was actually asked to achieve.
        forward_weights = self.target_weights[indices]
        residual = 0.5 * (
            float((forward_weights * forward).sum() / forward_weights.sum())
            + float((self.target_weights * backward).sum() / self.target_weights.sum())
        )

        agreement = 1.0
        if self.source_normals is not None and self.target_normals is not None:
            rotated = self.source_normals @ R.T
            agreement = float((rotated * self.target_normals[indices]).sum(axis=1).mean())

        return residual, agreement

    # -- diagnostics -------------------------------------------------------

    def landmark_report(self, best: Candidate) -> dict:
        """Distance from each of the target model's landmarks to the moved source.

        A similarity transform cannot make a fetal face into a neonatal one, so
        these are not expected to be small. They are here because they are
        readable: if the alignment came out upside down or mirrored, the nose
        and chin land nowhere near the surface and it shows up immediately.

        The landmarks are positions in the target's frame rather than vertex
        indices, so a cropped target is fine. Those that fall outside the crop
        are reported separately - the target no longer covers them, so their
        distance says nothing about the alignment.
        """
        points = getattr(self, "_landmark_points", None)
        if points is None:
            return {}

        self._best_R = best.R
        moved = best.s * (self.source_vertices @ best.R.T) + best.t
        distance, _ = cKDTree(moved).query(points)
        report = dict(zip(self._landmark_names, distance.tolist()))

        # 'Inside the target' = within the target's own vertex spacing of it.
        to_target, _ = cKDTree(self.target_vertices).query(points)
        spacing = float(np.percentile(
            cKDTree(self.target_vertices).query(self.target_vertices, k=2)[0][:, 1], 99
        ))
        inside = to_target <= 2.0 * spacing

        self._log("\nlandmarks of the target model, distance to the aligned source:")
        for name in ("prn", "n", "pg", "exR", "exL", "chR", "chL"):
            if name in report:
                index = self._landmark_names.index(name)
                mark = "" if inside[index] else "   (outside the cropped target)"
                self._log(f"  {name:<5} {report[name]:.6g}{mark}")

        if inside.any():
            covered = distance[inside]
            self._log(f"  {'':<5} {int(inside.sum())} of {len(distance)} landmarks lie on the "
                      f"target: mean {covered.mean():.6g}, max {covered.max():.6g}")
        if not inside.all():
            dropped = [n for n, keep in zip(self._landmark_names, inside) if not keep]
            self._log(f"  {'':<5} cropped away: {', '.join(dropped)}")

        self.nose_report(moved)
        return report

    def nose_report(self, moved: np.ndarray):
        """Do the two noses actually sit on top of each other?

        The landmark distances above are point-to-*surface*, which a nose can
        pass while sitting beside the target's nose rather than on it - the
        surface is right there either way. This instead finds the apex of each
        nose and measures apex to apex, which is the number that says whether
        the feature is aligned.
        """
        focus = self.config.focus
        names = getattr(self, "_landmark_names", None)
        if focus is None or not names or focus.landmark not in names:
            return

        centre = self._landmark_points[names.index(focus.landmark)]
        radius = focus.radius

        def apex(vertices, normals):
            """Most protruding point of the region, along its own mean normal."""
            near = np.linalg.norm(vertices - centre, axis=1) <= radius
            if near.sum() < 3 or normals is None:
                return None
            direction = normals[near].mean(axis=0)
            norm = np.linalg.norm(direction)
            if norm == 0:
                return None
            direction = direction / norm
            local = vertices[near]
            return local[np.argmax((local - centre) @ direction)]

        target_apex = apex(self.target_vertices, self.target_normals)
        source_apex = apex(moved, self.source_normals @ self._best_R.T
                           if self.source_normals is not None else None)
        if target_apex is None or source_apex is None:
            return

        scale = float(np.linalg.norm(
            self.target_vertices - self.target_vertices.mean(axis=0), axis=1).max())
        self._log(f"\n{focus.landmark} apex to apex: "
                  f"{np.linalg.norm(source_apex - target_apex):.6g} "
                  f"({np.linalg.norm(source_apex - target_apex) / (2 * scale):.2%} of the target)")

    # -- outputs -----------------------------------------------------------

    def write_outputs(self, best: Candidate):
        cfg = self.config
        cfg.outputs.dir.mkdir(parents=True, exist_ok=True)

        path = cfg.outputs.path("transform")
        np.savez(str(path), R=best.R, t=best.t, s=best.s,
                 direction="source_to_target", residual=best.residual)
        self.written_files.append(path)
        self._log(f"\ntransform written to {path.name} (x' = s R x + t, replayable "
                  f"with scripts/sim_tf.py)")

        moved = transform_polydata(self.source_polydata, best.R, best.t, best.s)
        path = cfg.outputs.path("aligned")
        save_vtp_mesh(moved, str(path))
        self.written_files.append(path)

        if cfg.outputs.write_target:
            # so the pair sits in one folder, in one frame, ready for whatever
            # nonrigid step comes next
            path = cfg.outputs.path("target")
            save_vtp_mesh(self.target_polydata, str(path))
            self.written_files.append(path)

    def _log(self, *args):
        if self.config.verbose:
            print(*args)
