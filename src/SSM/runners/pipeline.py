"""Runner for the morphing pipeline.

Step 1: landmark-driven similarity alignment (1a) + landmark fit of the SSM (1b).
Step 2: correspondence + MAP-regularised NICP fits (gamma sweep).
Step 3: regional Laplacian-constrained deformation.

Each stage registers its fitted shape in ``self.shapes`` / ``self.vertices``
under a key; later stages address earlier results by that key:

    refined          step 1 result (mean shape if step 1 is disabled)
    gamma_<value>    one per entry in nicp.gammas
    <variant name>   one per entry in nicp.variants

Stages are switched on and off with the ``steps`` config block. Step 3 can
therefore run straight off ``refined`` when step 2 is disabled.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.config.SSM.client import ConfigClient
from src.config.SSM.schema import PipelineConfig
from src.SSM.correspondence.vtk import save_matching_field
from src.landmarks.load_landmarks import load_landmarks
from src.transforms.similarity import SimilarityTransform
from src.geometry.spectral_operators import (
    compute_spectral_operators,
    load_spectral_data,
    save_spectral_data,
    test_LBO_spectrum,
)
from src.vtk.io import (
    load_vtp,
    save_vtp
)
from src.vtk.fields import copy_point_data
from src.vtk.extract import (
    get_vtp_faces,
    get_vtp_vertices
)
from src.vtk.create import (
    create_polydata
)

from src.SSM.laplacian.laplacian import solve_laplacian_deformation
from src.SSM.model.model_memory_io import load_model_file
from src.SSM.model.model import ShapeInstance, StatisticalShapeModel
from src.SSM.nicp import Correspondences, PointCloud
from src.SSM.nicp.problem import build_problem
from src.SSM.nicp.shape import ShapeSpace
from src.SSM.nicp.solvers import build_solver

#: registry key for the step 1 result
REFINED_KEY = "refined"


@dataclass
class PipelineResult:
    shape: ShapeInstance
    surface_vertices: np.ndarray
    aligned_surface: np.ndarray
    refined_vertices: np.ndarray
    transform: SimilarityTransform
    matches: Optional[Correspondences]
    template_vertices: Optional[np.ndarray] = None
    gmm_template_transform: Optional[SimilarityTransform] = None
    shapes: dict = field(default_factory=dict)      # key -> ShapeInstance
    vertices: dict = field(default_factory=dict)    # key -> (3N,) reconstruction
    deformed_vertices: Optional[np.ndarray] = None
    written_files: list = field(default_factory=list)


class PipelineRunner:
    """Executes the pipeline stages selected by a :class:`PipelineConfig`."""

    def __init__(self, config: PipelineConfig, client=None):
        # the config client validates each fit spec; the stage adapter passes
        # its own so ${VAR} interpolation behaves the same everywhere
        self.client = client or ConfigClient()
        self.config = config

        # state shared between the steps
        self.shape_model: Optional[StatisticalShapeModel] = None
        self.shape: Optional[ShapeInstance] = None
        self.surface_vertices: Optional[np.ndarray] = None
        self.target_landmarks: dict = {}
        self.transformed_landmarks: dict = {}
        self.aligned_surface: Optional[np.ndarray] = None
        self.refined_vertices: Optional[np.ndarray] = None
        self.transform: SimilarityTransform = SimilarityTransform.identity()
        self.matches: Optional[Correspondences] = None
        self.template_vertices: Optional[np.ndarray] = None
        self.gmm_template_transform: Optional[SimilarityTransform] = None
        self.shapes: dict = {}
        self.vertices: dict = {}
        self.deformed_vertices: Optional[np.ndarray] = None
        self.written_files: list = []
        self.model_faces: Optional[np.ndarray] = None    # model's fixed triangulation
        self.surface_faces: Optional[np.ndarray] = None  # target mesh_vtp's own triangulation
        self.template_point_data = None  # inputs.template_vtp's PointData, see _save_points
        self._specs = None      # resolved nicp.fits, built on first use

    def run(self) -> PipelineResult:
        self.validate_plan()
        self.load_inputs()
        self.align_gmm_to_template()
        self.align_and_fit_landmarks()
        self.fit_to_surface()
        self.deform_region()

        return PipelineResult(
            shape=self.shape,
            surface_vertices=self.surface_vertices,
            aligned_surface=self.aligned_surface,
            refined_vertices=self.refined_vertices,
            transform=self.transform,
            matches=self.matches,
            template_vertices=self.template_vertices,
            gmm_template_transform=self.gmm_template_transform,
            shapes=self.shapes,
            vertices=self.vertices,
            deformed_vertices=self.deformed_vertices,
            written_files=self.written_files,
        )

    # -- plan validation ---------------------------------------------------

    def _planned_fits(self) -> list:
        """The resolved ``nicp.fits``, built once and cached.

        Resolving validates every fit, so this doubles as the config check and
        is why an unknown kwarg in the last fit fails before the first one runs.
        """
        if self._specs is None:
            self._specs = self.config.nicp.resolve_fits(self.client)
        return self._specs

    def planned_keys(self) -> list:
        """Registry keys this configuration will produce, in order."""
        cfg = self.config
        keys = [REFINED_KEY]
        if cfg.steps.nicp:
            keys += [spec.name for spec in self._planned_fits()]
        return keys

    def validate_plan(self):
        """Catches unreachable step/key combinations before any work is done."""
        cfg = self.config
        keys = self.planned_keys()

        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            raise ValueError(
                f"result key(s) {duplicates} are produced more than once; "
                f"rename the clashing nicp.variants entries"
            )

        if self._needs_landmarks() and cfg.inputs.landmarks_json is None:
            raise ValueError(
                "steps.align / steps.landmark_fit need inputs.landmarks_json, "
                "but it is not set"
            )

        if cfg.steps.gmm_2_template and cfg.gmm_2_template is None:
            raise ValueError(
                "steps.gmm_2_template is on but the gmm_2_template config block is missing"
            )

        # Assemble every fit before any of them runs. The FitProblem constructor
        # rejects a pairing-consuming loss with correspondence 'none', and the
        # closed-form solver rejects a loss it cannot solve, so an impossible
        # combination fails here rather than after the first fit has finished.
        if cfg.steps.nicp:
            # A stand-in basis: this runs before the model is loaded, and none of
            # the checks look at the numbers - only at which components were
            # named and whether they can work together.
            probe_space = ShapeSpace(np.zeros(12), np.zeros((12, 2)), np.ones(2),
                                     device="cpu", dtype="float32")
            probe_cloud = PointCloud(np.zeros((4, 3)))
            for spec in self._planned_fits():
                problem = build_problem(spec, probe_space, probe_cloud)
                build_solver(spec.solver.name, spec.solver.kwargs).validate(problem)

        if not cfg.steps.laplacian:
            return

        for name, key in (
            ("laplacian.template_key", cfg.laplacian.template_key),
            ("laplacian.target_key", cfg.laplacian.target_key),
            ("spectral.source_key", cfg.spectral.source_key),
        ):
            if key is None:
                continue
            if key not in keys:
                hint = "" if cfg.steps.nicp else " (steps.nicp is off, so no 'gamma_*' keys exist)"
                raise ValueError(
                    f"{name}='{key}' will not be produced by this run{hint}. "
                    f"Available: {keys}"
                )

        if cfg.laplacian.template_key == cfg.laplacian.target_key:
            self._log(
                f"WARNING: laplacian template_key and target_key are both "
                f"'{cfg.laplacian.template_key}'; the deformation will be a no-op."
            )

    # -- step 0: inputs ----------------------------------------------------

    def _needs_landmarks(self) -> bool:
        """The landmark JSON is only read by steps 1a and 1b."""
        return self.config.steps.align or self.config.steps.landmark_fit

    def load_inputs(self):
        cfg = self.config
        inputs = cfg.inputs

        required = ["shape_model_mat", "mesh_vtp"]
        if self._needs_landmarks():
            required.append("landmarks_json")
        for name in required:
            path = getattr(inputs, name)
            if path is None or not path.exists():
                raise FileNotFoundError(f"inputs.{name} does not exist: {path}")

        cfg.outputs.dir.mkdir(parents=True, exist_ok=True)

        surface = load_vtp(str(inputs.mesh_vtp))
        self.surface_vertices = get_vtp_vertices(surface)
        self.surface_faces = get_vtp_faces(surface)

        # One loader for every model format this project has: the BabyFM .mat,
        # a Deformetrica atlas directory, and a .npz GPMM built by
        # scripts.SSM.build_model (src/SSM/saved_models/<domain>/). Which one
        # inputs.shape_model_mat names changes nothing below - a model is
        # (mean, Phi, Lambda, landmarks) plus a triangulation either way.
        self.shape_model = load_model_file(
            inputs.shape_model_mat, normalize=cfg.model.normalize,
            atlas_options=cfg.model.atlas.to_options(), verbose=cfg.verbose,
        )
        self.model_faces = self.shape_model.faces
        mean = self.shape_model.mean
        eigenfunctions = self.shape_model.eigenfunctions
        eigenvalues = self.shape_model.eigenvalues

        if inputs.template_vtp is not None:
            if not inputs.template_vtp.exists():
                raise FileNotFoundError(f"inputs.template_vtp does not exist: {inputs.template_vtp}")
            template_polydata = load_vtp(str(inputs.template_vtp))
            n_model_vertices = mean.shape[0] // 3
            if template_polydata.GetNumberOfPoints() != n_model_vertices:
                raise ValueError(
                    f"inputs.template_vtp ({inputs.template_vtp}) has "
                    f"{template_polydata.GetNumberOfPoints()} points, but the model has "
                    f"{n_model_vertices} vertices - they must match to carry template_vtp's "
                    f"point-data fields onto the model's reconstructions"
                )
            self.template_point_data = template_polydata

        # Landmark-driven stages address the model by landmark name, which a
        # model without any annotation cannot answer - caught here rather than
        # as an empty match set two steps later.
        if self._needs_landmarks() and not len(self.shape_model.landmark_indices()):
            raise ValueError(
                f"{inputs.shape_model_mat.name} carries no landmarks, so steps.align "
                f"and steps.landmark_fit cannot run against it. A Deformetrica atlas - "
                f"and any GPMM built on one - is unannotated until model.atlas.landmarks "
                f"names a vertex-index JSON for it; set that, turn both steps off, or use "
                f"gmm_2_template/inputs.to_model_space to get into model space instead."
            )
        self._log("-------- Model Summary:")
        if cfg.verbose:
            self.shape_model.print_summary()
        self._log("-----------------------")

        self.shape = ShapeInstance(self.shape_model)
        coeffs = self.shape.fit(mean)
        init_shape = self.shape.reconstruct()

        self._log("--------Shape  Summary:")
        if cfg.verbose:
            self.shape.model.print_summary(coeffs)
        self._log("-----------------------")
        self._log("mean.shape: ", mean.shape)
        self._log("init_shape.shape: ", init_shape.shape)
        self._log("eigenfunction.shape: ", eigenfunctions.shape)
        self._log("eigenvalues.shape: ", eigenvalues.shape)
        self._log("Mean norm:", np.linalg.norm(mean))

        self._save_points(mean, "mean", faces=self.model_faces)
        self._save_points(init_shape, "init_shape", faces=self.model_faces)
        self._save_points(self.surface_vertices, "surface", faces=self.surface_faces)

        if self._needs_landmarks():
            self.target_landmarks, _, _ = load_landmarks(str(inputs.landmarks_json))

            # An atlas (e.g. FetusModel_x4) may only carry a hand-annotated
            # subset of the full landmark set (see model.atlas.landmarks). Fit
            # against whatever overlaps rather than failing on the rest.
            model_landmark_names = set(self.shape.get_landmark_data())
            unmatched = [n for n in self.target_landmarks if n not in model_landmark_names]
            if unmatched:
                self._log(
                    f"WARNING: {inputs.shape_model_mat.name} has no '{unmatched}' landmark(s); "
                    f"dropping from target_landmarks. Available: {sorted(model_landmark_names)}"
                )
                self.target_landmarks = {
                    n: v for n, v in self.target_landmarks.items() if n in model_landmark_names
                }

            self._log("target_landmarks: ", self.target_landmarks)
        else:
            self.target_landmarks = {}
            self._log("landmarks not loaded (steps.align and steps.landmark_fit are off)")

    # -- step 0b: GMM <-> template alignment -------------------------------

    def align_gmm_to_template(self):
        """Solves the similarity transform between GMM space and template space.

        The template is vertex-wise corresponding with the model, so the
        transform is a closed-form Procrustes fit over every vertex - no
        landmarks involved.
        """
        cfg = self.config
        self._log("----------Step 0b: (GMM <-> template alignment)")

        if not cfg.steps.gmm_2_template:
            self._log("skipped (steps.gmm_2_template is off)")
            return

        sub = cfg.gmm_2_template
        if not sub.template_vtp.is_file():
            raise FileNotFoundError(f"gmm_2_template.template_vtp does not exist: {sub.template_vtp}")

        template = get_vtp_vertices(load_vtp(str(sub.template_vtp)))
        model_verts = self.shape_model.mean.reshape(-1, 3)
        if template.shape[0] != model_verts.shape[0]:
            raise ValueError(
                f"gmm_2_template needs vertex-wise correspondence: template has "
                f"{template.shape[0]} points, the model has {model_verts.shape[0]}"
            )

        if sub.direction == "to_model":
            transform = SimilarityTransform.from_correspondences(
                template, model_verts, with_scale=sub.allow_scale
            )
            self.template_vertices = transform.apply_points(template)

            if sub.apply_to_target:
                self.surface_vertices = transform.apply_points(self.surface_vertices)
                if self.target_landmarks:
                    self.target_landmarks = transform.apply_points_dict(self.target_landmarks)
        else:
            transform = SimilarityTransform.from_correspondences(
                model_verts, template, with_scale=sub.allow_scale
            )
            self.template_vertices = template
            self._transform_model(transform)
            self._log(
                f"WARNING: direction=to_template rescales the SSM. The eigenvalues are "
                f"now variances in template units (x{transform.scale ** 2:.4g}); nicp.gammas must be "
                f"rescaled by the same factor, and chi_clamp compares against a "
                f"hardcoded bound that does not scale with them."
            )

        self.gmm_template_transform = transform
        residual = np.sqrt(((self.template_vertices - self.shape_model.mean.reshape(-1, 3)) ** 2).sum(1).mean())
        self._log(f"direction={sub.direction} scale={transform.scale:.6g} residual_rms={residual:.3e}")

        # Saved so scripts/sim_tf.py can replay it (or its inverse) onto other meshes.
        tf_path = cfg.outputs.path("gmm_2_template_transform")
        np.savez(str(tf_path), R=transform.rotation, t=transform.translation, s=transform.scale,
                 direction=sub.direction, template=str(sub.template_vtp))
        self.written_files.append(tf_path)
        self._log(f"similarity transform saved to {tf_path}")

        self._save_points(self.template_vertices, "template_aligned", faces=self.model_faces)

    def _transform_model(self, transform: SimilarityTransform):
        """Maps the SSM itself through a similarity transform.

        Mean is transformed as a point set; eigenfunctions are direction fields
        so they rotate and scale but do not translate; eigenvalues are variances
        and therefore scale quadratically.

        What comes out is the four-field contract plus the triangulation, and
        deliberately not a GPMM's extra self-knowledge: a kernel, its continuous
        inference and the ``mass`` inner product its basis is orthonormal in are
        all statements about the *untransformed* domain, and a similarity scale
        invalidates each of them (Phi^T M Phi picks up an s^4). Nothing in the
        fit reads them - ``ShapeInstance.closed_form_solution`` forms Phi^T Phi
        rather than assuming it away - so they are dropped rather than carried
        along with a label that has stopped being true.
        """
        model = self.shape_model
        K = model.n_components
        s, R = transform.scale, transform.rotation

        mean = transform.apply_points(model.mean.reshape(-1, 3)).reshape(-1)
        Phi = model.eigenfunctions.reshape(-1, 3, K)
        Phi = (s * np.einsum("nik,ji->njk", Phi, R)).reshape(-1, K)

        self.shape_model = StatisticalShapeModel(
            mean=mean,
            eigenfunctions=Phi,
            eigenvalues=model.eigenvalues * (s ** 2),
            landmarks=model.landmarks,
            faces=model.faces,
        )
        self.shape = ShapeInstance(self.shape_model)

    # -- step 1: alignment + landmark fit ----------------------------------

    def align_and_fit_landmarks(self):
        cfg = self.config
        steps = cfg.steps
        self._log("----------Step 1: (rigid alignment and model refinement)")

        # defaults for whatever is switched off: identity transform, mean shape
        self.transform = SimilarityTransform.identity()
        self.aligned_surface = np.asarray(self.surface_vertices)
        self.transformed_landmarks = dict(self.target_landmarks)

        if steps.align or steps.landmark_fit:
            self._check_landmark_names()
        else:
            self._log("skipped (steps.align and steps.landmark_fit are both off)")

        surface_landmarks = [np.asarray(v) for v in self.target_landmarks.values()]
        model_landmarks = self._current_model_landmarks() if steps.align else []

        for _ in range(cfg.rigid.iterations if (steps.align or steps.landmark_fit) else 0):
            if steps.align:
                self._log("estimate")
                self._log("surface_landmarks: ", surface_landmarks)
                self._log("curr_model_landmarks: ", model_landmarks)
                self.transform = SimilarityTransform.from_correspondences(
                    surface_landmarks, model_landmarks, with_scale=cfg.rigid.allow_scale
                )

                self._log("apply")
                self.aligned_surface = self.transform.apply_points(self.surface_vertices)
                self.transformed_landmarks = self.transform.apply_points_dict(self.target_landmarks)
                self._log("new target_points: ", self.transformed_landmarks)

            if steps.landmark_fit:
                self._log("get coeffs")
                self.shape.fit_landmarks(
                    self.transformed_landmarks,
                    learning_rate=cfg.rigid.landmark_fit_learning_rate,
                )

            if steps.align:
                model_landmarks = self._current_model_landmarks()

        self.refined_vertices = self.shape.reconstruct()
        self._log("refined_vertices.shape: ", self.refined_vertices.shape)
        self._register(REFINED_KEY, self.shape, self.refined_vertices)

        self._log("----------Save Results")
        self._save_points(self.aligned_surface, "rigid_alignment", faces=self.surface_faces)
        self._save_points(self.refined_vertices, "refined_vertices", faces=self.model_faces)
        if self.target_landmarks:
            self._save_points(
                np.array([np.asarray(v) for v in self.target_landmarks.values()]),
                "original_landmarks",
            )
        if self.transformed_landmarks:
            self._save_points(
                np.array([np.asarray(v) for v in self.transformed_landmarks.values()]),
                "transformed_landmarks",
            )

    def _check_landmark_names(self):
        landmarks = self.shape.get_landmark_data()
        missing = [name for name in self.target_landmarks if name not in landmarks]
        if missing:
            raise KeyError(
                f"target landmark(s) {missing} are not in the shape model. "
                f"Available: {sorted(landmarks)}"
            )

    def _current_model_landmarks(self) -> list:
        """Model landmark positions, in the order of the target landmark dict."""
        landmarks = self.shape.get_landmark_data()
        return [landmarks[name]["position"] for name in self.target_landmarks]

    # -- step 2: correspondence + NICP fits --------------------------------

    def fit_to_surface(self):
        cfg = self.config
        self._log("----------Step 2: (configured NICP fits)")

        if not cfg.steps.nicp:
            self._log("skipped (steps.nicp is off)")
            return

        self._log("refined_vertices.shape: ", self.refined_vertices.shape)
        self._log("aligned_surface.shape: ", self.aligned_surface.shape)

        # Built once each: neither the model basis nor the target moves between
        # fits, only the coefficients do.
        space = ShapeSpace.from_model(
            self.shape_model,
            faces=self.model_faces if cfg.nicp.use_model_faces else None,
            device=cfg.nicp.device, dtype=cfg.nicp.dtype,
        )
        target_cloud = PointCloud(self.aligned_surface.reshape(-1, 3),
                                  faces=self.surface_faces)

        specs = self._planned_fits()
        self._log(f"{len(specs)} fit(s): {[s.name for s in specs]}")
        for spec in specs:
            self._nicp_fit(spec, space, target_cloud)

    def _nicp_fit(self, spec, space, target_cloud):
        """One configured fit, starting fresh from the step 1 shape.

        Everything that distinguishes one fit from another - whether it pairs
        the surfaces at all, what it minimises, how it is solved - lives in the
        spec. This only runs it and writes the result, so reaching a
        correspondence-free objective is a config change and nothing else.
        """
        cfg = self.config
        problem = build_problem(spec, space, target_cloud,
                                alpha0=self.shape.alpha, seed=cfg.nicp.seed)
        solver = build_solver(spec.solver.name, spec.solver.kwargs)
        self._log(f"  [{spec.name}] {problem.describe()}")
        self._log(f"  [{spec.name}] solver       ={solver.describe()}")

        def on_iteration(iteration, alpha, terms):
            # iteration 0 is the shape the fit starts from, so the stride counts
            # from it rather than skipping it.
            if iteration % cfg.nicp.save_every:
                return
            self._save_points(space.to_numpy(alpha), "nicp_fit_iteration",
                              faces=self.model_faces, iteration=iteration, name=spec.name)

        result = solver.solve(problem, log=self._log if cfg.verbose else None,
                              on_iteration=on_iteration)

        vertices = space.to_numpy(result.alpha)
        shape_fit = ShapeInstance(self.shape_model,
                                  alpha=result.alpha.detach().cpu().numpy().astype(float))
        self._register(spec.name, shape_fit, vertices)
        self._save_points(vertices, "nicp_fit", faces=self.model_faces, name=spec.name)

        # A correspondence-free fit has no pairs to draw, so there is nothing to
        # write: the one place anything downstream notices which family it was.
        matches = problem.context.correspondences
        self.matches = matches
        if matches is not None and not matches.is_empty:
            path = cfg.outputs.path("nicp_fit_matching", name=spec.name)
            save_matching_field(vertices.reshape(-1, 3), target_cloud.points,
                                matches.as_array(), filename=str(path))
            self.written_files.append(path)

    # -- step 3: regional Laplacian deformation ----------------------------

    def deform_region(self):
        cfg = self.config
        self._log("----------Step 3: (Regional Laplace Constrained Deformations)")

        if not cfg.steps.laplacian:
            self._log("skipped (steps.laplacian is off)")
            return

        template_key = cfg.laplacian.template_key
        target_key = cfg.laplacian.target_key
        for key in (template_key, target_key):
            if key not in self.vertices:
                raise KeyError(
                    f"laplacian key '{key}' is not a pipeline result. "
                    f"Available: {sorted(self.vertices)}"
                )

        L, M, evals, evecs = self._spectral_operators(template_key)
        self._log("got operators")

        if cfg.spectral.test_spectrum:
            self._log("testing spectrum")
            test_LBO_spectrum(L, M, evals, evecs, cfg.spectral.k_eig)
        self._log("L.shape: ", L.shape)
        self._log("spectral_evecs.shape: ", evecs.shape)
        self._log("spectral_evals.shape: ", evals.shape)

        region = cfg.laplacian.region
        template_shape = self.shapes[template_key]
        S_target = template_shape.get_selection_matrix(lmk_name=region.landmark, radius=region.radius)
        region_verts = template_shape.region_vertices(lmk_name=region.landmark, radius=region.radius)
        self._log(f"region '{region.landmark}' (r={region.radius}): {len(region_verts)} vertices")
        if len(region_verts) == 0:
            raise ValueError(
                f"laplacian.region selects no vertices around '{region.landmark}'; "
                f"increase laplacian.region.radius"
            )
        self._save_points(region_verts, "region_vertices")

        template_vertices = self.vertices[template_key].reshape(-1, 3)
        target_vertices = self.vertices[target_key].reshape(-1, 3)

        self.deformed_vertices = solve_laplacian_deformation(
            template_vertices,
            target_vertices,
            S_target,
            L,
            lambda_reg=cfg.laplacian.lambda_reg,
        )
        self._save_points(self.deformed_vertices, "deformed_vertices", faces=self.model_faces)

    def _spectral_operators(self, template_key: str):
        cfg = self.config.spectral

        if not cfg.compute:
            if not cfg.path.is_file():
                raise FileNotFoundError(
                    f"spectral.path does not exist: {cfg.path}. "
                    f"Set spectral.compute: true to build the operators."
                )
            self._log("loading operators from", cfg.path)
            return load_spectral_data(str(cfg.path))

        source_key = cfg.source_key or template_key
        if source_key not in self.vertices:
            raise KeyError(
                f"spectral.source_key '{source_key}' is not a pipeline result. "
                f"Available: {sorted(self.vertices)}"
            )

        self._log("compute operators")
        L, M, evals, evecs = compute_spectral_operators(
            self.vertices[source_key].reshape(-1, 3),
            k_eig=cfg.k_eig,
            downsampling=cfg.downsampling,
        )

        if cfg.save:
            cfg.path.parent.mkdir(parents=True, exist_ok=True)
            save_spectral_data(str(cfg.path), L, M, evals, evecs)
            self.written_files.append(cfg.path)

        return L, M, evals, evecs

    # -- helpers -----------------------------------------------------------

    def _register(self, key: str, shape: ShapeInstance, vertices: np.ndarray):
        self.shapes[key] = shape
        self.vertices[key] = vertices

    def _save_points(self, points, key: str, faces=None, **fmt) -> Path:
        path = self.config.outputs.path(key, **fmt)
        vtp = create_polydata(np.asarray(points).reshape(-1, 3), faces=faces)
        # Every reconstruction the model produces shares its mean shape's vertex
        # count and ordering; anything else saved through here (e.g. a subset
        # like region_vertices) does not, so the count check also acts as the
        # filter for which outputs this applies to.
        if (
            self.template_point_data is not None
            and vtp.GetNumberOfPoints() == self.template_point_data.GetNumberOfPoints()
        ):
            copy_point_data(self.template_point_data, vtp)
        save_vtp(vtp, str(path))
        self.written_files.append(path)
        return path

    def _log(self, *args):
        if self.config.verbose:
            print(*args)
