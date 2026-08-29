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
from src.vtk.extract import (
    get_vtp_faces,
    get_vtp_vertices
)
from src.vtk.create import (
    create_polydata
)

from src.SSM.laplacian.laplacian import solve_laplacian_deformation
from src.SSM.model.load import load_shape_model, load_shape_model_triangulation
from src.SSM.model.model import ShapeInstance, StatisticalShapeModel
from src.SSM.nicp import Correspondences, PointCloud, build_correspondence

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

    def __init__(self, config: PipelineConfig):
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
        self._strategy = None   # correspondence method, built on first use

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

    def planned_keys(self) -> list:
        """Registry keys this configuration will produce, in order."""
        cfg = self.config
        keys = [REFINED_KEY]
        if cfg.steps.nicp:
            keys += [f"gamma_{self._fmt_gamma(g)}" for g in cfg.nicp.model_optimization.gammas]
            keys += [v.name for v in cfg.nicp.model_optimization.variants]
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

        # The MAP step solves against a fixed set of matched vertices, so it has
        # no way to associate the surfaces on its own.
        if cfg.steps.nicp and not self._correspondence_strategy().produces_matches:
            raise ValueError(
                f"nicp.matching.mode='{cfg.nicp.matching.mode}' produces no correspondences, but "
                f"the pipeline's NICP step fits against a fixed match set. Use 'nearest' or "
                f"'constrained' here; correspondence-free fitting needs a data term that "
                f"associates the surfaces itself."
            )

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

        mean, eigenfunctions, eigenvalues, model_landmarks = load_shape_model(
            inputs.shape_model_mat, normalize=cfg.model.normalize,
            atlas_options=cfg.model.atlas.to_options(), verbose=cfg.verbose,
        )
        self.model_faces = load_shape_model_triangulation(inputs.shape_model_mat)

        self.shape_model = StatisticalShapeModel(
            mean=mean,
            eigenfunctions=eigenfunctions,
            eigenvalues=eigenvalues,
            landmarks=model_landmarks,
        )

        # Landmark-driven stages address the model by landmark name, which a
        # model without any annotation cannot answer - caught here rather than
        # as an empty match set two steps later.
        if self._needs_landmarks() and not len(self.shape_model.landmark_indices()):
            raise ValueError(
                f"{inputs.shape_model_mat.name} carries no landmarks, so steps.align "
                f"and steps.landmark_fit cannot run against it. Turn both off (a "
                f"Deformetrica atlas is unannotated), or use "
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
        self._log("----------Step 2: (correspondence + model-optimization NICP loop)")

        if not cfg.steps.nicp:
            self._log("skipped (steps.nicp is off)")
            return

        self._log("refined_vertices.shape: ", self.refined_vertices.shape)
        self._log("aligned_surface.shape: ", self.aligned_surface.shape)

        strategy = self._correspondence_strategy()
        self._log("correspondence:", strategy.describe())
        self._log(f"iterations (match -> fit rounds) per gamma/variant: {cfg.nicp.iterations}")

        # Built once: the target does not move between rounds, only the model does.
        target_cloud = PointCloud(self.aligned_surface.reshape(-1, 3))

        opt = cfg.nicp.model_optimization
        for gamma in opt.gammas:
            self._nicp_fit(
                key=f"gamma_{self._fmt_gamma(gamma)}",
                gamma=gamma,
                learning_rate=opt.learning_rate,
                chi_clamp=opt.chi_clamp,
                out_key="nicp_gamma",
                out_fmt={"gamma": self._fmt_gamma(gamma)},
                target_cloud=target_cloud,
            )

        for variant in opt.variants:
            self._nicp_fit(
                key=variant.name,
                gamma=variant.gamma,
                learning_rate=variant.learning_rate,
                chi_clamp=variant.chi_clamp,
                out_key="nicp_variant",
                out_fmt={"name": variant.name},
                target_cloud=target_cloud,
            )

    def _correspondence_strategy(self):
        """The matcher selected by the ``nicp.matching`` block, built once and cached."""
        if self._strategy is None:
            spec = self.config.nicp.matching.to_spec()
            self._strategy = build_correspondence(spec.name, spec.kwargs)
        return self._strategy

    def _nicp_fit(self, key, gamma, learning_rate, chi_clamp, out_key, out_fmt, target_cloud):
        """One gamma/variant's own match -> fit loop, starting fresh from step 1's shape.

        Each round re-estimates correspondences against the shape as it
        currently stands, then does one MAP-regularised solve at this entry's
        fixed gamma - the standard chicken-and-egg non-rigid ICP alternation,
        just with the regularisation strength held constant across rounds
        instead of annealed. ``nicp.iterations: 1`` reduces this to the
        original single-shot fit.
        """
        cfg = self.config
        strategy = self._correspondence_strategy()
        shape_fit = deepcopy(self.shape)
        matches = None

        for iteration in range(cfg.nicp.iterations):
            current_verts = shape_fit.reconstruct().reshape(-1, 3)
            matches = strategy.estimate(current_verts, target_cloud)
            self._log(f"  [{key}] iter {iteration + 1}/{cfg.nicp.iterations}: {matches.summary()}")

            if matches.is_empty:
                raise RuntimeError(
                    f"no valid matches found for '{key}' at iteration {iteration + 1}; raise "
                    f"nicp.matching.dist_thresh / nicp.matching.normal_thresh_deg, or use "
                    f"nicp.matching.mode: nearest"
                )

            shape_fit.fit_to_matching(
                self.aligned_surface,
                matches.as_array(),
                learning_rate=learning_rate,
                gamma=gamma,
                chi_clamp=chi_clamp,
            )

            is_last = iteration == cfg.nicp.iterations - 1
            if (iteration + 1) % cfg.nicp.save_every == 0 and not is_last:
                self._save_points(
                    shape_fit.reconstruct(), f"{out_key}_iteration",
                    faces=self.model_faces, iteration=iteration, **out_fmt,
                )

        vertices = shape_fit.reconstruct()
        self.matches = matches

        self._register(key, shape_fit, vertices)
        self._save_points(vertices, out_key, faces=self.model_faces, **out_fmt)

        matching_path = cfg.outputs.path(f"{out_key}_matching", **out_fmt)
        save_matching_field(
            vertices.reshape(-1, 3), target_cloud.points,
            matches.as_array(), filename=str(matching_path),
        )
        self.written_files.append(matching_path)

    @staticmethod
    def _fmt_gamma(gamma) -> str:
        return str(int(gamma)) if float(gamma).is_integer() else str(gamma)

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
        save_vtp(vtp, str(path))
        self.written_files.append(path)
        return path

    def _log(self, *args):
        if self.config.verbose:
            print(*args)
