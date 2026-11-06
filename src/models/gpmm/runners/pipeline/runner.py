"""The morphing pipeline runner: the plan, the inputs and the order of the steps (see __init__.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.models.gpmm.config import PipelineConfig
from src.models.gpmm.model.model import ShapeInstance, StatisticalShapeModel
from src.models.gpmm.model.model_memory_io import load_model_file
from src.models.gpmm.nicp import Correspondences, PointCloud
from src.models.gpmm.nicp.problem import build_problem
from src.models.gpmm.nicp.shape import ShapeSpace
from src.models.gpmm.nicp.solvers import build_solver
from src.models.gpmm.runners.pipeline.alignment import AlignmentSteps
from src.models.gpmm.runners.pipeline.fitting import FittingSteps
from src.models.gpmm.runners.pipeline.results import REFINED_KEY, PipelineResult, write_fit_results
from src.shared.core.config import ConfigClient
from src.shared.landmarks.load_landmarks import load_landmarks
from src.shared.transforms.similarity import SimilarityTransform
from src.shared.vtk.create import create_polydata
from src.shared.vtk.extract import get_vtp_faces, get_vtp_vertices
from src.shared.vtk.fields import copy_point_data
from src.shared.vtk.io import load_vtp, save_vtp


class PipelineRunner(AlignmentSteps, FittingSteps):
    """Executes the pipeline stages selected by a :class:`PipelineConfig`.

    Steps 0b and 1 are in alignment.py, steps 2 and 3 in fitting.py; all of
    them use the state that __init__ sets up here."""

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
        write_fit_results(self.config.outputs.dir, self.planned_keys(), self.config.inputs.shape_model_mat)

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
        # the ssm_build stage (assets/shape_models/<domain>/). Which one
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
