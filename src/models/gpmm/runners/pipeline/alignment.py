"""Steps 0b and 1 of the morphing pipeline runner: the GMM <-> template alignment,
then the landmark alignment and the landmark fit of the model."""

from __future__ import annotations

import numpy as np

from src.models.gpmm.model.model import ShapeInstance, StatisticalShapeModel
from src.models.gpmm.runners.pipeline.results import REFINED_KEY
from src.shared.transforms.similarity import SimilarityTransform
from src.shared.vtk.extract import get_vtp_vertices
from src.shared.vtk.io import load_vtp


class AlignmentSteps:
    """Steps 0b and 1 of runner.PipelineRunner, a mixin: they use its state (self.config, self.shape_model, ...)."""

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

        # Saved so it can be replayed (or inverted) onto other meshes.
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
