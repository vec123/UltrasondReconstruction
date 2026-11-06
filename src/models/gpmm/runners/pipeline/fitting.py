"""Steps 2 and 3 of the morphing pipeline runner: the correspondence + NICP fits,
then the regional Laplacian deformation."""

from __future__ import annotations

from src.models.gpmm.laplacian.laplacian import solve_laplacian_deformation
from src.models.gpmm.model.model import ShapeInstance
from src.models.gpmm.nicp import PointCloud
from src.models.gpmm.nicp.problem import build_problem
from src.models.gpmm.nicp.shape import ShapeSpace
from src.models.gpmm.nicp.solvers import build_solver
from src.shared.correspondence.vtk import save_correspondence_field, save_matching_lines
from src.shared.geometry.spectral_operators import (
    compute_spectral_operators,
    load_spectral_data,
    save_spectral_data,
    test_LBO_spectrum,
)
from src.shared.vtk.create import create_polydata


class FittingSteps:
    """Steps 2 and 3 of runner.PipelineRunner, a mixin: they use its state (self.config, self.shapes, ...)."""

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
        recon_file = self._save_points(vertices, "nicp_fit", faces=self.model_faces, name=spec.name)

        # A correspondence-free fit has no pairs to draw, so there is nothing to
        # write: the one place anything downstream notices which family it was.
        matches = problem.context.correspondences
        self.matches = matches
        if matches is not None and not matches.is_empty:
            path = cfg.outputs.path("nicp_fit_matching", name=spec.name)
            save_matching_lines(vertices.reshape(-1, 3), target_cloud.points,
                                matches.as_array(), filename=str(path))
            self.written_files.append(path)

            src_colors_path = cfg.outputs.path("nicp_fit_colors", name=spec.name)
            tgt_colors_path = cfg.outputs.path("nicp_fit_target_colors", name=spec.name)
            self.written_files.extend(save_correspondence_field(
                recon_file, create_polydata(target_cloud.points, self.surface_faces), matches,
                target_path=tgt_colors_path, recon_path=src_colors_path,
            ))

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
