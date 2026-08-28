"""Runner for the NICP step on its own: config in, fitted surfaces out.

Every entry of ``runs`` is merged onto ``defaults``, validated, and fitted
against the same model and target. All of them are resolved before any fitting
starts, so a typo in the last run fails immediately rather than after an hour
of sweeping.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from src.config.client import ConfigClient
from src.config.schema import NicpFitConfig, NicpFitSpec
from src.correspondence.vtk import save_matching_field
from src.fitting.rigid import (
    apply_similarity_transform,
    estimate_similarity_transform,
    invert_similarity_transform,
    load_similarity_npz,
)
from src.geometry.vtk import (
    create_polydata,
    get_vtp_faces,
    get_vtp_vertices,
    load_vtp_mesh,
    save_vtp_mesh,
    transform_polydata,
)
from src.model.load import load_shape_model, load_shape_model_triangulation
from src.nicp.cloud import PointCloud
from src.nicp.correspondence import build_correspondence
from src.nicp.losses import build_loss
from src.nicp.metrics import coefficient_metrics, displacement_metrics, surface_metrics
from src.nicp.priors import build_prior
from src.nicp.problem import FitProblem
from src.nicp.projections import build_projection
from src.nicp.shape import ShapeSpace
from src.nicp.solvers import build_solver


def deep_merge(base: Mapping, override: Mapping) -> Dict[str, Any]:
    """Recursive dict merge; ``override`` wins on leaves.

    One special case: a block that names an implementation is replaced wholesale
    when the name changes, instead of being merged. ``kwargs`` belong to the
    implementation, not to the slot it sits in, so a run that switches from
    ``normal_equations`` to ``gradient`` must not inherit the former's
    ``learning_rate``.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            if value.get("name", current.get("name")) != current.get("name"):
                merged[key] = dict(value)
            else:
                merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


@dataclass
class RunResult:
    name: str
    alpha: np.ndarray
    vertices: np.ndarray                  # (3N,) reconstruction
    metrics: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, float]] = field(default_factory=list)
    n_matches: Optional[int] = None
    seconds: float = 0.0
    converged: bool = False
    message: str = ""


@dataclass
class NicpResult:
    runs: Dict[str, RunResult] = field(default_factory=dict)
    written_files: List[Path] = field(default_factory=list)


class NicpRunner:
    """Fits the shape model to one target surface, once per configured run."""

    def __init__(self, config: NicpFitConfig, client: Optional[ConfigClient] = None):
        self.config = config
        self.client = client or ConfigClient()

        self.space: Optional[ShapeSpace] = None
        self.target: Optional[PointCloud] = None
        self.alpha0: Optional[torch.Tensor] = None
        self.initial_vertices: Optional[np.ndarray] = None
        self.runs: Dict[str, RunResult] = {}
        self.written_files: List[Path] = []

        # inputs kept as polydata so topology survives into the output folders
        self.mesh_world = None
        self.mesh_model = None
        self.template_world = None
        self.template_model = None
        self.transform = None            # world -> model space
        self._inverse_transform = None   # model -> world, built on first use
        self.transform_source = "none"

    # -- plan --------------------------------------------------------------

    def resolve_runs(self) -> List[NicpFitSpec]:
        """Merges every run onto the defaults and validates the result."""
        cfg = self.config
        if not cfg.runs:
            raise ValueError("no runs configured; add at least one entry under 'runs'")

        specs, seen = [], set()
        for index, run in enumerate(cfg.runs):
            if not isinstance(run, Mapping):
                raise ValueError(f"runs[{index}] must be a mapping, got {type(run).__name__}")
            if "name" not in run:
                raise ValueError(f"runs[{index}] is missing 'name'")

            merged = deep_merge(cfg.defaults, run)
            spec = self.client.from_dict(merged, NicpFitSpec, source=f"runs[{index}]")
            if spec.name in seen:
                raise ValueError(f"duplicate run name '{spec.name}'; names become file names")
            seen.add(spec.name)
            specs.append(spec)
        return specs

    def build_problem(self, spec: NicpFitSpec) -> FitProblem:
        return FitProblem(
            space=self.space,
            target=self.target,
            correspondence=build_correspondence(spec.correspondence.name, spec.correspondence.kwargs),
            loss=build_loss(spec.loss.name, spec.loss.kwargs),
            prior=build_prior(spec.prior.name, spec.prior.kwargs),
            projection=build_projection(spec.projection.name, spec.projection.kwargs),
            alpha0=self.alpha0,
            refresh_every=spec.refresh_every,
            seed=self.config.compute.seed,
        )

    # -- inputs ------------------------------------------------------------

    def load_inputs(self):
        cfg = self.config
        for name in ("shape_model_mat", "mesh_vtp"):
            path = getattr(cfg.inputs, name)
            if not path.exists():
                raise FileNotFoundError(f"inputs.{name} does not exist: {path}")

        cfg.outputs.dir.mkdir(parents=True, exist_ok=True)

        mean, eigenfunctions, eigenvalues, _ = load_shape_model(
            cfg.inputs.shape_model_mat, normalize=cfg.model.normalize,
            atlas_options=cfg.model.atlas.to_options(), verbose=cfg.verbose,
        )
        faces = load_shape_model_triangulation(cfg.inputs.shape_model_mat)
        if not cfg.compute.use_model_faces:
            faces = None
        self.space = ShapeSpace(
            mean, eigenfunctions, eigenvalues, faces=faces,
            device=cfg.compute.device, dtype=cfg.compute.dtype,
        )
        self._log(f"model:  {self.space}")
        self._log(f"        {0 if faces is None else len(faces)} triangles "
                  f"({'surface' if faces is not None else 'point cloud only'})")

        self.target = PointCloud(*self._load_target())
        self.alpha0 = self._initial_alpha()
        self.initial_vertices = self.space.to_numpy(self.alpha0)
        self._save_inputs()

    def _load_target(self):
        """The target surface and its topology, mapped into model space."""
        inputs = self.config.inputs
        self.mesh_world = load_vtp_mesh(str(inputs.mesh_vtp))
        faces = get_vtp_faces(self.mesh_world)
        self._log(f"target: {self.mesh_world.GetNumberOfPoints()} points, "
                  f"{0 if faces is None else len(faces)} triangles from {inputs.mesh_vtp.name}")

        self.transform = self._model_space_transform()
        if self.transform is None:
            self._log("        no transform applied - mesh_vtp is taken to be in model space")
            self.mesh_model = self.mesh_world
            return get_vtp_vertices(self.mesh_world), faces

        R, t, s = self.transform
        self.mesh_model = transform_polydata(self.mesh_world, R, t, s)
        verts = get_vtp_vertices(self.mesh_model)
        extent = verts.max(0) - verts.min(0)
        self._log(f"        mapped into model space (scale {s:.6g}), "
                  f"extent now {np.array2string(extent, precision=4)}")
        return verts, faces

    def _save_inputs(self):
        """Writes what the fit was actually given, in both frames.

        The fit happens in model space while the data lives in world space, and
        a wrong transform is invisible in the numbers but obvious the moment the
        two are opened together. Topology is preserved throughout: the target
        keeps its own triangles, and the template borrows the model's, since it
        is vertex-wise corresponding.
        """
        outputs = self.config.outputs
        space_cfg = self.config.inputs.to_model_space

        self._write_polydata(self.mesh_world, outputs.path("world_mesh"))
        self._write_polydata(self.mesh_model, outputs.path("model_mesh"))

        if space_cfg.template_vtp is not None and self.template_world is not None:
            self._write_polydata(self.template_world, outputs.path("world_template"))
            self._write_polydata(self.template_model, outputs.path("model_template"))

        self._write_mesh(self.space.to_numpy(self.space.zeros()), outputs.path("model_mean"))
        self._write_mesh(self.initial_vertices, outputs.path("model_init"))

        # Written whether it was solved here or loaded, so the output folder is
        # self-contained: scripts/sim_tf.py can replay it (or its inverse) onto
        # any other mesh without needing to know where it came from.
        if self.transform is not None and space_cfg.save_transform:
            R, t, s = self.transform
            path = outputs.path("transform")
            np.savez(str(path), R=R, t=np.asarray(t).flatten(), s=s,
                     direction="to_model", source=self.transform_source)
            self.written_files.append(path)
            self._log(f"        transform saved to {path.name} ({self.transform_source})")

    def _write_polydata(self, polydata, path: Path, verbose: bool = True):
        save_vtp_mesh(polydata, str(path), verbose=verbose)
        self.written_files.append(path)

    def _write_mesh(self, vertices, path: Path):
        """A model instance, as a surface when the model brought a triangulation."""
        save_vtp_mesh(create_polydata(np.asarray(vertices).reshape(-1, 3), self.space.faces), str(path))
        self.written_files.append(path)

    def _as_model_mesh(self, vertices):
        return create_polydata(np.asarray(vertices).reshape(-1, 3), self.space.faces)

    def _to_world(self, polydata):
        """Maps a model-space mesh back to the frame the inputs arrived in.

        Returns None when no transform was applied on the way in, in which case
        the two frames coincide and a second copy would be redundant.
        """
        if self.transform is None:
            return None
        if self._inverse_transform is None:
            self._inverse_transform = invert_similarity_transform(*self.transform)
        return transform_polydata(polydata, *self._inverse_transform)

    def _model_space_transform(self):
        """Solves the transform from the template, or loads a saved one."""
        space_cfg = self.config.inputs.to_model_space

        if space_cfg.template_vtp is not None:
            if not space_cfg.template_vtp.is_file():
                raise FileNotFoundError(
                    f"inputs.to_model_space.template_vtp does not exist: {space_cfg.template_vtp}"
                )
            # the template is vertex-corresponding with the model, so it is a
            # surface once the model's triangulation is attached to it
            template = get_vtp_vertices(load_vtp_mesh(str(space_cfg.template_vtp)))
            self.template_world = create_polydata(template, self.space.faces)
            model_verts = self.space.mean.reshape(-1, 3).cpu().numpy()
            if template.shape[0] != model_verts.shape[0]:
                raise ValueError(
                    f"to_model_space.template_vtp needs vertex-wise correspondence with the "
                    f"model: template has {template.shape[0]} points, the model has "
                    f"{model_verts.shape[0]}"
                )

            R, t, s = estimate_similarity_transform(
                template, model_verts, allow_scale=space_cfg.allow_scale
            )
            self.template_model = transform_polydata(self.template_world, R, t, s)
            residual = np.sqrt(
                ((get_vtp_vertices(self.template_model) - model_verts) ** 2).sum(1).mean()
            )
            self._log(f"        solved from {space_cfg.template_vtp.name}: "
                      f"scale {s:.6g}, residual rms {residual:.3e} "
                      f"(how well the template maps onto the model)")
            self.transform_source = f"solved from {space_cfg.template_vtp.name}"
            return R, t, s

        if space_cfg.transform_npz is not None:
            if not space_cfg.transform_npz.is_file():
                raise FileNotFoundError(
                    f"inputs.to_model_space.transform_npz does not exist: {space_cfg.transform_npz}"
                )
            R, t, s = load_similarity_npz(space_cfg.transform_npz)
            if space_cfg.inverse:
                R, t, s = invert_similarity_transform(R, t, s)
            self._log(f"        loaded from {space_cfg.transform_npz.name}"
                      f"{' (inverted)' if space_cfg.inverse else ''}")
            self.transform_source = f"loaded from {space_cfg.transform_npz.name}"
            return R, t, s

        return None

    def _initial_alpha(self) -> torch.Tensor:
        """Where the fit starts: the mean, a saved alpha, or a projected mesh."""
        inputs = self.config.inputs

        if inputs.init_alpha_npz is not None:
            if not inputs.init_alpha_npz.is_file():
                raise FileNotFoundError(f"inputs.init_alpha_npz does not exist: {inputs.init_alpha_npz}")
            data = np.load(str(inputs.init_alpha_npz))
            if "alpha" not in data:
                raise KeyError(f"{inputs.init_alpha_npz} has no 'alpha' array (found {list(data)})")
            self._log(f"start:  alpha from {inputs.init_alpha_npz.name}")
            return self.space.as_alpha(data["alpha"])

        if inputs.init_vtp is not None:
            if not inputs.init_vtp.is_file():
                raise FileNotFoundError(f"inputs.init_vtp does not exist: {inputs.init_vtp}")
            verts = get_vtp_vertices(load_vtp_mesh(str(inputs.init_vtp)))
            if verts.shape[0] != self.space.n_vertices:
                raise ValueError(
                    f"inputs.init_vtp has {verts.shape[0]} points, the model has "
                    f"{self.space.n_vertices}; it must be a model instance"
                )
            # orthonormal eigenfunctions, so the projection is a plain transpose
            flat = torch.as_tensor(verts.reshape(-1), dtype=self.space.dtype, device=self.space.device)
            self._log(f"start:  {inputs.init_vtp.name} projected onto the model basis")
            return self.space.Phi.T @ (flat - self.space.mean)

        self._log("start:  mean shape")
        return self.space.zeros()

    # -- run ---------------------------------------------------------------

    def run(self, only: Optional[List[str]] = None) -> NicpResult:
        specs = self.resolve_runs()
        if only:
            missing = sorted(set(only) - {s.name for s in specs})
            if missing:
                raise ValueError(f"unknown run name(s) {missing}. Configured: {[s.name for s in specs]}")
            specs = [s for s in specs if s.name in only]

        self.load_inputs()

        for spec in specs:
            self.runs[spec.name] = self.fit(spec)

        self._write_report()
        return NicpResult(runs=self.runs, written_files=self.written_files)

    def fit(self, spec: NicpFitSpec) -> RunResult:
        self._log(f"\n---------- run '{spec.name}'")
        problem = self.build_problem(spec)
        self._log(f"  {problem.describe()}")

        solver = build_solver(spec.solver.name, spec.solver.kwargs)
        self._log(f"  solver       ={solver.describe()}")

        started = time.time()
        result = solver.solve(
            problem,
            log=self._log if self.config.verbose else None,
            on_iteration=self._iteration_writer(spec),
        )
        seconds = time.time() - started

        alpha = result.alpha.detach()
        vertices = self.space.to_numpy(alpha)
        corr = problem.context.correspondences

        metrics = {}
        metrics.update(surface_metrics(vertices.reshape(-1, 3), self.target.points, self.target.tree))
        metrics.update(coefficient_metrics(alpha.cpu().numpy(),
                                           self.space.eigenvalues.cpu().numpy()))
        metrics.update(displacement_metrics(vertices, self.initial_vertices))
        metrics["objective"] = result.final.get("objective", float("nan"))
        metrics["iterations"] = result.iterations
        metrics["seconds"] = seconds

        self._log(f"  -> objective={metrics['objective']:.6e} "
                  f"model->target rms={metrics['model_to_target_rms']:.3e} "
                  f"target->model rms={metrics['target_to_model_rms']:.3e} "
                  f"mahalanobis={metrics['mahalanobis']:.3g} ({seconds:.1f}s, {result.message})")

        self._save_run(spec, alpha, vertices, corr)

        return RunResult(
            name=spec.name, alpha=alpha.cpu().numpy(), vertices=vertices, metrics=metrics,
            history=result.history if self.config.outputs.save_history else [],
            n_matches=None if corr is None else len(corr),
            seconds=seconds, converged=result.converged, message=result.message,
        )

    # -- outputs -----------------------------------------------------------

    def _iteration_writer(self, spec: NicpFitSpec):
        """Snapshot callback for the solver, or None when snapshots are off.

        Silent by design: the solver calls this on every step, so it must not
        print. The file count is reported once the run finishes.
        """
        outputs = self.config.outputs
        if not outputs.save_iterations:
            return None

        def write(iteration, alpha, terms):
            if iteration % outputs.iteration_stride:
                return
            polydata = self._as_model_mesh(self.space.to_numpy(alpha))
            self._write_polydata(
                polydata, outputs.run_path("iteration", run=spec.name, iteration=iteration),
                verbose=False,
            )
            world = self._to_world(polydata) if outputs.save_world else None
            if world is not None:
                self._write_polydata(
                    world, outputs.run_path("iteration_world", run=spec.name, iteration=iteration),
                    verbose=False,
                )

        return write

    def _save_run(self, spec, alpha, vertices, corr):
        outputs = self.config.outputs

        polydata = self._as_model_mesh(vertices)
        self._write_polydata(polydata, outputs.run_path("fit", run=spec.name))

        world = self._to_world(polydata) if outputs.save_world else None
        if world is not None:
            self._write_polydata(world, outputs.run_path("fit_world", run=spec.name))

        if outputs.save_alpha:
            path = outputs.run_path("alpha", run=spec.name)
            np.savez(str(path), alpha=alpha.cpu().numpy(), run=spec.name)
            self.written_files.append(path)

        if outputs.save_matching_field and corr is not None and not corr.is_empty:
            path = outputs.run_path("matching_field", run=spec.name)
            save_matching_field(vertices.reshape(-1, 3), self.target.points,
                                corr.as_array(), filename=str(path))
            self.written_files.append(path)

    def _write_report(self):
        if not self.runs:
            return
        outputs = self.config.outputs

        columns = ["run", "matches", "converged"]
        for result in self.runs.values():
            for key in result.metrics:
                if key not in columns:
                    columns.append(key)

        path = outputs.path("metrics")
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for result in self.runs.values():
                row = {"run": result.name, "matches": result.n_matches,
                       "converged": result.converged}
                row.update(result.metrics)
                writer.writerow(row)
        self.written_files.append(path)

        path = outputs.path("summary")
        summary = {
            "target": str(self.config.inputs.mesh_vtp),
            "model": str(self.config.inputs.shape_model_mat),
            "device": str(self.space.device),
            "runs": {
                name: {
                    "metrics": result.metrics,
                    "matches": result.n_matches,
                    "converged": result.converged,
                    "message": result.message,
                    "history": result.history,
                }
                for name, result in self.runs.items()
            },
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self.written_files.append(path)

        self._log(f"\nwrote {len(self.written_files)} file(s) to {outputs.dir}")

    def _log(self, *args):
        if self.config.verbose:
            print(*args)
