"""The shape-space explorer: the model, the state of the sliders, export (see __init__.py)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import vtk
from vtk.util import numpy_support

from src.models.gpmm.config import SampleShapesConfig
from src.models.gpmm.model.sources.load import load_shape_model, load_shape_model_triangulation
from src.models.gpmm.runners.sample_shapes.dropdown import DropdownMixin
from src.models.gpmm.runners.sample_shapes.events import EventsMixin
from src.models.gpmm.runners.sample_shapes.scene import SceneMixin
from src.models.gpmm.runners.sample_shapes.style import _THEMES
from src.models.gpmm.runners.sample_shapes.widgets import WidgetsMixin
from src.shared.vtk.create import create_polydata


@dataclass
class SampleShapesResult:
    """The state the window was left in."""

    alpha: np.ndarray                                  # (K,) coefficients, model units
    sigmas: np.ndarray                                 # (K,) the same, in sigma
    vertices: np.ndarray                               # (N, 3) the shape on screen
    mahalanobis: float
    written_files: List[Path] = field(default_factory=list)


class SampleShapesRunner(SceneMixin, WidgetsMixin, DropdownMixin, EventsMixin):
    """Builds the window, wires the sliders to the model, and runs it.

    The other parts of the window are mixins (scene.py, widgets.py, dropdown.py,
    events.py); all of them use the state that __init__ sets up here."""

    def __init__(self, config: SampleShapesConfig):
        self.config = config

        self.mean: Optional[np.ndarray] = None        # (N, 3)
        self.basis: Optional[np.ndarray] = None       # (3N, K) the exposed modes only
        self.sigma: Optional[np.ndarray] = None       # (K,) sqrt(eigenvalue) per mode
        self.alpha: Optional[np.ndarray] = None       # (K,) current coefficients
        self.faces: Optional[np.ndarray] = None
        self.total_std: Optional[np.ndarray] = None   # (N,)

        self.theme_name = config.view.theme
        self.theme = _THEMES[self.theme_name]
        # Redraw throttle, only used where a sample costs an integration.
        self._min_interval = 0.15 if config.source().parametrization == "lddmm" else 0.0
        self._last_update = 0.0
        self.sliders: List[vtk.vtkSliderWidget] = []
        self.button_label: Optional[vtk.vtkTextActor] = None
        self.color_button_label: Optional[vtk.vtkTextActor] = None
        self.save_button_label: Optional[vtk.vtkTextActor] = None
        self.scalar_bar: Optional[vtk.vtkScalarBarActor] = None
        self.ghost: Optional[vtk.vtkActor] = None
        self.written_files: List[Path] = []

        # What the surface is coloured by *now*. The config supplies the
        # opening value; 'c' and the COLOR button move it from there.
        self.color_mode = config.view.color_by
        # Outward vertex normals of the mean shape, the reference the signed
        # mode takes its sign against. None when the model has no faces.
        self.mean_normals: Optional[np.ndarray] = None
        # Per-vertex sd of the *normal component*, the signed mode's analogue
        # of total_std and the scale its diverging ramp is fixed against.
        self.normal_std: Optional[np.ndarray] = None

        # What the model itself reports about its own truncation, filled in by
        # load_model() and shown in the info line under the dropdown.
        self.noise = 0.0
        self.variance_explained = 1.0

        # The model dropdown. 'names' is fixed for the session; 'open' and
        # 'first' are its runtime state.
        self.model_names: List[str] = config.model_names()
        self.dropdown_open = False
        self.dropdown_first = 0
        self.dropdown_rows: List[tuple] = []      # (quad actor, points, text actor)
        self.dropdown_bar: Optional[vtk.vtkActor2D] = None
        self.dropdown_label: Optional[vtk.vtkTextActor] = None
        self._dropdown_rects: List[tuple] = []    # pixel hit boxes, parallel to rows
        self._bar_rect = (0, 0, 0, 0)

    def run(self) -> SampleShapesResult:
        self.load_model()
        self.build_scene()
        self.build_widgets()

        self.reset_to_mean()
        self.render_window.Render()

        if self.config.screenshot is not None:
            self.save_screenshot(self.config.screenshot)

        if self.config.interactive:
            self._log(f"window open - {len(self.model_names)} model(s) in the dropdown "
                      f"('m' opens it, '[' / ']' step through them), drag the sliders, "
                      f"scroll the panel for more modes, 'c' cycles the colour mode "
                      f"(displacement / signed / per-vertex sd / off), 's' saves the "
                      f"current shape as .vtp, 'r' resets to the mean, 't' switches theme")
            self.interactor.Start()

        return SampleShapesResult(
            alpha=self.alpha.copy(),
            sigmas=self.alpha / self.sigma,
            vertices=self.sample(),
            mahalanobis=self._mahalanobis(),
            written_files=self.written_files,
        )

    # -- the model ---------------------------------------------------------

    def load_model(self):
        cfg = self.config
        source = cfg.source()
        if not source.path.exists():
            raise FileNotFoundError(f"models.{cfg.model}.path does not exist: {source.path}")

        self._log(f"model '{cfg.model}': {source.path} ({source.parametrization})")

        if source.parametrization == "lddmm":
            self.faces = load_shape_model_triangulation(source.path)
            self._load_momentum_model()
            return

        from src.models.gpmm.model.model_memory_io import load_model_file

        loaded = load_model_file(
            source.path,
            normalize=source.normalize,
            atlas_options=source.atlas.to_options(),
            verbose=cfg.verbose,
        )
        self.faces = loaded.faces
        mean = loaded.mean
        eigenfunctions = loaded.eigenfunctions
        eigenvalues = loaded.eigenvalues

        eigenvalues = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
        n_modes = len(eigenvalues)
        if cfg.sliders.mode_count is not None:
            n_modes = min(int(cfg.sliders.mode_count), n_modes)

        self.mean = np.asarray(mean, dtype=np.float64).reshape(-1, 3)
        self.basis = np.ascontiguousarray(eigenfunctions[:, :n_modes], dtype=np.float64)
        self.sigma = np.sqrt(eigenvalues[:n_modes])
        self.alpha = np.zeros(n_modes)
        self.n_modes_total = len(eigenvalues)
        self.variance_share = eigenvalues[:n_modes] / eigenvalues.sum()
        self._set_variance_explained(
            eigenvalues, getattr(loaded, "noise", 0.0),
            getattr(loaded, "inner_product", "euclidean"),
        )

        # Per-vertex sd under the *whole* model, not just the exposed modes: it
        # is the natural scale for the colour ramp and the one mean_shape.py
        # writes, so the two pictures stay comparable.
        per_mode_sq = (eigenfunctions.reshape(len(self.mean), 3, -1) ** 2).sum(axis=1)
        self.total_std = np.sqrt(per_mode_sq @ eigenvalues)
        self._compute_mean_normals()
        self._compute_normal_std(eigenfunctions, eigenvalues)
        self._log_model()

    def _load_momentum_model(self):
        """The LDDMM parametrisation: tangent PCA now, a geodesic per sample."""
        from src.models.gpmm.model.sources.deformetrica import load_momentum_model

        cfg = self.config
        source = cfg.source()
        self.momentum_model = load_momentum_model(
            source.path, kernel_width=source.atlas.kernel_width,
            time_points=source.lddmm_time_points, verbose=cfg.verbose,
        )

        n_modes = self.momentum_model.n_modes
        if cfg.sliders.mode_count is not None:
            n_modes = min(int(cfg.sliders.mode_count), n_modes)

        self.n_modes_total = self.momentum_model.n_modes
        self.sigma = np.sqrt(self.momentum_model.eigenvalues[:n_modes])
        self.alpha = np.zeros(n_modes)
        self.variance_share = self.momentum_model.variance_share[:n_modes]

        # alpha = 0 is the mean *momentum* shot, which is not quite the
        # template, so it has to be integrated rather than read off the atlas.
        self._log("  shooting the mean momentum")
        self.mean = np.asarray(self.momentum_model.mean_shape(), dtype=np.float64)

        # The same population through the linear route, kept alongside so the
        # window can say what the linear model would miss on the current sample
        # - which is the whole question the two parametrisations differ on.
        linear_mean, eigenfunctions, eigenvalues, _ = load_shape_model(
            source.path, normalize=False,
            atlas_options=source.atlas.to_options(), verbose=False,
        )
        self.linear_mean = np.asarray(linear_mean, dtype=np.float64)
        self.linear_basis = np.ascontiguousarray(eigenfunctions, dtype=np.float64)

        per_mode_sq = (eigenfunctions.reshape(len(self.mean), 3, -1) ** 2).sum(axis=1)
        self.total_std = np.sqrt(per_mode_sq @ eigenvalues)
        # Tangent PCA on the momenta is not truncated by an approximation, so
        # there is no discarded-variance term to report.
        self.noise = 0.0
        self.variance_explained = float(self.momentum_model.variance_share.sum())
        self._compute_mean_normals()
        self._compute_normal_std(eigenfunctions, eigenvalues)
        self._log_model()

    def _set_variance_explained(self, eigenvalues, noise, inner_product: str):
        """What fraction of the model's covariance its modes actually carry.

        A GPMM built by the ssm_build stage is a *truncated*
        decomposition: the builder keeps modes until the approximation's
        tolerance or the truncation's cap stops it, and records the per-
        coordinate variance it gave up as ``noise``. The kept fraction is
        therefore ``sum(lambda) / (sum(lambda) + trace(noise))``, and the trace
        depends on which inner product the basis is orthonormal in:

        * ``mass``      ``Phi^T M Phi = I`` with the mesh quadrature, whose
                        weights sum to 1 by construction, so the trace over the
                        three coordinates of every vertex is ``3 * noise``;
        * ``euclidean`` the plain one, where it is ``3N * noise``.

        A PCA model has ``noise = 0`` and comes out at 100%, which is honest:
        it represents its own population exactly.
        """
        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        self.noise = float(noise or 0.0)
        weight = 3.0 if inner_product == "mass" else float(self.mean.size)
        discarded = self.noise * weight
        total = float(eigenvalues.sum()) + discarded
        self.variance_explained = float(eigenvalues.sum() / total) if total > 0 else 1.0

    def _log_model(self):
        self._log(f"  {len(self.mean)} vertices, {self.n_modes_total} modes "
                  f"({len(self.alpha)} on the panel, {self.variance_share.sum():.1%} "
                  f"of the variance), {0 if self.faces is None else len(self.faces)} triangles")
        self._log(f"  the modes carry {self.variance_explained:.2%} of the model's "
                  f"covariance ({1.0 - self.variance_explained:.2%} left as noise, "
                  f"{self.noise:.4g} per coordinate)")
        self._log(f"  mode 0 carries {self.variance_share[0]:.1%}, "
                  f"the last exposed mode {self.variance_share[-1]:.2%}")
        self._log(f"  per-vertex sd {self.total_std.min():.4g} .. {self.total_std.max():.4g} "
                  f"in model units")

    # -- state -------------------------------------------------------------

    @property
    def is_lddmm(self) -> bool:
        return self.config.source().parametrization == "lddmm"

    def sample(self) -> np.ndarray:
        """(N, 3) the shape the current coefficients stand for.

        Linear: one matrix product. LDDMM: a geodesic, integrated from the
        template with the momentum those coefficients build.
        """
        if self.is_lddmm:
            return self.momentum_model.reconstruct(self.alpha)
        return self.mean + (self.basis @ self.alpha).reshape(-1, 3)

    def update_shape(self):
        """Rebuilds the sample from the current coefficients, in place."""
        displacement = self.sample() - self.mean
        np.add(self.mean, displacement, out=self.vertices, casting="unsafe")
        self._points.Modified()

        magnitude = np.linalg.norm(displacement, axis=1)
        field = self._scalar_field(magnitude, displacement)
        if field is not None:
            self.scalars[:] = field
            self._scalar_array.Modified()

        self.polydata.Modified()
        self.update_status(magnitude)

    def _linear_residual(self) -> float:
        """Max distance from the current LDDMM sample to the linear model.

        The sample is projected onto the linear basis and the leftover measured,
        so this is the best the linear parametrisation could do on this shape -
        i.e. exactly what is given up by working in the ambient space.
        """
        residual = self.vertices.reshape(-1).astype(np.float64) - self.linear_mean
        recon = self.linear_basis @ (self.linear_basis.T @ residual)
        return float(np.linalg.norm((residual - recon).reshape(-1, 3), axis=1).max())

    def update_status(self, magnitude: np.ndarray):
        active = int(np.count_nonzero(self.alpha))
        text = (f"Mahalanobis {self._mahalanobis():5.2f} sd       "
                f"max displacement {float(magnitude.max()):.4g} model units       "
                f"{active} mode{'' if active == 1 else 's'} off the mean")
        if self.is_lddmm:
            text += f"       linear basis would miss it by {self._linear_residual():.4g}"
        self.status.SetInput(text)

    def _mahalanobis(self) -> float:
        """sqrt(sum (alpha_k / sigma_k)^2): how far the sample sits from the mean.

        The same quantity the mahalanobis prior and the chi-square clamp in the
        fitting code work with, so a shape that looks implausible here is one
        those would pull back.
        """
        return float(np.linalg.norm(self.alpha / self.sigma))

    # -- export ------------------------------------------------------------

    def export_vtp(self, path=None) -> Path:
        """Writes the shape currently on screen as a .vtp.

        The sample's vertices on the model's own triangulation, carrying the
        fields the window can colour by - so the file opens in ParaView showing
        what the window showed, including the signed displacement, rather than
        as bare geometry you would have to re-derive anything from.

        The coefficients travel with it as field data. That is what separates a
        saved sample from a screenshot: ``alpha``/``alpha_sigma`` name the point
        in shape space exactly, so the shape can be rebuilt - or fed to a
        fitting run as an initialisation - long after the window is closed.
        """
        cfg = self.config.export
        if path is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            safe = self.config.model.replace("/", "_").replace("\\", "_")
            path = Path(cfg.dir) / f"{safe}_{stamp}.vtp"
        path = Path(path)

        vertices = np.asarray(self.vertices, dtype=np.float64)
        polydata = create_polydata(vertices, self.faces)

        if cfg.include_fields:
            displacement = vertices - self.mean
            arrays = {
                "displacement": np.linalg.norm(displacement, axis=1),
                "signed_displacement": self.signed_displacement(displacement),
                "total_std": self.total_std,
            }
            for name, values in arrays.items():
                self._attach_point_array(polydata, name, values)
            # The vector itself, so the export can be warped or glyphed rather
            # than only shaded.
            self._attach_point_array(polydata, "displacement_vector", displacement)
            self._attach_field_array(polydata, "alpha", self.alpha)
            self._attach_field_array(polydata, "alpha_sigma", self.alpha / self.sigma)
            self._attach_field_array(polydata, "mahalanobis", [self._mahalanobis()])

        from src.shared.vtk.io import save_vtp

        save_vtp(polydata, str(path), binary=cfg.binary)
        self.written_files.append(path)
        self._log(f"wrote {path}")
        return path

    @staticmethod
    def _attach_point_array(polydata, name: str, values):
        values = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
        array = numpy_support.numpy_to_vtk(values, deep=True)
        array.SetName(name)
        polydata.GetPointData().AddArray(array)

    @staticmethod
    def _attach_field_array(polydata, name: str, values):
        values = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1))
        array = numpy_support.numpy_to_vtk(values, deep=True)
        array.SetName(name)
        polydata.GetFieldData().AddArray(array)

    def _on_save(self, widget, event):
        try:
            path = self.export_vtp()
        except Exception as exc:                       # noqa: BLE001 - reported, not raised
            self.status.SetInput(f"could not write the .vtp: {exc}")
        else:
            self.status.SetInput(f"wrote {path}")
        widget.GetRepresentation().SetState(0)
        self.render_window.Render()

    def reset_to_mean(self):
        """Back to alpha = 0 - every mode, not only the rows on screen."""
        self.alpha[:] = 0.0
        for widget in self.sliders:
            widget.GetRepresentation().SetValue(0.0)
        self.update_shape()

    # -- output ------------------------------------------------------------

    def save_screenshot(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.render_window.Render()
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(self.render_window)
        window_to_image.SetInputBufferTypeToRGB()
        window_to_image.ReadFrontBufferOff()
        window_to_image.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(window_to_image.GetOutputPort())
        writer.Write()

        self.written_files.append(path)
        self._log(f"screenshot written to {path}")

    def _log(self, *args):
        if self.config.verbose:
            print(*args)
