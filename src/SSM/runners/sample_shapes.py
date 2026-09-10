"""Interactive explorer for the shape space: one slider per mode.

The mean shape and the heatmaps from ``mean_shape.py`` describe the model in
the abstract - this walks it. Each slider sets one coefficient, so what the
window shows at any moment is

.. math:: x(\\alpha) = \\bar{x} + \\sum_k \\alpha_k \\phi_k

with the slider reading in *standard deviations* rather than raw coefficients:
:math:`\\alpha_k = s_k \\sqrt{\\lambda_k}`. That is the only scale on which the
modes are comparable - the raw coefficients differ by orders of magnitude
between the leading and the trailing modes, and by another few between the two
models - so slider 0 at +2 and slider 5 at +2 mean the same thing, namely two
standard deviations of a population that is Gaussian by construction. It also
makes the plausibility of a sample readable straight off the sliders: the sum
of their squares is the Mahalanobis distance the priors in the fitting code
penalise.

A model has far more modes than fit on screen (183 on the fetus model, 163 on
the baby), so the panel scrolls: a fixed pool of slider widgets is re-bound to
a sliding window of modes, which keeps the widget count - and the cost of
scrolling - independent of how many modes there are.

Which model opens is a config key, but it is not the end of it: the window
carries a dropdown listing every model the config declares under ``models``
*and* every archive found under ``library.root``, and picking one reloads the
window in place. That is what makes a folder of built GPMMs explorable - the
twenty-six kernels under ``configs/ssm/build_configs`` are a click apart
instead of an edit and a restart - and it is why nothing below is specific to
any one model: they all arrive through
:func:`src.SSM.model.model_memory_io.load_model_file` as the same
mean/eigenfunctions/eigenvalues triple.

The info line under the dropdown reports the two numbers that decide whether a
model is worth looking at: how many modes it has, and how much of its own
covariance those modes carry. The second is below 100% for every GPMM built by
``build_model.py``, because the decomposition is truncated and the remainder is
booked as ``noise`` - see :meth:`SampleShapesRunner._set_variance_explained`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import vtk
from vtk.util import numpy_support

from src.config.SSM.schema import SampleShapesConfig
from src.vtk.create import create_polydata
from src.SSM.model.load import load_shape_model, load_shape_model_triangulation


def _rgb(hex_color: str):
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


# One blue ramp, ordered near-zero -> maximum. It is reversed between the
# themes on purpose: a sequential encoding wants its low end to recede towards
# the surface, which is the lightest step on white and the darkest on black.
_BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

# Signed displacement needs a *diverging* ramp instead: the quantity has a
# meaningful zero (on the mean surface) and a sign, and a sequential ramp would
# hide both by mapping "denting in" and "on the mean" to adjacent colours.
# Red out / blue in, through an unsaturated middle that reads as "no change" in
# either theme - the neutral step is the only one that changes between them.
_DIVERGING_LIGHT = ("#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7",
                    "#fddbc7", "#ef8a62", "#b2182b")
# On black the neutral step has to be dark rather than white, or every
# unchanged vertex glows brighter than the deformed ones.
_DIVERGING_DARK = ("#4393c3", "#92c5de", "#5a6b7a", "#3a3a38",
                   "#8a5348", "#d6604d", "#f4a582")

_THEMES = {
    "light": {
        "background": (0.988, 0.988, 0.984),
        "text": (0.043, 0.043, 0.043),
        "muted": (0.322, 0.318, 0.306),
        "rail": (0.847, 0.843, 0.827),          # slider tube and end caps
        "track": (0.902, 0.898, 0.882),         # scrollbar trough
        "knob": _rgb("#2a78d6"),
        "knob_active": _rgb("#1c5cab"),
        "thumb": _rgb("#3987e5"),
        "button": (42, 120, 214),               # 0..255: it is drawn into an image
        "button_active": (28, 92, 171),
        "button_text": (1.0, 1.0, 1.0),
        "surface": (0.78, 0.76, 0.72),          # the shape when color_by: none
        "ghost": (0.45, 0.45, 0.43),
        "ramp": _BLUE,
        "diverging": _DIVERGING_LIGHT,
    },
    "dark": {
        "background": (0.102, 0.102, 0.098),
        "text": (0.92, 0.92, 0.90),
        "muted": (0.66, 0.65, 0.61),
        "rail": (0.26, 0.26, 0.25),
        "track": (0.20, 0.20, 0.19),
        "knob": _rgb("#6da7ec"),
        "knob_active": _rgb("#cde2fb"),
        "thumb": _rgb("#3987e5"),
        "button": (52, 80, 128),
        "button_active": (90, 130, 190),
        "button_text": (1.0, 1.0, 1.0),
        "surface": (0.82, 0.80, 0.76),
        "ghost": (0.55, 0.55, 0.52),
        "ramp": tuple(reversed(_BLUE)),
        "diverging": _DIVERGING_DARK,
    },
}

# Panel geometry, in normalised display coordinates: the slider column and the
# scrollbar beside it. _PANEL_Y is the centre of the top and bottom rows.
_PANEL_X = (0.030, 0.205)
_PANEL_Y = (0.880, 0.235)   # bottom clears the three-button stack below it
_SCROLLBAR_X = (0.216, 0.224)
# Where everything right of the panel starts.
_CONTENT_X = 0.26

# The model dropdown, in pixels down from the top of the content area.
_DROPDOWN_TOP = 30          # gap above the closed bar
_DROPDOWN_BAR_H = 28        # height of the closed bar
_DROPDOWN_PAD = 8           # text inset inside the bar and the rows

#: The one point-data array the mapper colours by. Its *contents* change with
#: the colour mode; its name must not, or the mapper would need re-pointing.
_COLOR_ARRAY = "color"

_COLOR_BAR_TITLES = {
    "displacement": "displacement",
    "signed_displacement": "signed displacement\n(+ out / - in)",
    "total_std": "per-vertex sd",
}
_COLOR_LABELS = {
    "displacement": "COLOR: DISPLACEMENT",
    "signed_displacement": "COLOR: SIGNED",
    "total_std": "COLOR: PER-VERTEX SD",
    "none": "COLOR: OFF",
}

#: Keys the window acts on. Consumed rather than passed through, because
#: vtkInteractorStyleTrackballCamera binds most single letters for itself.
_OUR_KEYS = frozenset({"c", "s", "m", "r", "t", "escape",
                       "bracketleft", "bracketright",
                       "prior", "next", "home", "end"})

_AXIS_VECTORS = {
    "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
}


@dataclass
class SampleShapesResult:
    """The state the window was left in."""

    alpha: np.ndarray                                  # (K,) coefficients, model units
    sigmas: np.ndarray                                 # (K,) the same, in sigma
    vertices: np.ndarray                               # (N, 3) the shape on screen
    mahalanobis: float
    written_files: List[Path] = field(default_factory=list)


class SampleShapesRunner:
    """Builds the window, wires the sliders to the model, and runs it."""

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

        from src.SSM.model.model_memory_io import load_model_file

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
        from src.SSM.model.deformetrica import load_momentum_model

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

        A GPMM built by ``scripts/SSM/build_model.py`` is a *truncated*
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

    # -- the scene ---------------------------------------------------------

    def _build_surface(self):
        """The polydata and the two numpy buffers VTK draws straight out of.

        Split out of :meth:`build_scene` because switching model replaces all
        three - a different model has a different vertex count and its own
        triangulation - while the mapper, actor and renderer around them stay.
        """
        view = self.config.view

        # The point buffer is handed to VTK without a copy and then written in
        # place, so a slider drag costs one matrix product and a Modified() -
        # no reallocation, no rebuilding of the mesh.
        self.vertices = np.array(self.mean, dtype=np.float32, order="C")
        self.polydata = create_polydata(self.vertices, self.faces)
        self._points = numpy_support.numpy_to_vtk(self.vertices, deep=False)
        self.polydata.GetPoints().SetData(self._points)

        # One array, always named the same, whatever is being shown in it. The
        # name is what the mapper selects by, so a mode change that renamed it
        # would have to re-point the mapper; rewriting the contents does not.
        self.scalars = np.zeros(len(self.mean), dtype=np.float32)
        self._scalar_array = numpy_support.numpy_to_vtk(self.scalars, deep=False)
        self._scalar_array.SetName(_COLOR_ARRAY)
        self.polydata.GetPointData().AddArray(self._scalar_array)

    def build_scene(self):
        view = self.config.view
        self._build_surface()
        self._resolve_color_range()

        mapper = vtk.vtkPolyDataMapper()
        if self.faces is not None:
            # Recomputed on every point change, so the lighting follows the
            # shape instead of staying frozen on the mean.
            self.normals = vtk.vtkPolyDataNormals()
            self.normals.SetInputData(self.polydata)
            self.normals.ComputePointNormalsOn()
            self.normals.ComputeCellNormalsOff()
            self.normals.SplittingOff()
            self.normals.ConsistencyOn()
            mapper.SetInputConnection(self.normals.GetOutputPort())
        else:
            self.normals = None
            mapper.SetInputData(self.polydata)

        # The lookup table exists in every mode, including 'none' - the mode is
        # a property of the window now, not of the build, so 'c' can turn the
        # colours back on without any of this having to be rebuilt.
        self.lookup_table = vtk.vtkColorTransferFunction()
        mapper.SetLookupTable(self.lookup_table)
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(_COLOR_ARRAY)

        self.mapper = mapper
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().SetSpecular(0.2)
        self.actor.GetProperty().SetSpecularPower(25)
        if view.show_edges:
            self.actor.GetProperty().EdgeVisibilityOn()
            self.actor.GetProperty().SetLineWidth(0.5)
        if self.faces is None:
            self.actor.GetProperty().SetPointSize(2)

        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.actor)
        if view.show_mean:
            self.ghost = self._mean_ghost()
            self.renderer.AddActor(self.ghost)
        # Built whatever the opening mode is, so that turning the colours back
        # on with 'c' does not have to construct one mid-session.
        if view.show_scalar_bar:
            self.scalar_bar = self._scalar_bar()
            self.renderer.AddActor2D(self.scalar_bar)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetSize(*view.window_size)
        self.render_window.SetWindowName(f"shape space - {self.config.model}")
        self.render_window.AddRenderer(self.renderer)
        if not self.config.interactive:
            self.render_window.SetOffScreenRendering(1)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self._place_camera()

    def _resolve_color_range(self):
        """The two fixed ranges: one for the magnitudes, one for the signed field.

        Both are derived from the *model* rather than from the current sample -
        a range that followed the sample would repaint the surface on every drag
        and mean nothing - and both are held for the session.

        They are not the same number. Signed displacement is only the normal
        component, so it never reaches the magnitude's range: sharing one would
        leave the diverging ramp permanently washed out, using perhaps half its
        width. ``normal_std`` is the exact analogue of ``total_std`` projected
        onto the normal, so ``sigma_range * max(normal_std)`` bounds the signed
        field the same way ``sigma_range * max(total_std)`` bounds the magnitude.
        """
        configured = self.config.view.color_range
        self.color_range = configured
        if self.color_range is None:
            self.color_range = float(self.config.sliders.sigma_range * self.total_std.max())
        if self.color_range <= 0:
            raise ValueError(f"view.color_range must be positive, got {self.color_range}")

        if configured is not None:
            self.signed_range = float(configured)
        elif self.normal_std is None:
            self.signed_range = self.color_range
        else:
            self.signed_range = float(
                self.config.sliders.sigma_range * self.normal_std.max()
            ) or self.color_range

    @property
    def active_range(self) -> float:
        """The half-width the active mode's ramp spans."""
        return self.signed_range if self.is_signed else self.color_range

    @property
    def is_signed(self) -> bool:
        """Whether the active mode is drawn on a diverging ramp about zero."""
        return self.color_mode == "signed_displacement"

    def _scalar_bar(self) -> vtk.vtkScalarBarActor:
        bar = vtk.vtkScalarBarActor()
        bar.SetLookupTable(self.lookup_table)
        bar.SetNumberOfLabels(5)
        bar.SetMaximumWidthInPixels(60)
        bar.SetMaximumHeightInPixels(240)
        bar.GetPositionCoordinate().SetValue(0.915, 0.10)
        bar.SetWidth(0.07)
        bar.SetHeight(0.32)
        # Without this the bar sizes its own text to the box and ignores the
        # font sizes set on the text properties.
        bar.UnconstrainedFontSizeOn()
        bar.GetTitleTextProperty().SetFontSize(13)
        bar.GetLabelTextProperty().SetFontSize(12)
        for prop in (bar.GetTitleTextProperty(), bar.GetLabelTextProperty()):
            prop.SetBold(False)
            prop.ShadowOff()
            prop.ItalicOff()
        return bar

    # -- colour mode -------------------------------------------------------

    def _scalar_field(self, magnitude, displacement) -> Optional[np.ndarray]:
        """The (N,) field the active mode paints, or None when it paints nothing."""
        if self.color_mode == "displacement":
            return magnitude
        if self.color_mode == "signed_displacement":
            return self.signed_displacement(displacement)
        if self.color_mode == "total_std":
            return self.total_std
        return None

    def signed_displacement(self, displacement: np.ndarray) -> np.ndarray:
        """``(x - mean) . n``, the outward normal component of the deformation.

        Is a signed displacement sensible? For a surface, yes, and it is the
        more informative of the two: ``|x - mean|`` cannot tell a bulge from a
        dent, so a nose growing forward and a nose collapsing backward paint
        identically. Projecting onto the normal separates them, which is what
        makes a mode readable as anatomy rather than as "something moved here".

        Two things it deliberately is not:

        * It is **not** the whole displacement. A vertex that slides
          tangentially along the surface reads zero, so ``|signed| <= magnitude``
          always. Tangential motion is largely a property of how the
          correspondence was established rather than of the shape, which is the
          other reason to project it out.
        * The normals are the **mean shape's**, computed once per model, not the
          current sample's. The mean surface is the reference the sign is taken
          against; re-deriving normals from the moving sample would make the
          zero level shift underneath the colours as the sliders move.
        """
        if self.mean_normals is None:
            # No triangulation, so no orientable surface and no meaningful sign.
            return np.linalg.norm(displacement, axis=1)
        return np.einsum("ij,ij->i", displacement, self.mean_normals)

    def _compute_mean_normals(self):
        """Outward vertex normals of the mean shape, for the signed mode."""
        if self.faces is None:
            self.mean_normals = None
            return
        from src.geometry.geometry import mesh_vertex_normals

        self.mean_normals = np.ascontiguousarray(
            mesh_vertex_normals(np.asarray(self.mean, dtype=np.float64), self.faces),
            dtype=np.float64,
        )

    def _compute_normal_std(self, eigenfunctions, eigenvalues):
        """Per-vertex sd of ``(x - mean) . n`` under the whole model.

        ``total_std`` is the sd of the displacement's length; this is the sd of
        its normal component, obtained by projecting every eigenfunction onto
        the mean normal before accumulating. It exists so the signed mode has a
        scale of its own - see :meth:`_resolve_color_range`.
        """
        if self.mean_normals is None:
            self.normal_std = None
            return
        n_vertices = len(self.mean)
        modes = np.asarray(eigenfunctions, dtype=np.float64).reshape(n_vertices, 3, -1)
        # (N, K): each mode's displacement at each vertex, along that normal.
        projected = np.einsum("nik,ni->nk", modes, self.mean_normals)
        self.normal_std = np.sqrt(projected ** 2 @ np.asarray(eigenvalues, dtype=np.float64))

    def set_color_mode(self, mode: str, render: bool = True):
        """Switches what the surface is coloured by, at runtime.

        Everything the mode implies is re-derived here rather than at build
        time: which ramp (sequential, or diverging for the signed mode), what
        range it spans (``[0, r]`` or ``[-r, +r]``), whether the scalar bar is
        on the screen at all, and whether the mapper is showing scalars or
        letting the flat surface colour through.
        """
        if mode not in self.config.view.COLOR_MODES:
            raise ValueError(f"unknown colour mode {mode!r}; "
                             f"expected one of {list(self.config.view.COLOR_MODES)}")
        self.color_mode = mode

        if mode == "none":
            self.mapper.ScalarVisibilityOff()
        else:
            self.mapper.ScalarVisibilityOn()
            span = self.active_range
            self.mapper.SetScalarRange(-span if self.is_signed else 0.0, span)

        self._apply_ramp()
        self._place_scalar_bar()
        self.update_shape()
        if self.color_button_label is not None:
            self.color_button_label.SetInput(_COLOR_LABELS[mode])
        if render:
            self.render_window.Render()

    def cycle_color_mode(self):
        modes = list(self.config.view.COLOR_MODES)
        self.set_color_mode(modes[(modes.index(self.color_mode) + 1) % len(modes)])

    def _apply_ramp(self):
        """Rebuilds the lookup table for the active mode and theme."""
        self.lookup_table.RemoveAllPoints()
        if self.color_mode == "none":
            return
        if self.is_signed:
            # Spread symmetrically about zero so the neutral step lands exactly
            # on "on the mean surface" - the whole point of a diverging ramp.
            ramp = self.theme["diverging"]
            half = self.signed_range
            for i, hex_color in enumerate(ramp):
                self.lookup_table.AddRGBPoint(
                    -half + 2.0 * half * i / (len(ramp) - 1), *_rgb(hex_color)
                )
        else:
            ramp = self.theme["ramp"]
            for i, hex_color in enumerate(ramp):
                self.lookup_table.AddRGBPoint(
                    self.color_range * i / (len(ramp) - 1), *_rgb(hex_color)
                )

    def _place_scalar_bar(self):
        """Adds, removes or retitles the colour key to match the active mode."""
        if self.scalar_bar is None:
            return
        show = self.color_mode != "none" and self.config.view.show_scalar_bar
        self.scalar_bar.SetVisibility(bool(show))
        if show:
            self.scalar_bar.SetTitle(f"{_COLOR_BAR_TITLES[self.color_mode]}\n(model units)")

    def _mean_ghost(self) -> vtk.vtkActor:
        """The mean shape, translucent, held in place behind the sample."""
        polydata = create_polydata(np.asarray(self.mean, dtype=np.float32), self.faces)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(self.config.view.mean_opacity)
        actor.GetProperty().LightingOff()
        return actor

    def _place_camera(self):
        source = self.config.source()
        centre = self.mean.mean(axis=0)
        radius = float(np.linalg.norm(self.mean - centre, axis=1).max())

        axis = np.array(_AXIS_VECTORS[source.camera_axis])
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*centre)
        camera.SetPosition(*(centre + axis * radius * 3.0))
        camera.SetViewUp(*_AXIS_VECTORS[source.view_up])
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()

    # -- the widgets -------------------------------------------------------

    def build_widgets(self):
        self._build_sliders()
        self._build_scrollbar()
        self._build_reset_button()
        self._build_labels()
        self._build_dropdown()

        # The wheel belongs to the panel while the cursor is over it and to the
        # camera everywhere else, so scrolling the modes and zooming the shape
        # never fight over it.
        self._wheel_tags = {
            event: self.interactor.AddObserver(event, self._on_wheel, 10.0)
            for event in ("MouseWheelForwardEvent", "MouseWheelBackwardEvent")
        }
        # Ahead of the camera style (priority 0), so a click that lands on the
        # dropdown picks a model instead of starting a rotate.
        self._click_tag = self.interactor.AddObserver(
            "LeftButtonPressEvent", self._on_click, 20.0
        )
        # The same actions from the keyboard, since the sliders already own the
        # mouse and reaching for the button breaks the flow of exploring.
        self._key_tag = self.interactor.AddObserver("KeyPressEvent", self._on_key)
        self.interactor.AddObserver("ConfigureEvent", self._on_resize)
        self.interactor.Initialize()

        self.apply_theme(self.theme_name)
        self.set_color_mode(self.color_mode, render=False)
        self.scroll_to(0)

    def _build_sliders(self):
        """The scrolling panel: a fixed pool of rows, re-bound as it scrolls.

        Only the geometry is set up here. Which mode each row *is* comes from
        :meth:`scroll_to`, so the widget count stays at ``visible_rows``
        however many modes the model has.
        """
        # The pool is always the configured height, not this model's mode
        # count: switching to a model with more modes must be able to fill it,
        # and VTK slider widgets cannot be created after the interactor starts.
        # A model with fewer modes disables the surplus rows in scroll_to().
        pool = int(self.config.sliders.visible_rows)
        self.n_rows = min(pool, len(self.alpha))
        self.first_row = 0
        limit = float(self.config.sliders.sigma_range)

        top, bottom = _PANEL_Y
        spacing = (top - bottom) / max(pool - 1, 1) if pool > 1 else 0.0

        for row in range(pool):
            y = top - row * spacing
            representation = vtk.vtkSliderRepresentation2D()
            representation.SetMinimumValue(-limit)
            representation.SetMaximumValue(limit)
            representation.SetValue(0.0)
            representation.SetLabelFormat("%+.2f")

            for coordinate, x in ((representation.GetPoint1Coordinate(), _PANEL_X[0]),
                                  (representation.GetPoint2Coordinate(), _PANEL_X[1])):
                coordinate.SetCoordinateSystemToNormalizedDisplay()
                coordinate.SetValue(x, y)

            representation.SetSliderLength(0.016)
            representation.SetSliderWidth(0.020)
            representation.SetEndCapLength(0.003)
            representation.SetEndCapWidth(0.018)
            representation.SetTubeWidth(0.005)
            # Text size on this representation is a *height*, as a fraction of
            # the viewport - the font size on the text property is overridden
            # and has no effect.
            representation.SetTitleHeight(0.016)
            representation.SetLabelHeight(0.014)

            widget = vtk.vtkSliderWidget()
            widget.SetInteractor(self.interactor)
            widget.SetRepresentation(representation)
            widget.SetAnimationModeToJump()
            widget.EnabledOn()
            callback = self._on_slider(row)
            widget.AddObserver("InteractionEvent", callback)
            widget.AddObserver("EndInteractionEvent", callback)
            self.sliders.append(widget)

    def _build_scrollbar(self):
        """Track and thumb beside the panel: how far down the modes you are."""
        self.scroll_track, self._track_points = self._quad()
        self.scroll_thumb, self._thumb_points = self._quad()
        self.renderer.AddActor2D(self.scroll_track)
        self.renderer.AddActor2D(self.scroll_thumb)

        top, bottom = _PANEL_Y
        self._set_quad(self._track_points, _SCROLLBAR_X, (bottom - 0.03, top + 0.03))

    @staticmethod
    def _quad(display: bool = False):
        """A filled rectangle, in normalised viewport or in pixels.

        The slider panel is laid out in fractions of the window, so its quads
        rescale with it; the dropdown is laid out in pixels, because its rows
        have to keep a fixed height for the text to sit in them.
        """
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(4)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(4)
        for i in range(4):
            cells.InsertCellPoint(i)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)

        coordinate = vtk.vtkCoordinate()
        if display:
            coordinate.SetCoordinateSystemToDisplay()
        else:
            coordinate.SetCoordinateSystemToNormalizedViewport()
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(polydata)
        mapper.SetTransformCoordinate(coordinate)

        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        return actor, points

    @staticmethod
    def _set_quad(points, x_range, y_range):
        x0, x1 = x_range
        y0, y1 = y_range
        for i, (x, y) in enumerate(((x0, y0), (x1, y0), (x1, y1), (x0, y1))):
            points.SetPoint(i, x, y, 0.0)
        points.Modified()

    def _build_reset_button(self):
        """The three action buttons, stacked under the slider column."""
        self.buttons = []
        specs = (
            ("RESET TO MEAN", self._on_reset),
            (_COLOR_LABELS[self.color_mode], self._on_color),
            ("SAVE .VTP", self._on_save),
        )
        for text, callback in specs:
            widget = vtk.vtkButtonWidget()
            widget.SetInteractor(self.interactor)
            representation = vtk.vtkTexturedButtonRepresentation2D()
            representation.SetNumberOfStates(2)
            representation.SetPlaceFactor(1)
            widget.SetRepresentation(representation)
            widget.On()
            widget.AddObserver("StateChangedEvent", callback)

            # Added *after* the widget is enabled: 2D props draw in the order
            # they were added, and the button's own texture would otherwise
            # cover the label that names it.
            label = vtk.vtkTextActor()
            label.SetInput(text)
            prop = label.GetTextProperty()
            prop.SetFontSize(13)
            prop.SetJustificationToCentered()
            prop.SetVerticalJustificationToCentered()
            prop.BoldOn()
            prop.ShadowOff()
            self.renderer.AddActor2D(label)
            self.buttons.append((widget, label))

        # Named handles for the two whose text changes or is read elsewhere.
        self.button, self.button_label = self.buttons[0]
        self.color_button_label = self.buttons[1][1]
        self.save_button_label = self.buttons[2][1]
        self._place_button()

    @staticmethod
    def _button_image(color) -> vtk.vtkImageData:
        """A plain filled plate with a lighter border; the label rides on top."""
        width, height = 190, 38
        canvas = vtk.vtkImageCanvasSource2D()
        canvas.SetScalarTypeToUnsignedChar()
        canvas.SetNumberOfScalarComponents(3)
        canvas.SetExtent(0, width - 1, 0, height - 1, 0, 0)
        canvas.SetDrawColor(*color)
        canvas.FillBox(0, width - 1, 0, height - 1)
        canvas.SetDrawColor(*[min(255, c + 55) for c in color])
        canvas.DrawSegment(0, 0, width - 1, 0)
        canvas.DrawSegment(0, height - 1, width - 1, height - 1)
        canvas.DrawSegment(0, 0, 0, height - 1)
        canvas.DrawSegment(width - 1, 0, width - 1, height - 1)
        canvas.Update()

        image = vtk.vtkImageData()
        image.DeepCopy(canvas.GetOutput())
        return image

    def _place_button(self):
        """The button stack below the slider column, in display (pixel) coords.

        Bottom-up, so the first button keeps the position it always had and the
        two added since grow the stack upwards into the gap under the panel.
        """
        width, height = self.render_window.GetSize()
        x0 = _PANEL_X[0] * width
        x1 = x0 + 190
        for index, (widget, label) in enumerate(getattr(self, "buttons", [])):
            y0 = 0.035 * height + index * 44
            y1 = y0 + 38
            widget.GetRepresentation().PlaceWidget([x0, x1, y0, y1, 0.0, 0.0])
            label.SetPosition((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def _text_actor(self, size: int, bold: bool = False) -> vtk.vtkTextActor:
        actor = vtk.vtkTextActor()
        prop = actor.GetTextProperty()
        prop.SetFontSize(size)
        prop.SetBold(bold)
        prop.ShadowOff()
        prop.ItalicOff()
        self.renderer.AddActor2D(actor)
        return actor

    def _build_labels(self):
        """The model info line, the panel header, the status line, and the hint."""
        # What the current model is, under the dropdown that names it. The
        # dropdown bar carries the name, so this carries the numbers.
        self.info = self._text_actor(12)

        # Sits above the panel and says where in the mode list you are.
        self.panel_header = self._text_actor(12)

        self.status = self._text_actor(14)
        self.hint = self._text_actor(12)
        self.hint.SetInput(
            "drag a slider to move along that mode      "
            "scroll over the panel for more modes (PgUp / PgDn, Home / End)      "
            "'m' model list, '[' / ']' step it      'c' colour mode      "
            "'s' save .vtp      'r' reset      't' theme"
        )
        self._update_info()
        self._place_labels()

    def _update_info(self):
        """The one line that says how much model is on screen.

        Three numbers, because they answer three different questions: how many
        modes the file has, how many of them the panel exposes, and how much of
        the model's own covariance those modes actually carry - which is below
        100% for every GPMM here, since the builder truncates the decomposition
        and books the remainder as noise.
        """
        exposed = ("all on the panel" if len(self.alpha) == self.n_modes_total
                   else f"{len(self.alpha)} on the panel")
        self.info.SetInput(
            f"{self.n_modes_total} modes, {exposed}     "
            f"modes carry {self.variance_explained:.2%} of the model's variance     "
            f"{len(self.mean)} vertices"
        )

    # -- the model dropdown ------------------------------------------------

    def _build_dropdown(self):
        """A bar naming the current model, and the list it drops down.

        VTK has no combo box, so this is assembled from the same two primitives
        the rest of the window uses - a filled quad and a text actor - one pair
        for the bar and one per visible row. The rows are a fixed pool bound to
        a sliding window of the model list, exactly as the slider panel does
        with modes, so a library of hundreds costs the same as one of five.

        Everything is placed in *pixels* rather than normalised coordinates:
        the text has to sit centred in its row at any window size, and only one
        of the two is a pixel quantity to begin with.
        """
        self.dropdown_bar, self._bar_points = self._quad(display=True)
        self.renderer.AddActor2D(self.dropdown_bar)

        self.dropdown_label = self._text_actor(13, bold=True)
        self.dropdown_label.GetTextProperty().SetVerticalJustificationToCentered()

        rows = min(self.config.dropdown.visible_rows, len(self.model_names))
        for _ in range(rows):
            actor, points = self._quad(display=True)
            actor.VisibilityOff()
            self.renderer.AddActor2D(actor)
            text = self._text_actor(12)
            text.GetTextProperty().SetVerticalJustificationToCentered()
            text.VisibilityOff()
            self.dropdown_rows.append((actor, points, text))

        self._place_dropdown()

    def _place_dropdown(self):
        """Lays the bar out, and the list under it when open."""
        width, height = self.render_window.GetSize()
        cfg = self.config.dropdown
        x0 = int(_CONTENT_X * width)
        x1 = x0 + cfg.width
        top = height - _DROPDOWN_TOP
        bottom = top - _DROPDOWN_BAR_H

        self._bar_rect = (x0, x1, bottom, top)
        self._set_quad(self._bar_points, (x0, x1), (bottom, top))
        # ASCII on purpose: VTK renders text through whatever font it finds,
        # and the geometric-shape glyphs are missing from several of them.
        caret = "v" if not self.dropdown_open else "^"
        label = f"{caret}  {self.config.model}"
        if self.dropdown_open:
            # Where you are in the list, since only a window of it is drawn and
            # nothing else says how much is above or below.
            here = self.model_names.index(self.config.model) + 1
            label += f"      {here} of {len(self.model_names)}"
        self.dropdown_label.SetInput(label)
        self.dropdown_label.SetDisplayPosition(x0 + _DROPDOWN_PAD, (bottom + top) // 2)

        self._dropdown_rects = []
        rows = len(self.dropdown_rows)
        self.dropdown_first = int(np.clip(
            self.dropdown_first, 0, max(0, len(self.model_names) - rows)
        ))
        for index, (actor, points, text) in enumerate(self.dropdown_rows):
            if not self.dropdown_open:
                actor.VisibilityOff()
                text.VisibilityOff()
                continue
            name_index = self.dropdown_first + index
            row_top = bottom - index * cfg.row_height
            row_bottom = row_top - cfg.row_height
            name = self.model_names[name_index]

            self._set_quad(points, (x0, x1), (row_bottom, row_top))
            actor.VisibilityOn()
            marker = "> " if name == self.config.model else "  "
            text.SetInput(f"{marker}{name}")
            text.SetDisplayPosition(x0 + _DROPDOWN_PAD, (row_bottom + row_top) // 2)
            text.VisibilityOn()
            self._dropdown_rects.append((x0, x1, row_bottom, row_top, name))

        self._theme_dropdown()

    def _theme_dropdown(self):
        """Colours the bar and the rows, marking the current model."""
        theme = self.theme
        self.dropdown_bar.GetProperty().SetColor(*theme["rail"])
        self.dropdown_label.GetTextProperty().SetColor(*theme["text"])
        for index, (actor, _, text) in enumerate(self.dropdown_rows):
            if index >= len(self._dropdown_rects):
                continue
            name = self._dropdown_rects[index][4]
            current = name == self.config.model
            # The current row is picked out by its plate, not by dimming the
            # others: every name in the list has to stay readable.
            actor.GetProperty().SetColor(*(theme["rail"] if current else theme["track"]))
            text.GetTextProperty().SetColor(*theme["text"])
            text.GetTextProperty().SetBold(current)

    def toggle_dropdown(self, open_it: Optional[bool] = None):
        self.dropdown_open = (not self.dropdown_open) if open_it is None else bool(open_it)
        if self.dropdown_open:
            # Open onto the current model rather than back at the top of the
            # list, which is what makes stepping through neighbours workable.
            rows = len(self.dropdown_rows)
            here = self.model_names.index(self.config.model)
            self.dropdown_first = int(np.clip(here - rows // 2, 0,
                                              max(0, len(self.model_names) - rows)))
        self._place_dropdown()
        self.render_window.Render()

    def _dropdown_hit(self, x: int, y: int) -> Optional[str]:
        """The model name under the cursor, or None if the click missed."""
        for x0, x1, y0, y1, name in self._dropdown_rects:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return name
        return None

    def _over_dropdown(self, x: int, y: int) -> bool:
        x0, x1, y0, y1 = self._bar_rect
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
        return self.dropdown_open and self._dropdown_hit(x, y) is not None

    def _on_click(self, interactor, event):
        x, y = interactor.GetEventPosition()
        if not (self.dropdown_open or self._over_dropdown(x, y)):
            return                      # nothing to do with us; let the camera have it

        bar_x0, bar_x1, bar_y0, bar_y1 = self._bar_rect
        if bar_x0 <= x <= bar_x1 and bar_y0 <= y <= bar_y1:
            self.toggle_dropdown()
        else:
            # An open list swallows the next click wherever it lands: on a row
            # it picks that model, anywhere else it just closes. Either way the
            # click must not also reach the camera, or the shape would lurch as
            # the list disappears.
            name = self._dropdown_hit(x, y)
            self.toggle_dropdown(False)
            if name is not None and name != self.config.model:
                self.switch_model(name)
        interactor.GetCommand(self._click_tag).SetAbortFlag(1)

    def scroll_dropdown(self, rows: int):
        self.dropdown_first += rows
        self._place_dropdown()
        self.render_window.Render()

    def switch_model(self, name: str):
        """Loads another model into the window that is already open.

        Everything downstream of the arrays is rebuilt, because none of it
        survives the change: a different model has its own vertex count and
        triangulation (so the polydata and both numpy buffers go), its own
        number of modes (so the slider pool is re-bound and re-sized), its own
        units (so the colour range and the camera distance are re-derived).
        What stays is the mapper, the actor and the renderer - the scene
        graph - which is why this is a reload rather than a new window.

        The slider positions carry over by default - see
        :meth:`_carry_coefficients`, and ``sliders.carry_across_models`` to turn
        it off - so the same point in shape space can be held still while the
        model under it changes.

        A model that fails to load leaves the window on the one it had, with
        the reason on the status line: a bad entry in a scanned folder should
        not take the session down.
        """
        previous = self.config.model
        # In sigma, because that is the only reading two models share. Taken
        # before load_model() replaces alpha and sigma.
        carried = (self.alpha / self.sigma) if self.config.sliders.carry_across_models else None

        self.config.model = name
        started = time.time()
        try:
            self.load_model()
        except Exception as exc:                       # noqa: BLE001 - reported, not raised
            self.config.model = previous
            self.status.SetInput(f"could not load '{name}': {exc}")
            self._place_dropdown()
            self.render_window.Render()
            return

        # An LDDMM model pays for every sample, a linear one does not.
        self._min_interval = 0.15 if self.config.source().parametrization == "lddmm" else 0.0

        self._build_surface()
        if self.normals is not None:
            self.normals.SetInputData(self.polydata)
        else:
            self.mapper.SetInputData(self.polydata)

        # The new model has its own units, so the range is re-derived; the
        # colour *mode* is the user's choice and survives the switch.
        self._resolve_color_range()

        if self.ghost is not None:
            self.renderer.RemoveActor(self.ghost)
            self.ghost = self._mean_ghost()
            self.renderer.AddActor(self.ghost)

        # Before _rebind_sliders, which reads alpha to position the rows.
        carried_count = self._carry_coefficients(carried)
        self._rebind_sliders()
        self._place_camera()
        self.render_window.SetWindowName(f"shape space - {name}")

        # Re-themed rather than tinted: the colour ramp is rebuilt against the
        # new colour range, which the theme owns.
        self.apply_theme(self.theme_name)
        self.set_color_mode(self.color_mode, render=False)
        self._update_info()
        self._place_labels()
        self._place_dropdown()

        if carried_count is None:
            self.reset_to_mean()
        else:
            self.update_shape()
        self.renderer.ResetCameraClippingRange()

        note = ""
        if carried_count is not None:
            kept, dropped = carried_count
            if kept:
                note = f", kept {kept} slider{'' if kept == 1 else 's'}"
            if dropped:
                # The new model is shorter, so those modes have nowhere to go.
                note += (f", dropped {dropped} past its {self.n_modes_total} modes")
                self.status.SetInput(
                    f"{self.status.GetInput()}       "
                    f"{dropped} carried mode{'' if dropped == 1 else 's'} dropped: "
                    f"'{name}' has only {self.n_modes_total}"
                )
        self.render_window.Render()
        self._log(f"  switched in {time.time() - started:.1f} s{note}")

    def _carry_coefficients(self, carried) -> Optional[int]:
        """Re-applies the previous model's slider readings to the new model.

        ``carried`` is in standard deviations, which is what makes this
        meaningful at all: ``alpha_k = s_k sqrt(lambda_k)``, and lambda differs
        by orders of magnitude between models - a factor of ~3.5e7 between the
        two domains here - so carrying the raw coefficient would produce a
        shape thousands of times off the mean, or indistinguishable from it.
        Carrying ``s`` instead holds the *position* in each model's own
        distribution, which also preserves the Mahalanobis distance over the
        modes both models have.

        Returns ``(kept, dropped)`` counting only sliders that were actually off
        the mean - carrying a zero is a no-op and not worth reporting - or None
        when the caller should fall back to resetting.
        """
        if carried is None:
            return None
        shared = min(len(carried), len(self.alpha))
        self.alpha[:] = 0.0
        self.alpha[:shared] = carried[:shared] * self.sigma[:shared]
        return (int(np.count_nonzero(carried[:shared])),
                int(np.count_nonzero(carried[shared:])))

    def _rebind_sliders(self):
        """Re-sizes the row pool to the new model's mode count.

        The pool was built for the first model. A model with fewer modes than
        there are rows leaves the surplus rows disabled rather than showing
        sliders that move nothing.
        """
        self.n_rows = min(int(self.config.sliders.visible_rows), len(self.alpha))
        for row, widget in enumerate(self.sliders):
            if row < self.n_rows:
                widget.EnabledOn()
            else:
                widget.EnabledOff()
        self.first_row = 0
        self.scroll_to(0)

    # -- theme -------------------------------------------------------------

    def apply_theme(self, name: str):
        """Repaints every element of the window in the named theme."""
        self.theme_name = name
        self.theme = _THEMES[name]
        theme = self.theme

        self.renderer.SetBackground(*theme["background"])
        self.actor.GetProperty().SetColor(*theme["surface"])
        if self.ghost is not None:
            self.ghost.GetProperty().SetColor(*theme["ghost"])

        # Both ramps run the other way per theme, so the near-zero end always
        # recedes towards the surface rather than shouting from it.
        self._apply_ramp()

        for widget in self.sliders:
            representation = widget.GetRepresentation()
            representation.GetSliderProperty().SetColor(*theme["knob"])
            representation.GetSelectedProperty().SetColor(*theme["knob_active"])
            representation.GetTubeProperty().SetColor(*theme["rail"])
            representation.GetCapProperty().SetColor(*theme["rail"])
            for prop in (representation.GetTitleProperty(), representation.GetLabelProperty()):
                prop.SetColor(*theme["muted"])

        self.scroll_track.GetProperty().SetColor(*theme["track"])
        self.scroll_thumb.GetProperty().SetColor(*theme["thumb"])

        idle = self._button_image(theme["button"])
        active = self._button_image(theme["button_active"])
        for widget, label in self.buttons:
            representation = widget.GetRepresentation()
            representation.SetButtonTexture(0, idle)
            representation.SetButtonTexture(1, active)
            label.GetTextProperty().SetColor(*theme["button_text"])

        self.status.GetTextProperty().SetColor(*theme["text"])
        for actor in (self.hint, self.panel_header, self.info):
            actor.GetTextProperty().SetColor(*theme["muted"])
        if self.dropdown_bar is not None:
            self._theme_dropdown()
        if self.scalar_bar is not None:
            self.scalar_bar.GetTitleTextProperty().SetColor(*theme["muted"])
            self.scalar_bar.GetLabelTextProperty().SetColor(*theme["muted"])

    def toggle_theme(self):
        self.apply_theme("dark" if self.theme_name == "light" else "light")
        self.render_window.Render()

    # -- scrolling ---------------------------------------------------------

    def scroll_to(self, first_row: int):
        """Binds the row pool to the modes starting at ``first_row``."""
        last_start = max(0, len(self.alpha) - self.n_rows)
        self.first_row = int(np.clip(first_row, 0, last_start))

        for row, widget in enumerate(self.sliders):
            if row >= self.n_rows:
                widget.EnabledOff()     # this model has fewer modes than rows
                continue
            widget.EnabledOn()
            mode = self.first_row + row
            representation = widget.GetRepresentation()
            representation.SetTitleText(f"mode {mode}    {self.variance_share[mode]:.2%} var")
            # The row shows whatever this mode was left at, so scrolling away
            # and back is lossless.
            representation.SetValue(float(self.alpha[mode] / self.sigma[mode]))

        # Thumb: as tall a fraction of the track as the window is of the modes.
        top, bottom = _PANEL_Y
        track_low, track_high = bottom - 0.03, top + 0.03
        span = track_high - track_low
        height = max(span * self.n_rows / float(len(self.alpha)), 0.02)
        offset = 0.0 if last_start == 0 else self.first_row / float(last_start)
        thumb_top = track_high - offset * (span - height)
        self._set_quad(self._thumb_points, _SCROLLBAR_X, (thumb_top - height, thumb_top))

        self.panel_header.SetInput(
            f"modes {self.first_row}-{self.first_row + self.n_rows - 1} of {len(self.alpha)}"
        )

    def scroll_by(self, rows: int):
        self.scroll_to(self.first_row + rows)
        self.render_window.Render()

    def _over_panel(self) -> bool:
        x, _ = self.interactor.GetEventPosition()
        return x <= (_SCROLLBAR_X[1] + 0.01) * self.render_window.GetSize()[0]

    def _on_wheel(self, interactor, event):
        step = -1 if event == "MouseWheelForwardEvent" else 1
        if self.dropdown_open:
            self.scroll_dropdown(step)
        elif self._over_panel():
            self.scroll_by(step)
        else:
            return                      # let the camera style zoom as usual
        interactor.GetCommand(self._wheel_tags[event]).SetAbortFlag(1)

    def _on_slider(self, row: int):
        """Callback for panel row ``row``; the value read off it is in sigma.

        A linear sample is a matrix product and can be redrawn on every event.
        An LDDMM sample is an integration - a few tenths of a second - so the
        drag is throttled and the release always gets the last word, which
        keeps the shape following the slider without queueing up shoots.
        """
        def callback(widget, event):
            mode = self.first_row + row
            self.alpha[mode] = widget.GetRepresentation().GetValue() * self.sigma[mode]

            if event == "InteractionEvent" and self._min_interval > 0.0:
                if time.time() - self._last_update < self._min_interval:
                    return
            self.update_shape()
            self._last_update = time.time()
            self.render_window.Render()
        return callback

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

        from src.vtk.io import save_vtp

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

    # -- events ------------------------------------------------------------

    def _on_color(self, widget, event):
        self.cycle_color_mode()
        widget.GetRepresentation().SetState(0)
        self.render_window.Render()

    def _on_save_key(self):
        """'s', routed through the same reporting as the button."""
        try:
            path = self.export_vtp()
        except Exception as exc:                       # noqa: BLE001 - reported, not raised
            self.status.SetInput(f"could not write the .vtp: {exc}")
        else:
            self.status.SetInput(f"wrote {path}")
        self.render_window.Render()

    def _on_reset(self, widget, event):
        self.reset_to_mean()
        # The button is momentary, not a toggle: put the plate back to idle so
        # it does not sit lit after the click.
        widget.GetRepresentation().SetState(0)
        self.render_window.Render()

    def _on_key(self, interactor, event):
        key = (interactor.GetKeySym() or "").lower()
        if key in _OUR_KEYS:
            # vtkInteractorStyleTrackballCamera binds most single letters
            # itself - 'c' switches to actor mode, 's' forces surface
            # representation - so a key we act on has to stop here.
            interactor.GetCommand(self._key_tag).SetAbortFlag(1)
        if key == "c":
            self.cycle_color_mode()
        elif key == "s":
            self._on_save_key()
        elif key == "m":
            self.toggle_dropdown()
        elif key == "escape" and self.dropdown_open:
            self.toggle_dropdown(False)
        elif key in ("bracketleft", "bracketright"):
            # Step to the neighbouring model without opening the list, which is
            # the fast way to compare two kernels built from the same domain.
            here = self.model_names.index(self.config.model)
            step = -1 if key == "bracketleft" else 1
            self.switch_model(self.model_names[(here + step) % len(self.model_names)])
        elif key == "r":
            self.reset_to_mean()
            self.render_window.Render()
        elif key == "t":
            self.toggle_theme()
        elif key == "prior":
            self.scroll_by(-self.n_rows)
        elif key == "next":
            self.scroll_by(self.n_rows)
        elif key == "home":
            self.scroll_by(-len(self.alpha))
        elif key == "end":
            self.scroll_by(len(self.alpha))

    def _place_labels(self):
        """The sliders are in normalised coordinates; the text is not."""
        width, height = self.render_window.GetSize()
        x = int(_CONTENT_X * width)
        # Under the dropdown bar, which is what now carries the model's name.
        self.info.SetDisplayPosition(x, height - _DROPDOWN_TOP - _DROPDOWN_BAR_H - 18)
        self.status.SetDisplayPosition(x, 38)
        self.hint.SetDisplayPosition(x, 16)
        self.panel_header.SetDisplayPosition(int(_PANEL_X[0] * width), height - 30)

    def _on_resize(self, interactor, event):
        self._place_button()
        self._place_labels()
        self._place_dropdown()

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
