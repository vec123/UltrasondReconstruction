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

Which model opens is a config key ('baby' or 'fetus'), and nothing below is
specific to either: both arrive through :func:`src.model.load.load_shape_model`
as the same mean/eigenfunctions/eigenvalues triple.
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
    },
}

# Panel geometry, in normalised display coordinates: the slider column and the
# scrollbar beside it. _PANEL_Y is the centre of the top and bottom rows.
_PANEL_X = (0.030, 0.205)
_PANEL_Y = (0.880, 0.155)
_SCROLLBAR_X = (0.216, 0.224)
# Where everything right of the panel starts.
_CONTENT_X = 0.26

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
        self.scalar_bar: Optional[vtk.vtkScalarBarActor] = None
        self.ghost: Optional[vtk.vtkActor] = None
        self.written_files: List[Path] = []

    def run(self) -> SampleShapesResult:
        self.load_model()
        self.build_scene()
        self.build_widgets()

        self.reset_to_mean()
        self.render_window.Render()

        if self.config.screenshot is not None:
            self.save_screenshot(self.config.screenshot)

        if self.config.interactive:
            self._log("window open - drag the sliders, scroll the panel for more modes, "
                      "'r' resets to the mean, 't' switches theme")
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
        self.faces = load_shape_model_triangulation(source.path)

        if source.parametrization == "lddmm":
            self._load_momentum_model()
            return

        mean, eigenfunctions, eigenvalues, _ = load_shape_model(
            source.path, normalize=source.normalize,
            atlas_options=source.atlas.to_options(), verbose=cfg.verbose,
        )

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

        # Per-vertex sd under the *whole* model, not just the exposed modes: it
        # is the natural scale for the colour ramp and the one mean_shape.py
        # writes, so the two pictures stay comparable.
        per_mode_sq = (eigenfunctions.reshape(len(self.mean), 3, -1) ** 2).sum(axis=1)
        self.total_std = np.sqrt(per_mode_sq @ eigenvalues)
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
        self._log_model()

    def _log_model(self):
        self._log(f"  {len(self.mean)} vertices, {self.n_modes_total} modes "
                  f"({len(self.alpha)} on the panel, {self.variance_share.sum():.1%} "
                  f"of the variance), {0 if self.faces is None else len(self.faces)} triangles")
        self._log(f"  mode 0 carries {self.variance_share[0]:.1%}, "
                  f"the last exposed mode {self.variance_share[-1]:.2%}")
        self._log(f"  per-vertex sd {self.total_std.min():.4g} .. {self.total_std.max():.4g} "
                  f"in model units")

    # -- the scene ---------------------------------------------------------

    def build_scene(self):
        view = self.config.view

        # The point buffer is handed to VTK without a copy and then written in
        # place, so a slider drag costs one matrix product and a Modified() -
        # no reallocation, no rebuilding of the mesh.
        self.vertices = np.array(self.mean, dtype=np.float32, order="C")
        self.polydata = create_polydata(self.vertices, self.faces)
        self._points = numpy_support.numpy_to_vtk(self.vertices, deep=False)
        self.polydata.GetPoints().SetData(self._points)

        self.scalars = np.zeros(len(self.mean), dtype=np.float32)
        if view.color_by == "total_std":
            self.scalars[:] = self.total_std
        self._scalar_array = numpy_support.numpy_to_vtk(self.scalars, deep=False)
        self._scalar_array.SetName(view.color_by)
        self.polydata.GetPointData().AddArray(self._scalar_array)

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
            mapper.SetInputData(self.polydata)

        if view.color_by == "none":
            mapper.ScalarVisibilityOff()
        else:
            self.lookup_table = vtk.vtkColorTransferFunction()
            self._resolve_color_range()
            mapper.SetLookupTable(self.lookup_table)
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray(view.color_by)
            mapper.SetScalarRange(0.0, self.color_range)
            mapper.ScalarVisibilityOn()

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
        if view.color_by != "none" and view.show_scalar_bar:
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
        self.color_range = self.config.view.color_range
        if self.color_range is None:
            # Fixed for the whole session, from the model rather than from the
            # current sample: a range that followed the sample would repaint
            # the surface on every drag and mean nothing.
            self.color_range = float(self.config.sliders.sigma_range * self.total_std.max())
        if self.color_range <= 0:
            raise ValueError(f"view.color_range must be positive, got {self.color_range}")

    def _scalar_bar(self) -> vtk.vtkScalarBarActor:
        titles = {"displacement": "displacement", "total_std": "per-vertex sd"}
        bar = vtk.vtkScalarBarActor()
        bar.SetLookupTable(self.lookup_table)
        bar.SetTitle(f"{titles[self.config.view.color_by]}\n(model units)")
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

        # The wheel belongs to the panel while the cursor is over it and to the
        # camera everywhere else, so scrolling the modes and zooming the shape
        # never fight over it.
        self._wheel_tags = {
            event: self.interactor.AddObserver(event, self._on_wheel, 10.0)
            for event in ("MouseWheelForwardEvent", "MouseWheelBackwardEvent")
        }
        # The same actions from the keyboard, since the sliders already own the
        # mouse and reaching for the button breaks the flow of exploring.
        self.interactor.AddObserver("KeyPressEvent", self._on_key)
        self.interactor.AddObserver("ConfigureEvent", self._on_resize)
        self.interactor.Initialize()

        self.apply_theme(self.theme_name)
        self.scroll_to(0)

    def _build_sliders(self):
        """The scrolling panel: a fixed pool of rows, re-bound as it scrolls.

        Only the geometry is set up here. Which mode each row *is* comes from
        :meth:`scroll_to`, so the widget count stays at ``visible_rows``
        however many modes the model has.
        """
        self.n_rows = min(int(self.config.sliders.visible_rows), len(self.alpha))
        self.first_row = 0
        limit = float(self.config.sliders.sigma_range)

        top, bottom = _PANEL_Y
        spacing = (top - bottom) / max(self.n_rows - 1, 1) if self.n_rows > 1 else 0.0

        for row in range(self.n_rows):
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
    def _quad():
        """A filled rectangle in normalised viewport coordinates."""
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
        self.button = vtk.vtkButtonWidget()
        self.button.SetInteractor(self.interactor)

        representation = vtk.vtkTexturedButtonRepresentation2D()
        representation.SetNumberOfStates(2)
        representation.SetPlaceFactor(1)
        self.button.SetRepresentation(representation)
        self._place_button()
        self.button.On()
        self.button.AddObserver("StateChangedEvent", self._on_reset)

        # Added *after* the widget is enabled: 2D props draw in the order they
        # were added, and the button's own texture would otherwise cover the
        # label that names it.
        self.button_label = vtk.vtkTextActor()
        self.button_label.SetInput("RESET TO MEAN")
        prop = self.button_label.GetTextProperty()
        prop.SetFontSize(13)
        prop.SetJustificationToCentered()
        prop.SetVerticalJustificationToCentered()
        prop.BoldOn()
        prop.ShadowOff()
        self.renderer.AddActor2D(self.button_label)
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
        """Below the slider column, in display (pixel) coordinates."""
        width, height = self.render_window.GetSize()
        x0, y0 = _PANEL_X[0] * width, 0.035 * height
        x1, y1 = x0 + 190, y0 + 38
        self.button.GetRepresentation().PlaceWidget([x0, x1, y0, y1, 0.0, 0.0])
        if self.button_label is not None:
            self.button_label.SetPosition((x0 + x1) / 2.0, (y0 + y1) / 2.0)

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
        """The panel header, the status line, and the controls hint."""
        exposed = "all" if len(self.alpha) == self.n_modes_total else str(len(self.alpha))
        self.title = self._text_actor(15, bold=True)
        self.title.SetInput(f"{self.config.model}   -   {self.n_modes_total} modes, "
                            f"{exposed} on the panel")

        # Sits above the panel and says where in the mode list you are.
        self.panel_header = self._text_actor(12)

        self.status = self._text_actor(14)
        self.hint = self._text_actor(12)
        self.hint.SetInput(
            "drag a slider to move along that mode      "
            "scroll over the panel for more modes (PgUp / PgDn, Home / End)      "
            "'r' resets to the mean      't' switches theme"
        )
        self._place_labels()

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

        if self.config.view.color_by != "none":
            # The ramp runs the other way per theme, so the low end always
            # recedes towards the surface rather than shouting from it.
            self.lookup_table.RemoveAllPoints()
            ramp = theme["ramp"]
            for i, hex_color in enumerate(ramp):
                self.lookup_table.AddRGBPoint(
                    self.color_range * i / (len(ramp) - 1), *_rgb(hex_color)
                )

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

        representation = self.button.GetRepresentation()
        representation.SetButtonTexture(0, self._button_image(theme["button"]))
        representation.SetButtonTexture(1, self._button_image(theme["button_active"]))
        self.button_label.GetTextProperty().SetColor(*theme["button_text"])

        for actor in (self.title, self.status):
            actor.GetTextProperty().SetColor(*theme["text"])
        for actor in (self.hint, self.panel_header):
            actor.GetTextProperty().SetColor(*theme["muted"])
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
        if not self._over_panel():
            return                      # let the camera style zoom as usual
        self.scroll_by(-1 if event == "MouseWheelForwardEvent" else 1)
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
        if self.config.view.color_by == "displacement":
            self.scalars[:] = magnitude
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

    def reset_to_mean(self):
        """Back to alpha = 0 - every mode, not only the rows on screen."""
        self.alpha[:] = 0.0
        for widget in self.sliders:
            widget.GetRepresentation().SetValue(0.0)
        self.update_shape()

    # -- events ------------------------------------------------------------

    def _on_reset(self, widget, event):
        self.reset_to_mean()
        # The button is momentary, not a toggle: put the plate back to idle so
        # it does not sit lit after the click.
        widget.GetRepresentation().SetState(0)
        self.render_window.Render()

    def _on_key(self, interactor, event):
        key = (interactor.GetKeySym() or "").lower()
        if key == "r":
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
        self.title.SetDisplayPosition(x, height - 30)
        self.status.SetDisplayPosition(x, 38)
        self.hint.SetDisplayPosition(x, 16)
        self.panel_header.SetDisplayPosition(int(_PANEL_X[0] * width), height - 30)

    def _on_resize(self, interactor, event):
        self._place_button()
        self._place_labels()

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
