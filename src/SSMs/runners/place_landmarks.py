"""Interactive vertex picker: click the mesh, choose a name, either order.

The BabyFM .mat carries 23 named landmarks baked into the file; the fetus
atlas carries none (see the module docstring of :mod:`src.model.deformetrica`).
This window is how that gap gets filled by hand: it renders a model's shape,
coloured by curvature so the anatomy - nose tip, eye sockets, chin - is easy to
find by eye, and exposes the same mode sliders ``sample_shapes.py`` opens the
shape space with, so a landmark can be placed on a deformed instance where the
point of interest is easier to see than at the mean. A landmark is just a
vertex *index*, so wherever it is placed it stays valid at every point in
shape space - and under either of the two ways the model can move through
that space:

* linear - the default. mean + sum_k alpha_k phi_k, PCA on ambient vertex
  coordinates. One matrix product, so a slider drag redraws every frame.
* lddmm - 'l' toggles it, atlas models only. PCA on the initial momenta
  instead, and every sample is reached by actually shooting the geodesic, so
  it cannot fold the surface through itself the way the linear sum can. Costs
  an integration per sample, so drags are throttled and the shape catches up
  on release, same as sample_shapes.py's fetus_lddmm.

Switching resets the sliders to zero: a coefficient in one basis has no
meaning in the other, so there is nothing sensible to carry over.

Picking and naming are deliberately two separate actions, committed together
rather than one triggering the other:

* click a vertex - it becomes the *pending* pick (a distinct marker, no name
  yet);
* Up/Down move which checklist name is *current* - independently, before or
  after the click;
* Enter links the two, writing the pending vertex under the current name.

So "pick, then choose a name" and "choose a name, then pick" both work; only
Enter commits either way.

'x' exports mean.vtp alongside landmarks_json: the mean shape (alpha = 0) -
for the linear parametrization that mean *is* the atlas template - carrying
the same fields mean_shape.py attaches: total_std, mode_XX_std, curvature,
landmark_distance, nearest_landmark, one heatmap_<name> per landmark
currently placed, and the combined heatmap_landmarks. It is a live preview of
what mean_shape.py will produce once landmarks_json is wired into
mean_shape_fetus.yaml, scoped to whatever is placed so far rather than the
full checklist.

'n'/'p' step through the atlas's real training subjects (atlas models only),
shooting each one's own momentum and displaying it in place of the alpha
sample - the most faithful "a different sample" there is for checking a
placement, since it carries none of a slider combination's risk of landing
outside anything a real fetus looks like. Touching a slider (or 'm') goes
back to the alpha sample; shot subjects are cached, since revisiting one
while comparing is the point. 'h' toggles the surface between curvature and
a live heatmap_landmarks - heat released at every *currently placed*
landmark and diffused across whichever shape is on screen, so a landmark
that has drifted off its anatomy shows up as a hot spot sitting off the
feature it should be centred on. Both exist for the same reason: judging
whether a landmark still means the same anatomical point away from the mean
it was placed at.

Picking is wired to the interactor directly (mirroring the wheel/panel split
in ``sample_shapes.py``) rather than through a custom interactor style, and it
fires on plain ``LeftButtonPressEvent`` - the same event the default trackball
camera's own rotate-drag starts from - so a click always registers a pending
pick even mid-drag; nothing is written until Enter confirms it, so that costs
nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import vtk
from vtk.util import numpy_support

from src.config.schema import PlaceLandmarksConfig
from src.geometry.geometry_np import compute_mean_curvature
from src.geometry.heatmap import landmark_heatmaps
from src.geometry.landmark_fields import build_landmark_field_polydata
from src.geometry.vtk import add_point_field, create_polydata, save_vtp_mesh
from src.model.deformetrica import load_momentum_model, shoot
from src.model.landmarks import read_landmark_map, write_landmark_map
from src.model.load import is_atlas_path, load_shape_model, load_shape_model_triangulation


def _rgb(hex_color: str):
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


_THEMES = {
    "light": {
        "background": (0.988, 0.988, 0.984),
        "text": (0.043, 0.043, 0.043),
        "muted": (0.322, 0.318, 0.306),
        "rail": (0.847, 0.843, 0.827),
        "track": (0.902, 0.898, 0.882),
        "knob": _rgb("#2a78d6"),
        "knob_active": _rgb("#1c5cab"),
        "thumb": _rgb("#3987e5"),
        "surface": (0.78, 0.76, 0.72),
        "concave": _rgb("#3f8fae"),
        "neutral": (0.80, 0.78, 0.74),
        "convex": _rgb("#d9793d"),
        "marker": _rgb("#8e3b9e"),
        "marker_current": _rgb("#e0a72c"),
        "marker_pending": _rgb("#1fb6a6"),
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
        "surface": (0.82, 0.80, 0.76),
        "concave": _rgb("#6db8d8"),
        "neutral": (0.42, 0.41, 0.38),
        "convex": _rgb("#f0a05c"),
        "marker": _rgb("#c77dd1"),
        "marker_current": _rgb("#f0c14b"),
        "marker_pending": _rgb("#5be0cf"),
    },
}

_AXIS_VECTORS = {
    "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
}

# Slider panel geometry - the same region sample_shapes.py uses, so a user of
# one recognises the other. The landmark checklist therefore lives on the
# opposite (right) side instead of sharing the left column.
_SLIDER_X = (0.030, 0.205)
_SLIDER_Y = (0.880, 0.155)
_SCROLLBAR_X = (0.216, 0.224)
_CONTENT_X = 0.26
_CHECKLIST_X = 0.80


@dataclass
class PlaceLandmarksResult:
    landmarks: Dict[str, int]
    names: List[str]
    written_files: List[Path] = field(default_factory=list)


class PlaceLandmarksRunner:
    """Builds the picker window and runs it until the user closes it."""

    def __init__(self, config: PlaceLandmarksConfig):
        self.config = config

        self.mean: Optional[np.ndarray] = None        # (N, 3) alpha = 0
        self.vertices: Optional[np.ndarray] = None     # (N, 3) the shape on screen
        self.faces: Optional[np.ndarray] = None
        self.centroid: Optional[np.ndarray] = None
        self.radius: float = 0.0
        self.marker_radius: float = 0.0
        self._outward: Optional[np.ndarray] = None     # (N, 3) unit vectors, for label offsets

        # The currently *active* triple - one of (linear_*, lddmm_*) below,
        # always kept the same length as each other. Rebound by
        # toggle_parametrization(); alpha is reset to zero whenever that happens.
        self.sigma: Optional[np.ndarray] = None        # (K,)
        self.alpha: Optional[np.ndarray] = None         # (K,)
        self.variance_share: Optional[np.ndarray] = None
        self.parametrization = "linear"

        self.linear_basis: Optional[np.ndarray] = None  # (3N, K_lin)
        self.linear_sigma: Optional[np.ndarray] = None
        self.linear_variance_share: Optional[np.ndarray] = None

        self.momentum_model = None                      # MomentumShapeModel, or None: no atlas
        self.lddmm_sigma: Optional[np.ndarray] = None
        self.lddmm_variance_share: Optional[np.ndarray] = None
        self._min_interval = 0.0                         # LDDMM-only slider-drag throttle
        self._last_update = 0.0

        # Real training subjects, browsed with n/p - alongside, not instead
        # of, the alpha sliders: None means "showing the alpha sample" (the
        # usual mode), an int means "showing atlas.momenta[index] shot
        # directly", ignoring alpha entirely until a slider is touched again.
        # Shot shapes are cached since re-shooting is the expensive part and
        # comparing landmark placement across subjects means revisiting them.
        self.subject_index: Optional[int] = None
        self._subject_cache: Dict[int, np.ndarray] = {}

        # heatmap_landmarks coloring - 'h' toggles it in place of curvature.
        # Recomputing it is a sparse factorisation, too slow for every slider
        # drag frame, so it only refreshes on "settled" state changes (drag
        # release, subject change, a landmark placed/cleared, ...), not here.
        self._showing_heatmap = False
        self._heatmap_values: Optional[np.ndarray] = None
        self._heatmap_vtk_array = None
        self._heatmap_lut: Optional[vtk.vtkColorTransferFunction] = None

        self.n_modes_total = 0
        self.sliders: List[vtk.vtkSliderWidget] = []
        self.n_rows = 0
        self.first_row = 0

        self.names: List[str] = []
        self.placements: Dict[str, int] = {}
        self.current_index: int = 0
        self.pending_vertex: Optional[int] = None
        self._pending_marker: Optional[dict] = None
        self._dirty = False
        self._editing = False
        self._edit_buffer = ""

        self.markers: Dict[str, dict] = {}             # name -> {sphere, sphere_actor, label}
        self.theme_name = config.theme
        self.theme = _THEMES[self.theme_name]
        self.scalar_bar: Optional[vtk.vtkScalarBarActor] = None
        self.representation = "surface"
        self.written_files: List[Path] = []

    def run(self) -> PlaceLandmarksResult:
        self.load_model()
        self.load_landmarks()
        self.build_scene()
        self.build_ui()

        for name in list(self.placements):
            self._refresh_marker(name)
        self._refresh_panel()
        self._refresh_status()
        self.render_window.Render()

        self._log("window open - click a vertex (it becomes the pending pick), "
                  "Up/Down chooses a landmark name, Enter links the two - either "
                  "order works. 'a' adds a custom name, Delete clears the current "
                  "one, 'm' resets the shape to the mean, 'l' toggles linear/LDDMM, "
                  "'n'/'p' step through real subjects, 'h' toggles heatmap coloring, "
                  "'s' saves, 'x' exports mean.vtp, 'q' quits (autosaves on close)")
        self.interactor.Start()

        self.save()
        return PlaceLandmarksResult(
            landmarks=dict(self.placements),
            names=list(self.names),
            written_files=list(self.written_files),
        )

    # -- inputs --------------------------------------------------------------

    def load_model(self):
        cfg = self.config
        if not cfg.shape_model.exists():
            raise FileNotFoundError(f"shape_model does not exist: {cfg.shape_model}")

        mean, eigenfunctions, eigenvalues, _ = load_shape_model(
            cfg.shape_model, normalize=cfg.model.normalize,
            atlas_options=cfg.model.atlas.to_options(), verbose=cfg.verbose,
        )
        self.mean = np.asarray(mean, dtype=np.float64).reshape(-1, 3)
        self.faces = load_shape_model_triangulation(cfg.shape_model)
        self.centroid = self.mean.mean(axis=0)
        self.radius = float(np.linalg.norm(self.mean - self.centroid, axis=1).max())
        self.marker_radius = (
            float(cfg.marker_radius) if cfg.marker_radius is not None else self.radius * 0.006
        )

        outward = self.mean - self.centroid
        norm = np.linalg.norm(outward, axis=1, keepdims=True)
        norm[norm < 1e-12] = 1.0
        self._outward = outward / norm

        eigenvalues = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
        n_modes = len(eigenvalues)
        if cfg.sliders.mode_count is not None:
            n_modes = min(int(cfg.sliders.mode_count), n_modes)

        self.linear_basis = np.ascontiguousarray(
            np.asarray(eigenfunctions, dtype=np.float64)[:, :n_modes]
        )
        self.linear_sigma = np.sqrt(eigenvalues[:n_modes])
        self.linear_variance_share = (
            eigenvalues[:n_modes] / eigenvalues.sum() if eigenvalues.sum() > 0
            else np.zeros(n_modes)
        )
        self.n_modes_total = len(eigenvalues)

        self.parametrization = "linear"
        self.sigma = self.linear_sigma
        self.variance_share = self.linear_variance_share
        self.alpha = np.zeros(n_modes)

        self._log(f"model: {cfg.shape_model}")
        self._log(f"  {len(self.mean)} vertices, "
                  f"{0 if self.faces is None else len(self.faces)} triangles, "
                  f"{self.n_modes_total} modes ({n_modes} on the panel)")

        self._load_momentum_model()

    def _load_momentum_model(self):
        """LDDMM sliders, alongside the linear ones - atlas models only.

        Loaded eagerly rather than on first 'l' press: it is just an SVD on
        the momenta (184 x 1236 here), fast enough not to matter, and this
        way the toggle can report unavailability up front instead of only
        when pressed.
        """
        cfg = self.config
        if not is_atlas_path(cfg.shape_model):
            self._log("  LDDMM unavailable: not a Deformetrica atlas (no momenta to shoot)")
            return

        try:
            self.momentum_model = load_momentum_model(
                cfg.shape_model, kernel_width=cfg.model.atlas.kernel_width,
                time_points=cfg.lddmm_time_points, verbose=False,
            )
        except Exception as exc:                       # noqa: BLE001 - optional feature, not fatal
            self._log(f"  LDDMM unavailable: {type(exc).__name__}: {exc}")
            return

        k = self.momentum_model.n_modes
        if cfg.sliders.mode_count is not None:
            k = min(int(cfg.sliders.mode_count), k)
        self.lddmm_sigma = np.sqrt(self.momentum_model.eigenvalues[:k])
        self.lddmm_variance_share = self.momentum_model.variance_share[:k]
        self._log(f"  LDDMM available: {self.momentum_model.n_modes} momentum modes "
                  f"({k} on the panel), {cfg.lddmm_time_points} shooting time points")

    def load_landmarks(self):
        cfg = self.config
        n_vertices = len(self.mean)
        loaded = read_landmark_map(cfg.landmarks_json)

        self.names = list(cfg.landmark_names)
        for name in loaded:
            if name not in self.names:
                self.names.append(name)
        if not self.names:
            raise ValueError(
                "landmark_names is empty and landmarks_json has no entries to fall back on"
            )

        dropped = []
        for name, index in loaded.items():
            if 0 <= index < n_vertices:
                self.placements[name] = index
            else:
                dropped.append((name, index))
        if dropped:
            self._log(f"  {len(dropped)} landmark(s) dropped (out of range for "
                      f"{n_vertices} vertices): {dropped}")

        self.current_index = 0
        for i, name in enumerate(self.names):
            if name not in self.placements:
                self.current_index = i
                break

        self._log(f"landmarks_json: {cfg.landmarks_json}")
        self._log(f"  {len(self.placements)}/{len(self.names)} already placed")

    # -- the scene -------------------------------------------------------------

    def build_scene(self):
        cfg = self.config

        # A live buffer: the polydata's points become a zero-copy view onto
        # this array (numpy_to_vtk(deep=False)), so a slider drag is an
        # in-place numpy write followed by Modified() - no rebuilding, the
        # same trick sample_shapes.py uses.
        self.vertices = np.array(self.mean, dtype=np.float32, order="C")
        self.polydata = create_polydata(self.vertices, self.faces)
        self._points = numpy_support.numpy_to_vtk(self.vertices, deep=False)
        self.polydata.GetPoints().SetData(self._points)

        mapper = vtk.vtkPolyDataMapper()
        if self.faces is not None:
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(self.polydata)
            normals.ComputePointNormalsOn()
            normals.ComputeCellNormalsOff()
            normals.SplittingOff()
            normals.ConsistencyOn()
            mapper.SetInputConnection(normals.GetOutputPort())
        else:
            mapper.SetInputData(self.polydata)

        self.curvature = None
        if cfg.color_by == "curvature":
            self.curvature = self._compute_curvature()
        if self.curvature is not None:
            add_point_field(self.polydata, self.curvature.astype(np.float32), "curvature")
            self.lookup_table = vtk.vtkColorTransferFunction()
            self._rebuild_curvature_lut()

        # heatmap_landmarks: wired unconditionally (independent of
        # cfg.color_by) so 'h' works even when curvature coloring is off. A
        # live zero-copy array, same trick as self._points - _refresh_heatmap()
        # writes into it in place rather than re-adding it to the polydata.
        self._heatmap_values = np.zeros(len(self.mean), dtype=np.float32)
        self._heatmap_vtk_array = numpy_support.numpy_to_vtk(self._heatmap_values, deep=False)
        self._heatmap_vtk_array.SetName("heatmap_landmarks")
        self.polydata.GetPointData().AddArray(self._heatmap_vtk_array)
        self._heatmap_lut = vtk.vtkColorTransferFunction()
        self._rebuild_heatmap_lut()

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().SetSpecular(0.15)
        self.actor.GetProperty().SetSpecularPower(20)

        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.actor)
        self.scalar_bar = self._scalar_bar()
        self.renderer.AddActor2D(self.scalar_bar)
        self._apply_color_mode()

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetSize(*cfg.window_size)
        self.render_window.SetWindowName(f"place landmarks - {cfg.shape_model.name}")
        self.render_window.AddRenderer(self.renderer)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        self.point_picker = vtk.vtkPointPicker()
        self.point_picker.SetTolerance(cfg.pick_tolerance)
        self.interactor.SetPicker(self.point_picker)

        self._place_camera()

    def _compute_curvature(self) -> Optional[np.ndarray]:
        try:
            curvature = np.asarray(
                compute_mean_curvature(self.mean, self.faces), dtype=np.float64
            ).reshape(-1)
        except Exception as exc:                       # noqa: BLE001 - colouring only, not fatal
            self._log(f"  curvature skipped, falling back to flat colour: "
                      f"{type(exc).__name__}: {exc}")
            return None
        return curvature

    def _rebuild_curvature_lut(self):
        self.curvature_range = float(max(np.percentile(np.abs(self.curvature), 97), 1e-12))
        theme = self.theme
        self.lookup_table.RemoveAllPoints()
        self.lookup_table.AddRGBPoint(-self.curvature_range, *theme["concave"])
        self.lookup_table.AddRGBPoint(0.0, *theme["neutral"])
        self.lookup_table.AddRGBPoint(self.curvature_range, *theme["convex"])

    def _rebuild_heatmap_lut(self):
        """0 (no landmark nearby) -> 1 (at a landmark), the same warm colour
        curvature uses for 'convex' so a hot spot reads the same intuition."""
        theme = self.theme
        self._heatmap_lut.RemoveAllPoints()
        self._heatmap_lut.AddRGBPoint(0.0, *theme["neutral"])
        self._heatmap_lut.AddRGBPoint(1.0, *theme["convex"])

    def _apply_color_mode(self):
        """Points the mapper (and the scalar bar) at whichever array is
        active: heatmap_landmarks if 'h' is on, else curvature if configured,
        else a flat surface colour."""
        mapper = self.actor.GetMapper()
        if self._showing_heatmap:
            mapper.SetLookupTable(self._heatmap_lut)
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray("heatmap_landmarks")
            mapper.SetScalarRange(0.0, 1.0)
            mapper.ScalarVisibilityOn()
            self.scalar_bar.SetLookupTable(self._heatmap_lut)
            self.scalar_bar.SetTitle("landmark\ncoverage")
            self.scalar_bar.SetVisibility(1)
        elif self.curvature is not None:
            mapper.SetLookupTable(self.lookup_table)
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray("curvature")
            mapper.SetScalarRange(-self.curvature_range, self.curvature_range)
            mapper.ScalarVisibilityOn()
            self.scalar_bar.SetLookupTable(self.lookup_table)
            self.scalar_bar.SetTitle("curvature")
            self.scalar_bar.SetVisibility(1)
        else:
            mapper.ScalarVisibilityOff()
            self.actor.GetProperty().SetColor(*self.theme["surface"])
            self.scalar_bar.SetVisibility(0)

    def _scalar_bar(self) -> vtk.vtkScalarBarActor:
        bar = vtk.vtkScalarBarActor()
        bar.SetNumberOfLabels(3)
        bar.SetMaximumWidthInPixels(60)
        bar.SetMaximumHeightInPixels(200)
        bar.GetPositionCoordinate().SetValue(0.915, 0.06)
        bar.SetWidth(0.07)
        bar.SetHeight(0.26)
        bar.UnconstrainedFontSizeOn()
        bar.GetTitleTextProperty().SetFontSize(13)
        bar.GetLabelTextProperty().SetFontSize(11)
        for prop in (bar.GetTitleTextProperty(), bar.GetLabelTextProperty()):
            prop.SetBold(False)
            prop.ShadowOff()
            prop.ItalicOff()
        return bar

    def _place_camera(self):
        cfg = self.config
        axis = np.array(_AXIS_VECTORS[cfg.camera_axis])
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*self.centroid)
        camera.SetPosition(*(self.centroid + axis * self.radius * 3.0))
        camera.SetViewUp(*_AXIS_VECTORS[cfg.view_up])
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()

    # -- shape space (mode sliders) --------------------------------------------

    def sample(self) -> np.ndarray:
        """(N, 3) the shape the current coefficients stand for.

        Linear: one matrix product. LDDMM: an integration - the momentum is
        built from *only* the exposed modes (self.alpha's length), so
        self.momentum_model.modes is sliced to match rather than calling its
        reconstruct()/momentum(), which multiply against every mode it has and
        would raise (or silently do the wrong thing) the moment sliders.mode_count
        truncates the panel below the model's full rank.
        """
        if self.parametrization == "lddmm":
            model = self.momentum_model
            k = len(self.alpha)
            momentum = model.mean_momentum
            if k:
                momentum = momentum + model.modes[:, :k] @ self.alpha
            return shoot(
                model.atlas.control_points, momentum.reshape(-1, 3), model.atlas.template,
                model.atlas.kernel_width, self.config.lddmm_time_points,
            )
        if len(self.alpha) == 0:
            return self.mean
        return self.mean + (self.linear_basis @ self.alpha).reshape(-1, 3)

    def update_shape(self):
        """Rebuilds the on-screen shape in place: the alpha sample, or a
        browsed real subject (self.subject_index) if one is active - browsing
        bypasses the sliders entirely rather than reading alpha at all."""
        if self.subject_index is not None:
            source = self._subject_cache[self.subject_index]
        else:
            source = self.sample()
        np.copyto(self.vertices, source, casting="unsafe")
        self._points.Modified()
        self.polydata.Modified()
        self._reposition_all_markers()

    def reset_to_mean(self):
        self.subject_index = None
        self.alpha[:] = 0.0
        for widget in self.sliders:
            widget.GetRepresentation().SetValue(0.0)
        self.update_shape()
        self._refresh_heatmap()
        self._flash_status("shape reset to the mean")

    def toggle_parametrization(self):
        """linear <-> lddmm - see the module docstring for what each means.

        Resets the sliders to zero: alpha is a coefficient in whichever
        basis is active, and a linear-PCA coefficient means nothing as an
        LDDMM momentum coefficient (or the reverse), so there is nothing
        sensible to carry across the switch.
        """
        if self.momentum_model is None:
            self._flash_status("LDDMM unavailable - this model has no atlas momenta to shoot")
            return

        if self.parametrization == "linear":
            self.parametrization = "lddmm"
            self.sigma = self.lddmm_sigma
            self.variance_share = self.lddmm_variance_share
            self._min_interval = 0.15   # a sample now costs an integration
        else:
            self.parametrization = "linear"
            self.sigma = self.linear_sigma
            self.variance_share = self.linear_variance_share
            self._min_interval = 0.0

        self.subject_index = None
        self.alpha = np.zeros(len(self.sigma))
        for widget in self.sliders:
            widget.GetRepresentation().SetValue(0.0)
        if self.sliders:
            self.scroll_to(self.first_row)
        self.update_shape()
        self._refresh_heatmap()
        self._flash_status(f"parametrization: {self.parametrization} (shape reset to the mean)")

    def _show_subject(self, index: int):
        """Shoots (or reuses the cached shot of) real training subject
        ``index`` and displays it, bypassing the alpha sliders entirely.

        Unlike a slider sample, this is not a point in the *model's* shape
        space at all - it is one of the actual shapes the model was fit to,
        the most faithful "a different sample" there is for checking whether
        a landmark tracks its anatomy, since it carries none of a synthetic
        combination's risk of landing outside what any real fetus looks like.
        """
        atlas = self.momentum_model.atlas
        index = int(index) % atlas.n_subjects
        self.subject_index = index
        if index not in self._subject_cache:
            self._log(f"  shooting subject {index + 1}/{atlas.n_subjects}")
            momentum = atlas.momenta[index]
            self._subject_cache[index] = shoot(
                atlas.control_points, momentum, atlas.template,
                atlas.kernel_width, self.config.lddmm_time_points,
            )
        self.update_shape()
        self._refresh_heatmap()
        self._flash_status(f"subject {index + 1}/{atlas.n_subjects} (a real fetus, not a sample)")

    def _step_subject(self, delta: int):
        if self.momentum_model is None:
            self._flash_status("subject browsing unavailable - this model has no atlas momenta")
            return
        start = (-1 if delta > 0 else 0) if self.subject_index is None else self.subject_index
        self._show_subject(start + delta)

    def _refresh_heatmap(self):
        """Recomputes heatmap_landmarks on the *current* on-screen shape.

        Skipped while it is not the active colouring: the recompute is a
        sparse factorisation, cheap enough for a drag release or a subject
        change but too slow to run on every InteractionEvent of a drag.
        """
        if not self._showing_heatmap:
            return
        if not self.placements:
            self._heatmap_values[:] = 0.0
        else:
            names = list(self.placements.keys())
            indices = np.array(list(self.placements.values()), dtype=np.int64)
            _, combined = landmark_heatmaps(
                self.vertices.astype(np.float64), self.faces, names, indices,
                want_per_landmark=False, want_combined=True, combined_mode="max",
                log=self._log if self.config.verbose else None,
            )
            self._heatmap_values[:] = combined
        self._heatmap_vtk_array.Modified()
        self.polydata.Modified()

    def toggle_color_mode(self):
        """curvature (or flat) <-> heatmap_landmarks, recomputed fresh on the
        current shape so it reads correctly right after a subject change or a
        parametrization toggle, even though those did not themselves keep it
        updated while it was hidden."""
        self._showing_heatmap = not self._showing_heatmap
        self._refresh_heatmap()
        self._apply_color_mode()
        if self._showing_heatmap:
            note = "" if self.placements else " - none placed yet, flat"
            self._flash_status(f"coloring: landmark coverage ({len(self.placements)} landmark(s){note})")
        else:
            self._flash_status(f"coloring: {'curvature' if self.curvature is not None else 'flat'}")

    def toggle_representation(self):
        """Surface <-> raw vertices - the point cloud shows through folds and
        the far side of the shape, where a filled surface would hide them."""
        self.representation = "points" if self.representation == "surface" else "surface"
        prop = self.actor.GetProperty()
        if self.representation == "points":
            prop.SetRepresentationToPoints()
            prop.SetPointSize(self.config.point_size)
        else:
            prop.SetRepresentationToSurface()
        self._flash_status(f"surface shown as {self.representation}")

    # -- the UI --------------------------------------------------------------

    def build_ui(self):
        self.title = self._text_actor(15, bold=True)
        self.title.SetInput(f"place landmarks - {self.config.shape_model.name}")

        self.panel = self._text_actor(12)
        self.slider_header = self._text_actor(12)
        self.status = self._text_actor(13)
        self.message = self._text_actor(12)
        self.hint = self._text_actor(11)
        self.hint.SetInput(
            "click = pending pick      Up/Down = choose name      Enter = link them      "
            "a = custom name      Delete = clear current      m = reset shape      "
            "l = linear/LDDMM      n/p = next/prev real subject      h = heatmap coloring      "
            "v = surface/points      s = save      x = export mean.vtp      "
            "t = theme      c = reset camera      q = quit (autosaves)"
        )

        if len(self.alpha):
            self._build_sliders()
            self._build_scrollbar()

        self._place_labels()
        self.apply_theme(self.theme_name)
        if len(self.alpha):
            self.scroll_to(0)

        # Picking: added directly on the interactor (not a custom style), same
        # split the wheel/panel handling in sample_shapes.py uses - so the
        # default trackball camera still owns rotate/pan/zoom untouched.
        self.interactor.AddObserver("LeftButtonPressEvent", self._on_left_press, 10.0)
        if self.sliders:
            self._wheel_tags = {
                event: self.interactor.AddObserver(event, self._on_wheel, 10.0)
                for event in ("MouseWheelForwardEvent", "MouseWheelBackwardEvent")
            }
        # Priority 10 + an aborted flag while editing: the trackball style has
        # its own key bindings on these same events ('q'/'e' quit, 'w' goes
        # wireframe), and a typed landmark name will contain those letters -
        # aborting stops the style from ever seeing the keystroke while a name
        # is being typed, the same way sample_shapes.py keeps the mouse wheel
        # from reaching the camera while it is over the slider panel.
        self._keypress_tag = self.interactor.AddObserver("KeyPressEvent", self._on_key, 10.0)
        self._char_tag = self.interactor.AddObserver("CharEvent", self._on_char, 10.0)
        self.interactor.AddObserver("ConfigureEvent", self._on_resize)
        self.interactor.Initialize()

    def _text_actor(self, size: int, bold: bool = False) -> vtk.vtkTextActor:
        actor = vtk.vtkTextActor()
        prop = actor.GetTextProperty()
        prop.SetFontSize(size)
        prop.SetBold(bold)
        prop.ShadowOff()
        prop.ItalicOff()
        self.renderer.AddActor2D(actor)
        return actor

    def _place_labels(self):
        width, height = self.render_window.GetSize()
        self.title.SetDisplayPosition(int(_CONTENT_X * width), height - 30)
        self.slider_header.SetDisplayPosition(int(_SLIDER_X[0] * width), height - 30)
        self.panel.SetDisplayPosition(int(_CHECKLIST_X * width), height - 30)
        self.status.SetDisplayPosition(int(_CONTENT_X * width), 60)
        self.message.SetDisplayPosition(int(_CONTENT_X * width), 38)
        self.hint.SetDisplayPosition(int(_CONTENT_X * width), 16)

    def _on_resize(self, interactor, event):
        self._place_labels()

    # -- mode sliders (ported from sample_shapes.py) --------------------------

    def _build_sliders(self):
        cfg = self.config
        self.n_rows = min(int(cfg.sliders.visible_rows), len(self.alpha))
        self.first_row = 0
        limit = float(cfg.sliders.sigma_range)

        top, bottom = _SLIDER_Y
        spacing = (top - bottom) / max(self.n_rows - 1, 1) if self.n_rows > 1 else 0.0

        for row in range(self.n_rows):
            y = top - row * spacing
            representation = vtk.vtkSliderRepresentation2D()
            representation.SetMinimumValue(-limit)
            representation.SetMaximumValue(limit)
            representation.SetValue(0.0)
            representation.SetLabelFormat("%+.2f")

            for coordinate, x in ((representation.GetPoint1Coordinate(), _SLIDER_X[0]),
                                  (representation.GetPoint2Coordinate(), _SLIDER_X[1])):
                coordinate.SetCoordinateSystemToNormalizedDisplay()
                coordinate.SetValue(x, y)

            representation.SetSliderLength(0.016)
            representation.SetSliderWidth(0.020)
            representation.SetEndCapLength(0.003)
            representation.SetEndCapWidth(0.018)
            representation.SetTubeWidth(0.005)
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
        self.scroll_track, self._track_points = self._quad()
        self.scroll_thumb, self._thumb_points = self._quad()
        self.renderer.AddActor2D(self.scroll_track)
        self.renderer.AddActor2D(self.scroll_thumb)

        top, bottom = _SLIDER_Y
        self._set_quad(self._track_points, _SCROLLBAR_X, (bottom - 0.03, top + 0.03))

    @staticmethod
    def _quad():
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

    def scroll_to(self, first_row: int):
        """Rebinds the row pool to a window of modes starting at ``first_row``.

        The row pool is a fixed size (self.n_rows), chosen once at startup
        from the linear model's mode count. Toggling to LDDMM can leave fewer
        modes than that if the two models' ranks happen to differ - rare (both
        come from the same 184 subjects) but not guaranteed - so any row past
        the end of the *current* self.alpha is disabled rather than indexed.
        """
        n_modes = len(self.alpha)
        last_start = max(0, n_modes - self.n_rows)
        self.first_row = int(np.clip(first_row, 0, last_start))

        for row, widget in enumerate(self.sliders):
            mode = self.first_row + row
            representation = widget.GetRepresentation()
            if mode >= n_modes:
                representation.SetTitleText("")
                representation.SetValue(0.0)
                widget.SetEnabled(0)
                continue
            widget.SetEnabled(1)
            representation.SetTitleText(f"mode {mode}    {self.variance_share[mode]:.2%} var")
            representation.SetValue(float(self.alpha[mode] / self.sigma[mode]))

        top, bottom = _SLIDER_Y
        track_low, track_high = bottom - 0.03, top + 0.03
        span = track_high - track_low
        visible_rows = min(self.n_rows, n_modes) if n_modes else self.n_rows
        height = min(span, max(span * visible_rows / float(max(n_modes, 1)), 0.02))
        offset = 0.0 if last_start == 0 else self.first_row / float(last_start)
        thumb_top = track_high - offset * (span - height)
        self._set_quad(self._thumb_points, _SCROLLBAR_X, (thumb_top - height, thumb_top))

        self.slider_header.SetInput(
            f"modes {self.first_row}-{self.first_row + visible_rows - 1} of {n_modes}"
        )

    def scroll_by(self, rows: int):
        self.scroll_to(self.first_row + rows)
        self.render_window.Render()

    def _over_panel(self) -> bool:
        x, _ = self.interactor.GetEventPosition()
        return x <= (_SCROLLBAR_X[1] + 0.01) * self.render_window.GetSize()[0]

    def _on_wheel(self, interactor, event):
        if not self._over_panel():
            return
        self.scroll_by(-1 if event == "MouseWheelForwardEvent" else 1)
        interactor.GetCommand(self._wheel_tags[event]).SetAbortFlag(1)

    def _on_slider(self, row: int):
        """Callback for panel row ``row``.

        An LDDMM sample costs an integration (self._min_interval > 0), so a
        drag is throttled and the release always gets the last word - same
        trade sample_shapes.py makes for its fetus_lddmm model. A linear
        sample is a matrix product and redraws on every event.
        """
        def callback(widget, event):
            mode = self.first_row + row
            if mode >= len(self.alpha):
                return
            self.subject_index = None
            self.alpha[mode] = widget.GetRepresentation().GetValue() * self.sigma[mode]

            if event == "InteractionEvent" and self._min_interval > 0.0:
                if time.time() - self._last_update < self._min_interval:
                    return
            self.update_shape()
            self._last_update = time.time()
            if event == "EndInteractionEvent":
                self._refresh_heatmap()
            self.render_window.Render()
        return callback

    # -- theme -----------------------------------------------------------------

    def apply_theme(self, name: str):
        self.theme_name = name
        self.theme = _THEMES[name]
        theme = self.theme

        self.renderer.SetBackground(*theme["background"])
        if self.curvature is not None:
            self._rebuild_curvature_lut()
        self._rebuild_heatmap_lut()
        self._apply_color_mode()
        if self.scalar_bar is not None:
            self.scalar_bar.GetTitleTextProperty().SetColor(*theme["muted"])
            self.scalar_bar.GetLabelTextProperty().SetColor(*theme["muted"])

        for actor in (self.title, self.status):
            actor.GetTextProperty().SetColor(*theme["text"])
        for actor in (self.panel, self.slider_header, self.message, self.hint):
            actor.GetTextProperty().SetColor(*theme["muted"])

        for widget in self.sliders:
            representation = widget.GetRepresentation()
            representation.GetSliderProperty().SetColor(*theme["knob"])
            representation.GetSelectedProperty().SetColor(*theme["knob_active"])
            representation.GetTubeProperty().SetColor(*theme["rail"])
            representation.GetCapProperty().SetColor(*theme["rail"])
            for prop in (representation.GetTitleProperty(), representation.GetLabelProperty()):
                prop.SetColor(*theme["muted"])
        if self.sliders:
            self.scroll_track.GetProperty().SetColor(*theme["track"])
            self.scroll_thumb.GetProperty().SetColor(*theme["thumb"])

        for name_ in list(self.placements):
            self._refresh_marker(name_)
        if self.pending_vertex is not None:
            self._update_pending_style()

    def toggle_theme(self):
        self.apply_theme("dark" if self.theme_name == "light" else "light")
        self.render_window.Render()

    # -- landmark state --------------------------------------------------------

    def _set_current(self, index: int):
        old_name = self.names[self.current_index] if self.names else None
        self.current_index = int(index) % len(self.names)
        new_name = self.names[self.current_index]

        if old_name is not None and old_name != new_name and old_name in self.placements:
            self._update_marker_style(old_name, is_current=False)
        if new_name in self.placements:
            self._update_marker_style(new_name, is_current=True)

        self._refresh_panel()
        self._refresh_status()

    def _move_current(self, delta: int):
        self._set_current(self.current_index + delta)

    def _commit_pending(self):
        """Links the pending pick to the current name. Does not move the
        cursor afterwards - there is no canonical next landmark, so which one
        to do next is entirely up to Up/Down, not an implied checklist order."""
        if self.pending_vertex is None:
            self._flash_status("nothing pending - click a vertex first")
            return
        name = self.names[self.current_index]
        self.placements[name] = self.pending_vertex
        self._dirty = True
        vertex = self.pending_vertex
        self._clear_pending()
        self._refresh_marker(name)
        self._refresh_panel()
        self._refresh_heatmap()
        self._flash_status(f"placed '{name}' at vertex {vertex}")

    def _clear_current(self):
        name = self.names[self.current_index]
        if name not in self.placements:
            self._flash_status(f"'{name}' has no placement to clear")
            return
        del self.placements[name]
        self._dirty = True
        self._remove_marker(name)
        self._refresh_heatmap()
        self._flash_status(f"cleared '{name}'")
        self._refresh_panel()
        self._refresh_status()

    # -- pending pick (the click that hasn't been named yet) --------------------

    def _set_pending(self, vertex: int):
        self.pending_vertex = int(vertex)
        self._ensure_pending_marker()
        self._update_pending_geometry()
        self._update_pending_style()
        self._flash_status(
            f"picked vertex {vertex} - Enter assigns it to "
            f"'{self.names[self.current_index]}' (Up/Down to change), Esc discards"
        )

    def _clear_pending(self):
        self.pending_vertex = None
        if self._pending_marker is not None:
            self.renderer.RemoveActor(self._pending_marker["actor"])
            self._pending_marker = None

    def _discard_pending(self):
        if self.pending_vertex is None:
            return
        self._clear_pending()
        self._flash_status("discarded the pending pick")

    def _ensure_pending_marker(self):
        if self._pending_marker is not None:
            return
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(self.marker_radius * 1.15)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().LightingOff()
        actor.PickableOff()
        self.renderer.AddActor(actor)
        self._pending_marker = {"sphere": sphere, "actor": actor}

    def _update_pending_geometry(self):
        if self._pending_marker is None or self.pending_vertex is None:
            return
        self._pending_marker["sphere"].SetCenter(*self.vertices[self.pending_vertex])

    def _update_pending_style(self):
        if self._pending_marker is None:
            return
        self._pending_marker["actor"].GetProperty().SetColor(*self.theme["marker_pending"])

    # -- markers -----------------------------------------------------------

    def _remove_marker(self, name: str):
        marker = self.markers.pop(name, None)
        if marker is None:
            return
        self.renderer.RemoveActor(marker["sphere_actor"])
        self.renderer.RemoveActor(marker["label"])

    def _ensure_marker(self, name: str):
        if name in self.markers:
            return
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(self.marker_radius)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(mapper)
        sphere_actor.GetProperty().LightingOff()
        # Otherwise a click near an existing marker can hand vtkPointPicker the
        # sphere glyph's own (unrelated) point id instead of a mesh vertex.
        sphere_actor.PickableOff()
        self.renderer.AddActor(sphere_actor)

        label = vtk.vtkBillboardTextActor3D()
        label.SetInput(name)
        prop = label.GetTextProperty()
        prop.SetFontSize(14)
        prop.SetJustificationToCentered()
        prop.ShadowOff()
        prop.SetBackgroundOpacity(0.55)
        label.PickableOff()
        self.renderer.AddActor(label)

        self.markers[name] = {"sphere": sphere, "sphere_actor": sphere_actor, "label": label}

    def _update_marker_geometry(self, name: str, vertex: int):
        marker = self.markers.get(name)
        if marker is None:
            return
        position = self.vertices[vertex]
        marker["sphere"].SetCenter(*position)
        marker["label"].SetPosition(*(position + self._outward[vertex] * (self.marker_radius * 4.0)))

    def _update_marker_style(self, name: str, is_current: bool):
        marker = self.markers.get(name)
        if marker is None:
            return
        color = self.theme["marker_current"] if is_current else self.theme["marker"]
        marker["sphere_actor"].GetProperty().SetColor(*color)
        marker["label"].GetTextProperty().SetColor(*self.theme["text"])
        marker["label"].GetTextProperty().SetBackgroundColor(*self.theme["background"])

    def _refresh_marker(self, name: str):
        """Full sync for one landmark: create if new, then geometry and style."""
        if name not in self.placements:
            self._remove_marker(name)
            return
        self._ensure_marker(name)
        self._update_marker_geometry(name, self.placements[name])
        self._update_marker_style(name, self.names[self.current_index] == name)

    def _reposition_all_markers(self):
        """Cheap geometry-only refresh, called after every shape update."""
        for name, vertex in self.placements.items():
            self._update_marker_geometry(name, vertex)
        self._update_pending_geometry()

    # -- text panels ---------------------------------------------------------

    def _refresh_panel(self):
        lines = [f"landmarks  ({len(self.placements)}/{len(self.names)} placed)", ""]
        for i, name in enumerate(self.names):
            cursor = ">" if i == self.current_index else " "
            box = "x" if name in self.placements else " "
            lines.append(f"{cursor} [{box}] {name}")
        self.panel.SetInput("\n".join(lines))

    def _refresh_status(self):
        if self._editing:
            text = f"new landmark name: {self._edit_buffer}_      (Enter = confirm, Esc = cancel)"
        else:
            current = self.names[self.current_index]
            state = "placed" if current in self.placements else "not yet placed"
            pending = "   pending pick ready to link" if self.pending_vertex is not None else ""
            dirty = "   *unsaved changes*" if self._dirty else ""
            if self.subject_index is not None:
                shape_label = f"subject {self.subject_index + 1}/{self.momentum_model.atlas.n_subjects}"
            else:
                shape_label = self.parametrization
            color_label = "heatmap" if self._showing_heatmap else "curvature"
            text = (f"[{shape_label}][{color_label}]  current target: {current}  ({state})      "
                    f"{len(self.placements)}/{len(self.names)} placed{pending}{dirty}")
        self.status.SetInput(text)

    def _flash_status(self, message: str):
        self.message.SetInput(message)
        self._refresh_status()

    # -- editing (typing a custom name) ---------------------------------------

    def _start_edit(self):
        self._editing = True
        self._edit_buffer = ""
        self._refresh_status()

    def _confirm_edit(self):
        name = self._edit_buffer.strip().replace(" ", "_")
        self._editing = False
        self._edit_buffer = ""

        if not name:
            self._flash_status("cancelled: empty name")
        elif name in self.names:
            self._set_current(self.names.index(name))
            self._flash_status(f"'{name}' is already on the list - selected it")
        else:
            self.names.append(name)
            self._set_current(len(self.names) - 1)
            self._flash_status(f"added '{name}' to the checklist")
        self._refresh_panel()

    def _cancel_edit(self):
        self._editing = False
        self._edit_buffer = ""
        self._flash_status("cancelled")

    # -- events --------------------------------------------------------------

    def _on_left_press(self, interactor, event):
        """Picks on button-down rather than disambiguating click vs. drag.

        An earlier version tracked press and release positions separately and
        only picked when they were within a few pixels of each other - but
        that made picking depend on both events firing and on that bookkeeping
        being right, which turned out to be a real source of "clicking does
        nothing". Picking directly off the same LeftButtonPressEvent that
        drives the trackball camera's own rotate-start removes that dependency
        entirely: if the camera can be rotated at all, this event is firing.
        The cost is that starting a rotate-drag also sets a pending pick at
        the drag's starting point - harmless, since Enter is still required to
        write anything, and Esc discards it.
        """
        if self._editing:
            return
        x, y = interactor.GetEventPosition()
        hit = self.point_picker.Pick(x, y, 0, self.renderer)
        vertex = self.point_picker.GetPointId()
        self._log(f"  click at ({x}, {y}) -> hit={bool(hit)} vertex={vertex}")
        if not hit or vertex < 0:
            self._flash_status(f"no vertex under the click at ({x}, {y})")
        else:
            self._set_pending(int(vertex))
        self.render_window.Render()

    def _on_char(self, interactor, event):
        if not self._editing:
            return
        interactor.GetCommand(self._char_tag).SetAbortFlag(1)
        code = interactor.GetKeyCode()
        if not code or ord(code) < 32:
            return                            # control chars: handled in _on_key
        if len(self._edit_buffer) >= 40:
            return
        if code.isalnum() or code in "_- ":
            self._edit_buffer += code
            self._refresh_status()
            self.render_window.Render()

    def _on_key(self, interactor, event):
        keysym = interactor.GetKeySym() or ""

        if self._editing:
            interactor.GetCommand(self._keypress_tag).SetAbortFlag(1)
            if keysym in ("Return", "KP_Enter"):
                self._confirm_edit()
            elif keysym == "Escape":
                self._cancel_edit()
            elif keysym == "BackSpace":
                self._edit_buffer = self._edit_buffer[:-1]
                self._refresh_status()
            else:
                return
            self.render_window.Render()
            return

        key = keysym.lower()
        if key in ("return", "kp_enter"):
            self._commit_pending()
        elif key == "escape":
            self._discard_pending()
        elif key == "up":
            self._move_current(-1)
        elif key == "down":
            self._move_current(1)
        elif key in ("delete", "backspace"):
            self._clear_current()
        elif key == "a":
            self._start_edit()
        elif key == "m":
            self.reset_to_mean()
        elif key == "l":
            self.toggle_parametrization()
        elif key == "v":
            self.toggle_representation()
        elif key == "s":
            self.save(explicit=True)
        elif key == "x":
            self.export_mean_vtp()
        elif key == "h":
            self.toggle_color_mode()
        elif key == "n":
            self._step_subject(1)
        elif key == "p":
            self._step_subject(-1)
        elif key == "t":
            self.toggle_theme()
            return                            # toggle_theme renders itself
        elif key == "c":
            self._place_camera()
        elif key == "prior" and self.sliders:
            self.scroll_by(-self.n_rows)
            return                            # scroll_by renders itself
        elif key == "next" and self.sliders:
            self.scroll_by(self.n_rows)
            return
        elif key == "home" and self.sliders:
            self.scroll_by(-len(self.alpha))
            return
        elif key == "end" and self.sliders:
            self.scroll_by(len(self.alpha))
            return
        else:
            return
        self.render_window.Render()

    # -- output ----------------------------------------------------------------

    def save(self, explicit: bool = False):
        write_landmark_map(self.config.landmarks_json, self.placements)
        self._dirty = False
        if self.config.landmarks_json not in self.written_files:
            self.written_files.append(self.config.landmarks_json)
        if explicit:
            self._flash_status(
                f"saved {len(self.placements)} landmark(s) to {self.config.landmarks_json.name}"
            )
        else:
            self._refresh_status()
        self._log(f"saved {len(self.placements)} landmark(s) to {self.config.landmarks_json}")

    def export_mean_vtp(self):
        """Writes mean.vtp: the mean shape (alpha = 0) with the same fields
        mean_shape.py attaches - total_std, mode_XX_std, curvature,
        landmark_distance, nearest_landmark, heatmap_<name>, heatmap_landmarks
        - except the landmark ones are scoped to whatever is placed *right
        now* in this window, not the full checklist mean_shape.py would need
        landmarks_json written out for.

        The mean rather than the on-screen sample: a landmark is a vertex
        index, valid at any point in shape space, but every field here is a
        property of the model's own mean surface, the same reference picture
        mean_shape.py exports - and for the linear parametrization at
        alpha = 0, that mean *is* the atlas template. total_std/mode_XX_std
        always come from the linear basis for that reason, never the LDDMM
        one, even mid-LDDMM-toggle: mean_shape.py has no LDDMM concept to
        match.
        """
        if not self.placements:
            self._flash_status("no landmarks placed yet - nothing to export")
            return

        polydata = build_landmark_field_polydata(
            self.mean, self.faces, self.placements,
            linear_basis=self.linear_basis, linear_sigma=self.linear_sigma,
            log=self._log if self.config.verbose else None,
        )

        path = self.config.landmarks_json.parent / "mean.vtp"
        save_vtp_mesh(polydata, str(path), verbose=False)
        if path not in self.written_files:
            self.written_files.append(path)
        self._flash_status(f"exported mean.vtp ({len(self.placements)} landmark field(s))")
        self._log(f"exported mean shape with {len(self.placements)} landmark field(s) to {path}")

    def _log(self, *args):
        if self.config.verbose:
            print(*args)
