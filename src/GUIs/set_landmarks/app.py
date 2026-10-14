"""The landmark picker window: click a vertex, choose a name, either order; Enter links them.

The BabyFM .mat carries 23 named landmarks baked into the file; the fetus
atlas carries none (see :mod:`src.SSM.model.deformetrica`). This window fills
that gap by hand. It shows the model's mean shape coloured by curvature, so
the anatomy - nose tip, eye sockets, chin - is easy to find, and the same mode
sliders ``sample_shapes.py`` uses, so a landmark can be placed on a deformed
instance where its anatomy is easier to see (:mod:`.shape_space`).

Picking and naming are separate actions (:mod:`.checklist`): a click sets the
*pending* pick, Up/Down move the *current* name, Enter writes one under the
other. 'h' recolours the surface by a live ``heatmap_landmarks`` - heat
released at every placed landmark and diffused over the shape on screen - so a
landmark that has drifted off its anatomy away from the mean shows up as a hot
spot beside the feature it should sit on. 'x' exports ``mean.vtp`` beside
``landmarks_json`` with the field set ``mean_shape.py`` writes, scoped to what
is placed so far.

This module is the composition root only: state lives in
:class:`~.checklist.LandmarkChecklist` and :class:`~.shape_space.ShapeSpace`,
every VTK widget comes from :mod:`src.GUIs.core`, and the methods here keep
the two in sync. The public action methods are what the keys are bound to,
so the window can also be driven without a keyboard (e.g. in tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.config.SSM.schema import PlaceLandmarksConfig
from src.geometry import compute_mean_curvature
from src.geometry.heatmap import landmark_heatmaps
from src.geometry.landmark_fields import build_landmark_field_polydata
from src.GUIs.core import (
    DIVERGING, SEQUENTIAL, FlatColoring, Hud, KeyboardRouter, KeyMap, MeshView,
    ModeSliderPanel, PanelGeometry, ScalarColoring, SurfaceColoring, TextPrompt, TextSlot,
    Throttle, VertexMarkers, VertexPicker, Viewer, get_theme, next_theme, symmetric_range,
)
from src.GUIs.set_landmarks.checklist import AddResult, LandmarkChecklist
from src.GUIs.set_landmarks.formatting import (
    HINT, START_MESSAGE, checklist_text, prompt_text, status_text,
)
from src.GUIs.set_landmarks.shape_space import load_shape_space
from src.SSM.model.landmarks import read_landmark_map, write_landmark_map
from src.vtk.io import save_vtp

# -- layout ---------------------------------------------------------------------
# The slider column on the left (same region as sample_shapes.py, so a user of
# one recognises the other), the checklist on the right, text in between.
_PANEL = PanelGeometry(x=(0.030, 0.205), y=(0.880, 0.155), scrollbar_x=(0.216, 0.224))
_CONTENT_X = 0.26
_CHECKLIST_X = 0.80
_TEXT_SLOTS = (
    TextSlot("title", _CONTENT_X, 30, from_top=True, size=15, bold=True, role="text"),
    TextSlot("checklist", _CHECKLIST_X, 30, from_top=True, size=12),
    TextSlot("status", _CONTENT_X, 60, size=13, role="text"),
    TextSlot("message", _CONTENT_X, 38, size=12),
    TextSlot("hint", _CONTENT_X, 16, size=11),
)

_COVERAGE = "coverage"           # colour mode: live heatmap_landmarks
_COVERAGE_FIELD = "heatmap_landmarks"
_PENDING = "pending"             # the one key in the pending-pick marker set


def _quiet(*args) -> None:
    pass


@dataclass
class SetLandmarksResult:
    landmarks: Dict[str, int]
    names: List[str]
    written_files: List[Path] = field(default_factory=list)


class SetLandmarksApp:
    """Builds the picker window from a :class:`PlaceLandmarksConfig` and runs it until closed."""

    def __init__(self, config: PlaceLandmarksConfig, *, offscreen: bool = False):
        """Store the config; nothing is loaded or opened until :meth:`build` / :meth:`run`.

        Args:
            config: what to load, where to save, and how the window looks.
            offscreen: build an invisible window (tests, scripted use).
        """
        self.config = config
        self.offscreen = offscreen
        self._log = print if config.verbose else _quiet
        self.written_files: List[Path] = []
        self.panel: Optional[ModeSliderPanel] = None

    # -- lifecycle ---------------------------------------------------------------

    def run(self) -> SetLandmarksResult:
        """Open the window, block until it closes, autosave, and report what was placed."""
        self.build()
        self._log(START_MESSAGE)
        self.viewer.start()
        self.save()
        return self.result()

    def build(self) -> None:
        """Load the model and landmarks and assemble the window, without showing it."""
        self._load()
        self._build_scene()
        self._build_ui()
        for name in self.checklist.placements:
            self._place_marker(name)
        self._refresh_checklist()
        self._refresh_status()
        self.viewer.initialize()
        self.viewer.render()

    def result(self) -> SetLandmarksResult:
        return SetLandmarksResult(landmarks=dict(self.checklist.placements),
                                  names=list(self.checklist.names),
                                  written_files=list(self.written_files))

    def _load(self) -> None:
        cfg = self.config
        loaded = load_shape_space(cfg, self._log)
        self.space = loaded.space
        self.faces = loaded.faces

        self.checklist, dropped = LandmarkChecklist.from_saved(
            cfg.landmark_names, read_landmark_map(cfg.landmarks_json), len(self.space.mean)
        )
        if dropped:
            self._log(f"  {len(dropped)} landmark(s) dropped (out of range for "
                      f"{len(self.space.mean)} vertices): {dropped}")
        self._log(f"landmarks_json: {cfg.landmarks_json}")
        self._log(f"  {self.checklist.n_placed}/{len(self.checklist.names)} already placed")

    def _build_scene(self) -> None:
        cfg = self.config
        mean = self.space.mean
        self.centroid = mean.mean(axis=0)
        self.radius = float(np.linalg.norm(mean - self.centroid, axis=1).max())
        marker_radius = (float(cfg.marker_radius) if cfg.marker_radius is not None
                         else self.radius * 0.006)
        self.theme = get_theme(cfg.theme)

        self.viewer = Viewer(f"place landmarks - {cfg.shape_model.name}", cfg.window_size,
                             offscreen=self.offscreen)
        renderer = self.viewer.renderer
        self.mesh = MeshView(mean, self.faces)
        renderer.AddActor(self.mesh.actor)

        # Coverage is wired whatever color_by says, so 'h' always works.
        self.mesh.add_point_field(_COVERAGE_FIELD)
        coverage = ScalarColoring(_COVERAGE, _COVERAGE_FIELD, (0.0, 1.0), "landmark\ncoverage",
                                  SEQUENTIAL, label="heatmap")
        self.base_coloring = self._base_coloring()
        self.coloring = SurfaceColoring(self.mesh, renderer, [self.base_coloring, coverage],
                                        active=self.base_coloring.name, theme=self.theme)

        self.markers = VertexMarkers(renderer, self.mesh, marker_radius, self.theme)
        self.pending_marker = VertexMarkers(renderer, self.mesh, marker_radius * 1.15,
                                            self.theme, labelled=False)
        self.reset_camera()

    def _base_coloring(self):
        """Curvature from the mean shape (a fixed guide to where the anatomy is), else flat."""
        if self.config.color_by != "curvature":
            return FlatColoring()
        try:
            curvature = np.asarray(compute_mean_curvature(self.space.mean, self.faces),
                                   dtype=np.float64).reshape(-1)
        except Exception as exc:                       # noqa: BLE001 - colouring only, not fatal
            self._log(f"  curvature skipped, falling back to flat colour: "
                      f"{type(exc).__name__}: {exc}")
            return FlatColoring()
        self.mesh.add_point_field("curvature", curvature)
        return ScalarColoring("curvature", "curvature", symmetric_range(curvature),
                              "curvature", DIVERGING)

    def _build_ui(self) -> None:
        cfg = self.config
        self.hud = Hud(self.viewer.renderer, _TEXT_SLOTS)
        self.hud.set_text("title", f"place landmarks - {cfg.shape_model.name}")
        self.hud.set_text("hint", HINT)

        self.throttle = Throttle(self.space.drag_interval)
        if len(self.space.alpha):
            self.panel = ModeSliderPanel(self.viewer, self.space, cfg.sliders.visible_rows,
                                         cfg.sliders.sigma_range, self._on_slider, _PANEL)

        self.prompt = TextPrompt(on_confirm=self._on_name_typed,
                                 on_cancel=lambda: self._flash("cancelled"),
                                 on_change=self._refresh_status)
        self.picker = VertexPicker(self.viewer, cfg.pick_tolerance, self._on_pick, self._on_miss,
                                   enabled=lambda: not self.prompt.active, log=self._log)
        self.keymap = self._build_keymap()
        self.router = KeyboardRouter(self.viewer, self.keymap, self.prompt)

        self.viewer.on_resize(self._layout)
        self.apply_theme(cfg.theme)
        self.viewer.relayout()
        if self.panel is not None:
            self.panel.scroll_to(0)

    def _build_keymap(self) -> KeyMap:
        keys = KeyMap()
        keys.bind(("return", "kp_enter"), self.commit_pending, "link the pending pick to the current name")
        keys.bind("escape", self.discard_pending, "discard the pending pick")
        keys.bind("up", lambda: self.move_current(-1), "previous name")
        keys.bind("down", lambda: self.move_current(1), "next name")
        keys.bind(("delete", "backspace"), self.clear_current, "clear the current placement")
        keys.bind("a", self.start_custom_name, "type a new name")
        keys.bind("m", self.reset_to_mean, "reset the shape to the mean")
        keys.bind("l", self.toggle_parametrization, "linear / LDDMM sliders")
        keys.bind("n", lambda: self.step_subject(1), "next real subject")
        keys.bind("p", lambda: self.step_subject(-1), "previous real subject")
        keys.bind("h", self.toggle_coloring, "curvature / coverage heatmap")
        keys.bind("v", self.toggle_representation, "surface / points")
        keys.bind("s", lambda: self.save(explicit=True), "save landmarks_json")
        keys.bind("x", self.export_mean_vtp, "export mean.vtp")
        keys.bind("t", self.toggle_theme, "light / dark")
        keys.bind("c", self.reset_camera, "reset the camera")
        if self.panel is not None:
            panel = self.panel
            keys.bind("prior", lambda: panel.scroll_by(-panel.n_rows), "scroll modes up a page")
            keys.bind("next", lambda: panel.scroll_by(panel.n_rows), "scroll modes down a page")
            keys.bind("home", lambda: panel.scroll_by(-len(self.space.alpha)), "first modes")
            keys.bind("end", lambda: panel.scroll_by(len(self.space.alpha)), "last modes")
        return keys

    def _layout(self, width: int, height: int) -> None:
        self.hud.layout(width, height)
        if self.panel is not None:
            self.panel.layout(width, height)

    # -- landmark actions ------------------------------------------------------

    def commit_pending(self) -> None:
        placed = self.checklist.commit_pending()
        if placed is None:
            self._flash("nothing pending - click a vertex first")
            return
        name, vertex = placed
        self.pending_marker.clear()
        self._place_marker(name)
        self._refresh_checklist()
        self._refresh_coverage()
        self._flash(f"placed '{name}' at vertex {vertex}")

    def discard_pending(self) -> None:
        if not self.checklist.discard_pending():
            return
        self.pending_marker.clear()
        self._flash("discarded the pending pick")

    def move_current(self, delta: int) -> None:
        self._cursor_moved(*self.checklist.move(delta))

    def clear_current(self) -> None:
        name = self.checklist.clear_current()
        if name is None:
            self._flash(f"'{self.checklist.current_name}' has no placement to clear")
            return
        self.markers.remove(name)
        self._refresh_coverage()
        self._flash(f"cleared '{name}'")
        self._refresh_checklist()

    def start_custom_name(self) -> None:
        self.prompt.open()

    def _on_name_typed(self, text: str) -> None:
        previous = self.checklist.current_name
        result, name = self.checklist.add_name(text)
        if result is AddResult.EMPTY:
            self._flash("cancelled: empty name")
        else:
            self._cursor_moved(previous, self.checklist.current_name)
            if result is AddResult.EXISTING:
                self._flash(f"'{name}' is already on the list - selected it")
            else:
                self._flash(f"added '{name}' to the checklist")
        self._refresh_checklist()

    def _on_pick(self, vertex: int) -> None:
        self.checklist.set_pending(vertex)
        self.pending_marker.place(_PENDING, vertex, role="marker_pending")
        self._flash(f"picked vertex {vertex} - Enter assigns it to "
                    f"'{self.checklist.current_name}' (Up/Down to change), Esc discards")

    def _on_miss(self, x: int, y: int) -> None:
        self._flash(f"no vertex under the click at ({x}, {y})")

    def _cursor_moved(self, previous: str, current: str) -> None:
        if previous != current:
            self.markers.set_role(previous, "marker")
        self.markers.set_role(current, "marker_current")
        self._refresh_checklist()
        self._refresh_status()

    def _place_marker(self, name: str) -> None:
        role = "marker_current" if name == self.checklist.current_name else "marker"
        self.markers.place(name, self.checklist.placements[name], role)

    # -- shape-space actions --------------------------------------------------

    def reset_to_mean(self) -> None:
        self.space.reset()
        if self.panel is not None:
            self.panel.reset_values()
        self._show_shape()
        self._refresh_coverage()
        self._flash("shape reset to the mean")

    def toggle_parametrization(self) -> None:
        if not self.space.has_lddmm:
            self._flash("LDDMM unavailable - this model has no atlas momenta to shoot")
            return
        parametrization = self.space.toggle_parametrization()
        self.throttle.min_interval = self.space.drag_interval
        if self.panel is not None:
            self.panel.reset_values()
            self.panel.refresh()
        self._show_shape()
        self._refresh_coverage()
        self._flash(f"parametrization: {parametrization} (shape reset to the mean)")

    def step_subject(self, delta: int) -> None:
        if not self.space.has_lddmm:
            self._flash("subject browsing unavailable - this model has no atlas momenta")
            return
        index = self.space.step_subject(delta, log=self._log)
        self._show_shape()
        self._refresh_coverage()
        self._flash(f"subject {index + 1}/{self.space.n_subjects} (a real fetus, not a sample)")

    def _on_slider(self, mode: int, final: bool) -> None:
        """A slider moved ``alpha[mode]``. LDDMM drags are throttled; the release always redraws."""
        self.space.leave_subject()
        if not final and not self.throttle.ready():
            return
        self._show_shape()
        self.throttle.mark()
        if final:
            self._refresh_coverage()
        self.viewer.render()

    def _show_shape(self) -> None:
        self.mesh.set_vertices(self.space.current_shape())
        self.markers.reposition()
        self.pending_marker.reposition()

    # -- display actions ---------------------------------------------------------

    def toggle_coloring(self) -> None:
        """Base colouring (curvature or flat) <-> coverage heatmap, recomputed on the current shape."""
        showing_coverage = self.coloring.active.name == _COVERAGE
        self.coloring.set_active(self.base_coloring.name if showing_coverage else _COVERAGE)
        self._refresh_coverage()
        if self.coloring.active.name == _COVERAGE:
            note = "" if self.checklist.placements else " - none placed yet, flat"
            self._flash(f"coloring: landmark coverage ({self.checklist.n_placed} landmark(s){note})")
        else:
            self._flash(f"coloring: {self.base_coloring.label}")

    def _refresh_coverage(self) -> None:
        """Recompute heatmap_landmarks on the shape on screen - only while it is shown.

        The recompute is a sparse factorisation: fine on a drag release or a
        placement, too slow for every frame of a drag, and wasted while hidden.
        """
        if self.coloring.active.name != _COVERAGE:
            return
        placements = self.checklist.placements
        if not placements:
            self.mesh.set_point_field(_COVERAGE_FIELD, 0.0)
            return
        _, combined = landmark_heatmaps(
            self.mesh.vertices.astype(np.float64), self.faces, list(placements),
            np.array(list(placements.values()), dtype=np.int64),
            want_per_landmark=False, want_combined=True, combined_mode="max",
            log=self._log if self.config.verbose else None,
        )
        self.mesh.set_point_field(_COVERAGE_FIELD, combined)

    def toggle_representation(self) -> None:
        representation = self.mesh.toggle_representation(self.config.point_size)
        self._flash(f"surface shown as {representation}")

    def toggle_theme(self) -> None:
        self.apply_theme(next_theme(self.theme.name))

    def apply_theme(self, name: str) -> None:
        self.theme = get_theme(name)
        self.viewer.apply_theme(self.theme)
        self.coloring.apply_theme(self.theme)
        self.hud.apply_theme(self.theme)
        self.markers.apply_theme(self.theme)
        self.pending_marker.apply_theme(self.theme)
        if self.panel is not None:
            self.panel.apply_theme(self.theme)

    def reset_camera(self) -> None:
        cfg = self.config
        self.viewer.place_camera(self.centroid, self.radius, cfg.camera_axis, cfg.view_up)

    # -- output -----------------------------------------------------------------

    def save(self, explicit: bool = False) -> None:
        """Write landmarks_json (also done automatically when the window closes)."""
        path = self.config.landmarks_json
        write_landmark_map(path, self.checklist.placements)
        self.checklist.mark_saved()
        self._record_written(path)
        if explicit:
            self._flash(f"saved {self.checklist.n_placed} landmark(s) to {path.name}")
        else:
            self._refresh_status()
        self._log(f"saved {self.checklist.n_placed} landmark(s) to {path}")

    def export_mean_vtp(self) -> None:
        """Write mean.vtp beside landmarks_json: the mean shape with mean_shape.py's field set.

        The mean rather than the on-screen sample: every field is a property
        of the model's own mean surface (for the linear parametrization the
        mean *is* the template), and total_std/mode_XX_std always come from the
        linear basis, since mean_shape.py has no LDDMM equivalent to match.
        """
        placements = self.checklist.placements
        if not placements:
            self._flash("no landmarks placed yet - nothing to export")
            return
        polydata = build_landmark_field_polydata(
            self.space.mean, self.faces, dict(placements),
            linear_basis=self.space.linear_basis, linear_sigma=self.space.linear_sigma,
            log=self._log if self.config.verbose else None,
        )
        path = self.config.landmarks_json.parent / "mean.vtp"
        save_vtp(polydata, str(path), verbose=False)
        self._record_written(path)
        self._flash(f"exported mean.vtp ({len(placements)} landmark field(s))")
        self._log(f"exported mean shape with {len(placements)} landmark field(s) to {path}")

    def _record_written(self, path: Path) -> None:
        if path not in self.written_files:
            self.written_files.append(path)

    # -- text -----------------------------------------------------------------------

    def _refresh_checklist(self) -> None:
        self.hud.set_text("checklist", checklist_text(self.checklist))

    def _refresh_status(self) -> None:
        if self.prompt.active:
            text = prompt_text(self.prompt.buffer)
        else:
            text = status_text(self.checklist, self.space.label(), self.coloring.active.label)
        self.hud.set_text("status", text)

    def _flash(self, message: str) -> None:
        self.hud.set_text("message", message)
        self._refresh_status()
