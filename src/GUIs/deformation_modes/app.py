"""The deformation-mode editor: author axes, constraint sets and modes on a template, and preview them.

Layout: amplitude sliders on the left (one per compiled mode), the mesh in the
middle, a clickable side panel on the right (every item with its state, the
selected item's values, region steps, field and warnings, and the edit
history - see ``panel.py``), status and key hints along the bottom.

Working loop (mouse, or the keys in brackets):
  1. click an item in the panel (Up/Down) - or [+ new ...] (a / c / n);
  2. shape its region: Ctrl+click brushes a geodesic patch (Ctrl+Shift+click erases),
     [landmarks...] adds patches around named landmarks (k), [field...] selects by a
     point field (f), [mirror] mirrors it (m); each step is listed and [remove]-able;
  3. set its values: click the arrows or turn the wheel over a value (Left/Right, then , .);
  4. [compile] solves the selected mode (p, e for all) and its slider deforms the mesh.

Everything is saved as recipes (see ``src.deformation_modes.spec``) - how each
region was made, not just which vertices - and every edit is undoable.

This module only composes: the spec and its maths are ``src.deformation_modes``,
the document and its undo stack ``document.py``, the inspector rows
``properties.py``, the text ``formatting.py``, and every widget ``src.GUIs.core``.
Its public methods are what the keys are bound to, so it can be driven
without a keyboard too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from src.deformation_modes import (
    AwayFrom, Axis, Between, ConstraintSet, FieldThreshold, Mirror, Mode, NearLandmarks,
)
from src.deformation_modes import Brush as BrushOp
from src.deformation_modes.compile import mode_sites, sample
from src.GUIs.core import (
    DIVERGING, SEQUENTIAL, BrushTool, FieldPreview, FlatColoring, FrameGlyphs, Hud, Inspector,
    KeyboardRouter, KeyMap, ListPanel, MeshView, ModeSliderPanel, PanelGeometry, PlaneGlyph,
    ScalarColoring, SelectionOverlay, SurfaceColoring, TextPrompt, TextSlot, VectorGlyphs,
    VertexMarkers, VertexSelection, Viewer, get_theme, next_theme, spread_subset, symmetric_range,
)
from src.GUIs.deformation_modes.config import DeformationModesConfig
from src.GUIs.deformation_modes.document import GROUP_NOUNS, SETTINGS, EditorDocument, ItemRef
from src.GUIs.deformation_modes.formatting import HINTS, prompt_line, status_text
from src.GUIs.deformation_modes.panel import panel_rows
from src.GUIs.deformation_modes.properties import PropertyBuilder, join_rule, split_rule

# -- layout ---------------------------------------------------------------------
_PANEL = PanelGeometry(x=(0.030, 0.205), y=(0.880, 0.360), scrollbar_x=(0.216, 0.224))
_CONTENT_X = 0.26
_SIDE_X = 0.64
_SIDE_WIDTH = 0.35
_VIEW_SHIFT = 0.16          # camera window centre (-1..1): pushes the mesh left of the side panel
_TEXT_SLOTS = (
    TextSlot("title", _CONTENT_X, 16, from_top=True, size=15, bold=True, role="text"),
    TextSlot("status", _CONTENT_X, 84, size=13, role="text"),
    TextSlot("message", _CONTENT_X, 62, size=12),
    TextSlot("hint", _CONTENT_X, 10, size=11),
)

_NAME_CHARS = lambda c: c.isalnum() or c in "_-"
_LABEL_CHARS = lambda c: c.isalnum() or c in "_ ,"
_FIELD_CHARS = lambda c: c.isalnum() or c in "_<>.-+ "
_FIELD_RULE = re.compile(r"^\s*([A-Za-z_]\w*)\s*([<>])\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")
_SKIP_COLOR_FIELDS = ("heatmap_", "mode_", "nearest_landmark", "normal")
_NEW_STEMS = {"axes": "axis", "constraint_sets": "set", "modes": "mode"}


def _quiet(*args) -> None:
    pass


@dataclass
class DeformationModesResult:
    path: Path
    saved: bool
    n_axes: int
    n_constraint_sets: int
    n_modes: int


class DeformationModesApp:
    """Builds the editor window from a :class:`DeformationModesConfig` and runs it until closed."""

    def __init__(self, config: DeformationModesConfig, *, offscreen: bool = False):
        self.config = config
        self.offscreen = offscreen
        self._log = print if config.verbose else _quiet
        self._prompt_action: Optional[Callable[[str], None]] = None
        self._show_landmarks = False
        self._glyph_key = None
        self._glyph_data: Dict[str, tuple] = {}
        self._saved = False

    # -- lifecycle -----------------------------------------------------------------

    def run(self) -> DeformationModesResult:
        """Open the window, block until it closes, save if anything changed, report."""
        self.build()
        self._log("\n".join(HINTS))
        self.viewer.start()
        if self.doc.dirty:
            self.save()
        return self.result()

    def result(self) -> DeformationModesResult:
        ms = self.doc.modeset
        return DeformationModesResult(self.doc.path, self._saved, len(ms.axes),
                                      len(ms.constraint_sets), len(ms.modes))

    def build(self) -> None:
        """Load the template and spec and assemble the window, without showing it."""
        cfg = self.config
        self.doc, created = EditorDocument.open(cfg.template, cfg.spec, cfg.sigma)
        mesh = self.doc.mesh
        self.radius = mesh.radius
        self.theme = get_theme(cfg.theme)

        self.viewer = Viewer(f"deformation modes - {Path(cfg.template).name}", cfg.window_size,
                             offscreen=self.offscreen)
        renderer = self.viewer.renderer
        self.view = MeshView(mesh.vertices, mesh.faces)
        renderer.AddActor(self.view.actor)
        self.coloring = SurfaceColoring(self.view, renderer, self._color_modes(),
                                        active=self._color_modes_first, theme=self.theme)

        self.selection = VertexSelection(mesh.n_vertices)
        self.context = VertexSelection(mesh.n_vertices)
        self.selection_overlay = SelectionOverlay(renderer, self.view, self.selection, self.theme,
                                                  point_size=5.0)
        self.context_overlay = SelectionOverlay(renderer, self.view, self.context, self.theme,
                                                point_size=3.0, role="marker")
        self.vectors = VectorGlyphs(renderer, self.theme, role="axis_t", scale=0.10 * self.radius)
        self.frames = FrameGlyphs(renderer, self.theme, length=0.035 * self.radius)
        self.plane = PlaneGlyph(renderer, self.theme)
        try:
            self.plane.set_plane(mesh.midline.point, mesh.midline.normal, 2.2 * self.radius)
        except ValueError:
            pass                                       # no bilateral landmarks: no midline
        self.landmark_markers = VertexMarkers(renderer, self.view, 0.012 * self.radius, self.theme)

        self.preview = FieldPreview(self.view, renderer, self.theme, faces=mesh.faces,
                                    arrow_count=cfg.arrow_count,
                                    reference_fraction=cfg.reference_fraction)
        self.preview.attach_coloring(self.coloring)
        self.preview.on_shape_changed(self._follow_shape)

        self.hud = Hud(renderer, _TEXT_SLOTS)
        self.hud.set_text("title", f"deformation modes - {Path(cfg.template).name}  ->  "
                                   f"{Path(cfg.spec).name}")
        self.hud.set_text("hint", "\n".join(HINTS))
        self.side = ListPanel(self.viewer, self.theme, x=_SIDE_X, top=16, width=_SIDE_WIDTH,
                              on_click=self._on_panel_click, on_wheel=self._on_panel_wheel,
                              pitch=16, bottom=110)
        self.panel = ModeSliderPanel(self.viewer, self.preview, cfg.visible_rows, cfg.slider_range,
                                     self._on_amplitude, _PANEL, row_title=self.preview.row_title,
                                     fixed_pool=True, noun="modes")
        self.preview.on_fields_changed(self.panel.refresh)

        self.inspector = Inspector()
        self.properties = PropertyBuilder(self.doc, self._after_edit)
        self.prompt = TextPrompt(on_confirm=self._on_prompt, on_cancel=lambda: self._flash("cancelled"),
                                 on_change=self._refresh_text)
        self.brush = BrushTool(self.viewer, None, mesh.geodesic_ball, cfg.brush_radius * self.radius,
                               tolerance=cfg.pick_tolerance, min_radius=0.005 * self.radius,
                               max_radius=self.radius, on_stroke=self._on_stroke,
                               on_miss=lambda x, y: self._flash(f"no surface under ({x}, {y})"))
        self.router = KeyboardRouter(self.viewer, self._build_keymap(), self.prompt)

        self.viewer.on_resize(self._layout)
        self.apply_theme(self.theme.name)
        self.viewer.relayout()
        self.panel.scroll_to(0)
        self.reset_camera()

        items = self.doc.items()
        self.active: ItemRef = next((ref for ref in items if ref[0] == "modes"), items[0])
        self._refresh()
        if created:
            self._flash(f"new mode set - nothing saved yet; s writes {self.doc.path}")
        else:
            ms = self.doc.modeset
            self._flash(f"loaded {self.doc.path.name}: {len(ms.axes)} axes, "
                        f"{len(ms.constraint_sets)} constraint sets, {len(ms.modes)} modes")
        self.viewer.initialize()
        self.viewer.render()

    def _color_modes(self):
        fields = self.doc.mesh.point_fields
        modes, first = [], None
        if "heatmap_landmarks" in fields:
            self.view.add_point_field("heatmap_landmarks", fields["heatmap_landmarks"])
            modes.append(ScalarColoring("landmarks", "heatmap_landmarks", (0.0, 1.0),
                                        "landmark\nheatmap", SEQUENTIAL, label="landmark heatmap"))
        modes.append(FlatColoring("uniform", "uniform"))
        if "curvature" in fields:
            curvature = np.asarray(fields["curvature"], float)
            self.view.add_point_field("curvature", curvature)
            modes.append(ScalarColoring("curvature", "curvature", symmetric_range(curvature),
                                        "curvature", DIVERGING))
        for name, values in sorted(fields.items()):
            values = np.asarray(values)
            if (values.ndim != 1 or name == "curvature" or not np.issubdtype(values.dtype, np.floating)
                    or name.startswith(_SKIP_COLOR_FIELDS)):
                continue
            self.view.add_point_field(name, values)
            low, high = float(values.min()), float(values.max())
            modes.append(ScalarColoring(name, name, (low, high if high > low else low + 1.0),
                                        name.replace("_", "\n", 1), SEQUENTIAL))
        self._color_modes_first = modes[0].name
        return modes

    def _build_keymap(self) -> KeyMap:
        keys = KeyMap()
        keys.bind("up", lambda: self.move_item(-1), "previous item")
        keys.bind("down", lambda: self.move_item(1), "next item")
        keys.bind("left", lambda: self.move_property(-1), "previous property")
        keys.bind("right", lambda: self.move_property(1), "next property")
        # [ ] on US layouts; , . too, since [ ] need AltGr on German and other layouts.
        keys.bind(("bracketleft", "comma"), lambda: self.step_property(-1), "decrease / previous choice")
        keys.bind(("bracketright", "period"), lambda: self.step_property(1), "increase / next choice")
        keys.bind(("return", "kp_enter"), self.rename, "rename")
        keys.bind("a", lambda: self.new_item("axes"), "new axis")
        keys.bind("c", lambda: self.new_item("constraint_sets"), "new constraint set")
        keys.bind("n", lambda: self.new_item("modes"), "new mode")
        keys.bind("delete", self.delete_item, "delete item")
        keys.bind("backspace", self.drop_last_step, "drop the last selection step")
        keys.bind(("plus", "equal", "kp_add"), lambda: self.scale_brush(1.25), "bigger brush")
        keys.bind(("minus", "kp_subtract"), lambda: self.scale_brush(0.8), "smaller brush")
        keys.bind("k", self.ask_landmarks, "select around landmarks")
        keys.bind("f", self.ask_field, "select by point field")
        keys.bind("m", self.mirror_selection, "mirror the selection")
        keys.bind("p", self.compile_active, "compile the selected mode")
        keys.bind("e", self.compile_all, "compile every stale mode")
        keys.bind("0", self.reset_amplitudes, "amplitudes to zero")
        keys.bind("h", self.cycle_coloring, "next colouring")
        keys.bind("l", self.toggle_landmarks, "landmark markers")
        keys.bind("b", self.toggle_midline, "midline plane")
        keys.bind("t", self.toggle_theme, "light / dark")
        keys.bind("u", self.undo, "undo")
        keys.bind("y", self.redo, "redo")
        keys.bind("s", self.save_now, "save")
        keys.bind("prior", lambda: self.panel.scroll_by(-self.panel.n_rows), "sliders up a page")
        keys.bind("next", lambda: self.panel.scroll_by(self.panel.n_rows), "sliders down a page")
        return keys

    def _layout(self, width: int, height: int) -> None:
        self.hud.layout(width, height)
        self.panel.layout(width, height)
        if hasattr(self, "active"):
            self._refresh_text()                   # which rows fit, and where text is cut, depend on the size

    # -- the side panel ---------------------------------------------------------------

    def _on_panel_click(self, key) -> None:
        kind = key[0]
        if kind == "select":
            self.select_item(key[1])
        elif kind == "new":
            self.new_item(key[1])
        elif kind == "focus":
            self.focus_property(key[1])
        elif kind == "step":
            self.step_property_at(key[1], key[2])
        elif kind == "drop":
            self.drop_steps(key[1], key[2])
        elif kind == "action":
            self._panel_actions()[key[1]]()
        self.viewer.render()

    def _on_panel_wheel(self, key, delta: int) -> None:
        if key[0] == "prop":
            self.step_property_at(key[1], delta)
            self.viewer.render()

    def _panel_actions(self) -> Dict[str, Callable[[], None]]:
        return {"rename": self.rename, "delete": self.delete_item, "compile": self.compile_active,
                "mirror": self.mirror_selection, "landmarks": self.ask_landmarks,
                "field": self.ask_field, "undo": self.undo, "redo": self.redo, "save": self.save_now}

    # -- navigation and editing -------------------------------------------------------

    def select_item(self, ref: ItemRef) -> None:
        self.active = ref
        self._refresh()

    def move_item(self, delta: int) -> None:
        items = self.doc.items()
        index = items.index(self.active) if self.active in items else 0
        self.select_item(items[(index + delta) % len(items)])

    def move_property(self, delta: int) -> None:
        self.inspector.move(delta)
        self._refresh_text()

    def step_property(self, delta: int) -> None:
        if not self.inspector.step(delta):
            self._flash("nothing to change here - Left/Right picks an editable property")

    def focus_property(self, index: int) -> None:
        if self.inspector.focus(index):
            self._refresh_text()

    def step_property_at(self, index: int, delta: int) -> None:
        """Focus property ``index`` and change it by ``delta`` - a click on its arrows, or a wheel turn."""
        if self.inspector.focus(index):
            self.inspector.step(delta)

    def new_item(self, group: str) -> None:
        noun = GROUP_NOUNS[group]
        initial = self.doc.modeset.unique_name(group, _NEW_STEMS[group])
        self._ask(f"new {noun} name", _NAME_CHARS, lambda text: self._create(group, text), initial)

    def _create(self, group: str, name: str) -> None:
        name = name.strip()
        if not name or self.doc.modeset.has(group, name):
            self._flash(f"'{name}' is empty or already taken")
            return
        ms = self.doc.modeset
        if group == "axes":
            rule = Between("aR", "aL") if {"aR", "aL"} <= self.doc.mesh.landmarks.keys() else AwayFrom(
                (sorted(self.doc.mesh.landmarks)[0],))
            item = Axis(name, rule)
        elif group == "constraint_sets":
            item = ConstraintSet(name, "anchor")
        else:
            item = Mode(name, "displacement", axis=ms.axes[0].name if ms.axes else "")
        self._commit(ms.with_item(item), f"new {GROUP_NOUNS[group]} {name}")
        self.select_item((group, name))
        if group != "axes":
            self._flash(f"new {GROUP_NOUNS[group]} {name} - Ctrl+click to brush its region")

    def rename(self) -> None:
        if self.active == SETTINGS:
            self._flash("the settings item cannot be renamed")
            return
        group, old = self.active
        self._ask(f"rename {old}", _NAME_CHARS, lambda text: self._rename(group, old, text.strip()), old)

    def _rename(self, group: str, old: str, new: str) -> None:
        if not new:
            self._flash("empty name - not renamed")
            return
        try:
            modeset = self.doc.modeset.renamed(group, old, new)
        except ValueError as exc:
            self._flash(str(exc))
            return
        if group == "modes":
            self._drop_preview(old)
        self._commit(modeset, f"renamed {old} -> {new}")
        self.select_item((group, new))

    def delete_item(self) -> None:
        if self.active == SETTINGS:
            self._flash("the settings item cannot be deleted")
            return
        items = self.doc.items()
        index = items.index(self.active)
        group, name = self.active
        if group == "modes":
            self._drop_preview(name)
        self._commit(self.doc.modeset.without(group, name), f"deleted {GROUP_NOUNS[group]} {name}")
        items = self.doc.items()
        self.select_item(items[min(index, len(items) - 1)])

    def undo(self) -> None:
        description = self.doc.undo()
        self._after_history(f"undid: {description}" if description else "nothing to undo")

    def redo(self) -> None:
        description = self.doc.redo()
        self._after_history(f"redid: {description}" if description else "nothing to redo")

    def _after_history(self, message: str) -> None:
        for name in list(self.preview.names):
            if not self.doc.modeset.has("modes", name):
                self._drop_preview(name)
        if self.active not in self.doc.items():
            self.active = self.doc.items()[0]
        self._refresh()
        self._flash(message)

    # -- regions ------------------------------------------------------------------------

    def _region_item(self) -> Optional[Tuple[object, str]]:
        """(item, attribute) of the selected item's region, or None (settings, axes)."""
        item = self.doc.get(self.active)
        if isinstance(item, ConstraintSet):
            return item, "selection"
        if isinstance(item, Mode):
            return item, "sites"
        return None

    def _append_step(self, op, description: str) -> None:
        target = self._region_item()
        if target is None:
            self._flash("select a constraint set or a mode to shape its region")
            return
        item, attribute = target
        updated = replace(item, **{attribute: getattr(item, attribute).plus(op)})
        self._commit(self.doc.modeset.with_item(updated), f"{item.name}: {description}")

    def _on_stroke(self, stroke, touched: int) -> None:
        item = self.doc.get(self.active)
        if isinstance(item, Axis):
            self._snap_landmark(item, stroke.seed)
            return
        if self._region_item() is None:
            self._flash("select an axis, a constraint set or a mode first")
            return
        self._append_step(BrushOp(stroke.seed, stroke.radius, stroke.subtract),
                          f"{'erase' if stroke.subtract else 'brush'} {touched} vertices")

    def _snap_landmark(self, axis: Axis, vertex: int) -> None:
        mesh = self.doc.mesh
        labels = sorted(mesh.landmarks)
        positions = np.array([mesh.landmark_position(label) for label in labels])
        label = labels[int(np.argmin(np.linalg.norm(positions - mesh.vertices[vertex], axis=1)))]
        base, tangential, reversed_ = split_rule(axis.rule)
        focused = self.inspector.focused.label if self.inspector.focused else ""
        if isinstance(base, Between):
            attribute = "start" if focused == "start" else "end"
            base = replace(base, **{attribute: label})
            description = f"{attribute} = {label}"
        elif isinstance(base, AwayFrom):
            center = tuple(c for c in base.center if c != label) if label in base.center else base.center + (label,)
            base = AwayFrom(center or (label,))
            description = f"center = {'+'.join(base.center)}"
        else:
            self._flash(f"nearest landmark is {label}, but rule '{base.describe()}' uses none - "
                        f"set rule to between or away_from")
            return
        self._commit(self.doc.modeset.with_item(replace(axis, rule=join_rule(base, tangential, reversed_))),
                     f"{axis.name}: {description} (snapped)")

    def ask_landmarks(self) -> None:
        if self._region_item() is None:
            self._flash("select a constraint set or a mode first")
            return
        self._ask("landmarks to select around (e.g. aL aR)", _LABEL_CHARS, self._add_landmarks)

    def _add_landmarks(self, text: str) -> None:
        labels = tuple(t for t in re.split(r"[\s,]+", text.strip()) if t)
        missing = [l for l in labels if l not in self.doc.mesh.landmarks]
        if not labels or missing:
            self._flash(f"unknown landmark(s) {missing or '(none given)'}; have "
                        f"{', '.join(sorted(self.doc.mesh.landmarks))}")
            return
        self._append_step(NearLandmarks(labels, self.brush.radius), f"near {','.join(labels)}")

    def ask_field(self) -> None:
        if self._region_item() is None:
            self._flash("select a constraint set or a mode first")
            return
        self._ask("point field rule (e.g. kinetic_weight>0.8)", _FIELD_CHARS, self._add_field)

    def _add_field(self, text: str) -> None:
        match = _FIELD_RULE.match(text)
        if match is None:
            self._flash(f"could not read {text!r} - write it as name>value or name<value")
            return
        name, sign, value = match.group(1), match.group(2), float(match.group(3))
        values = self.doc.mesh.point_fields.get(name)
        if values is None or np.asarray(values).ndim != 1:
            self._flash(f"no scalar point field {name!r}")
            return
        op = FieldThreshold(name, above=value) if sign == ">" else FieldThreshold(name, below=value)
        self._append_step(op, f"{name}{sign}{value:g}")

    def mirror_selection(self) -> None:
        try:
            self.doc.mesh.mirror
        except ValueError as exc:
            self._flash(f"cannot mirror: {exc}")
            return
        self._append_step(Mirror(), "mirror")

    def drop_last_step(self) -> None:
        target = self._region_item()
        if target is None or not getattr(target[0], target[1]).ops:
            self._flash("no selection step to drop")
            return
        item, attribute = target
        updated = replace(item, **{attribute: getattr(item, attribute).without_last()})
        self._commit(self.doc.modeset.with_item(updated), f"{item.name}: dropped last step")

    def drop_steps(self, start: int, stop: int) -> None:
        """Remove region steps ``start .. stop - 1`` of the selected item; the rest replay as before."""
        target = self._region_item()
        if target is None:
            return
        item, attribute = target
        selection = getattr(item, attribute)
        if not 0 <= start < stop <= len(selection.ops):
            return
        updated = replace(item, **{attribute: selection.without_steps(start, stop)})
        n = stop - start
        self._commit(self.doc.modeset.with_item(updated),
                     f"{item.name}: removed {n} region step{'s' * (n > 1)}")

    def scale_brush(self, factor: float) -> None:
        radius = self.brush.scale_radius(factor)
        reference, name = self.doc.reference_length
        self._flash(f"brush radius {radius:.3g} ({radius / reference:.0%} of {name})")

    # -- compiling and previewing ---------------------------------------------------------

    def compile_active(self) -> None:
        if self.active[0] != "modes":
            self._flash("select a mode to compile")
            return
        self._compile(self.active[1])

    def compile_all(self) -> None:
        stale = [m.name for m in self.doc.modeset.modes if self.doc.is_stale(m.name)]
        if not stale:
            self._flash("every mode is up to date")
            return
        for name in stale:
            if not self._compile(name):
                break

    def _compile(self, name: str) -> bool:
        self._flash(f"compiling {name} ...")
        self.viewer.render()
        try:
            entry = self.doc.compile(name)
        except (ValueError, KeyError) as exc:
            self._flash(f"{name}: {exc}")
            return False
        self.preview.set_field(name, entry.field)
        if self.active == ("modes", name):
            self.preview.set_active(name)
        peak = float(np.linalg.norm(entry.field, axis=1).max())
        message = (f"compiled {name}: {entry.kernel.n_rows} constraint rows, peak {peak:.3g}, "
                   f"{entry.seconds:.1f}s")
        for warning in entry.warnings:
            self._log(f"  [{name}] {warning}")
        if entry.warnings:
            more = f" (+{len(entry.warnings) - 1} more in the console)" if len(entry.warnings) > 1 else ""
            message += f"  |  {entry.warnings[0]}{more}"
        self._refresh()
        self._flash(message)
        return True

    def _drop_preview(self, name: str) -> None:
        if name in self.preview.names:
            self.preview.remove_field(name)
        self.doc.forget(name)

    def _on_amplitude(self, mode: int, final: bool) -> None:
        self.preview.update()
        if final:
            flips = self.preview.flipped_faces()
            note = f", {flips} triangles flipped - folding" if flips else ""
            self._flash(f"{self.preview.names[mode]} at {self.preview.alpha[mode] / self.preview.sigma[mode]:+.2f}"
                        f"{note}")
        self.viewer.render()

    def reset_amplitudes(self) -> None:
        self.preview.reset_amplitudes()
        self.panel.refresh()
        self._flash("amplitudes reset")

    # -- display --------------------------------------------------------------------------

    def cycle_coloring(self) -> None:
        self._flash(f"colour: {self.coloring.cycle().label}")

    def toggle_landmarks(self) -> None:
        self._show_landmarks = not self._show_landmarks
        self._update_landmarks()
        self._flash(f"landmarks {'shown' if self._show_landmarks else 'hidden'}")

    def toggle_midline(self) -> None:
        self.plane.set_visible(not self.plane.visible)
        self._flash(f"midline plane {'shown' if self.plane.visible else 'hidden'}")

    def toggle_theme(self) -> None:
        self.apply_theme(next_theme(self.theme.name))

    def apply_theme(self, name: str) -> None:
        self.theme = get_theme(name)
        for component in (self.viewer, self.coloring, self.hud, self.panel, self.side,
                          self.selection_overlay,
                          self.context_overlay, self.vectors, self.frames, self.plane,
                          self.landmark_markers, self.preview):
            component.apply_theme(self.theme)

    def reset_camera(self) -> None:
        mesh = self.doc.mesh
        if {"prn", "n", "pg"} <= mesh.landmarks.keys():
            self.viewer.look_along(mesh.centroid, mesh.landmark_position("prn") - mesh.centroid,
                                   mesh.landmark_position("n") - mesh.landmark_position("pg"),
                                   self.radius, distance=3.2, zoom=1.25)
        else:
            self.viewer.place_camera(mesh.centroid, self.radius, "+z", "+y")
        # Centre the mesh in the free band between the sliders and the side panel.
        self.viewer.renderer.GetActiveCamera().SetWindowCenter(_VIEW_SHIFT, 0.0)

    def save_now(self) -> None:
        path = self.save()
        self._flash(f"saved {path}")

    def save(self) -> Path:
        path = self.doc.save()
        self._saved = True
        self._log(f"saved {path}")
        return path

    # -- refresh -------------------------------------------------------------------------

    def _commit(self, modeset, description: str) -> None:
        if self.doc.edit(modeset, description):
            self._after_edit(description)

    def _after_edit(self, description: str) -> None:
        self._refresh()
        self._flash(description)

    def _refresh(self) -> None:
        """Bring every view in line with the document and the selected item."""
        if self.active not in self.doc.items():
            self.active = SETTINGS
        self.inspector.show(*self.properties.build(self.active))

        region, context = self._regions()
        self.selection.set(region)
        self.context.set(context)
        self._glyph_key = None                      # recompute glyph data for this item
        self._draw_glyphs()
        if (self.active[0] == "modes" and self.active[1] in self.preview.names
                and self.preview.names.index(self.active[1]) != self.preview.active):
            self.preview.set_active(self.active[1])
        self._update_landmarks()
        self._refresh_text()

    def _regions(self) -> Tuple[np.ndarray, np.ndarray]:
        """(selected item's region, the regions it is constrained by) - empty when not applicable."""
        empty = np.zeros(0, dtype=np.int64)
        target = self._region_item()
        if target is None:
            return empty, empty
        item, attribute = target
        try:
            region = self.doc.resolve(getattr(item, attribute))
            context = empty
            if isinstance(item, Mode) and item.constraints:
                context = np.unique(np.concatenate([
                    self.doc.resolve(self.doc.modeset.constraint_set(name).selection)
                    for name in item.constraints]))
            return region, context
        except (KeyError, ValueError) as exc:
            self._flash(f"{item.name}: {exc}")
            return empty, empty

    def _draw_glyphs(self) -> None:
        """Arrows / frames for the selected item, at the mesh's current (possibly deformed) shape."""
        if self._glyph_key != self.active:
            self._glyph_data = self._compute_glyphs()
            self._glyph_key = self.active
        shape = self.view.vertices
        if "vectors" in self._glyph_data:
            indices, vectors = self._glyph_data["vectors"]
            self.vectors.set_data(shape[indices], vectors)
        else:
            self.vectors.clear()
        if "frames" in self._glyph_data:
            indices, frames, magnitudes = self._glyph_data["frames"]
            self.frames.set_frames(shape[indices], frames, magnitudes)
        else:
            self.frames.clear()

    def _compute_glyphs(self) -> Dict[str, tuple]:
        mesh, ms = self.doc.mesh, self.doc.modeset
        item = self.doc.get(self.active)
        try:
            if isinstance(item, Axis):
                indices = self._axis_sample()
                return {"vectors": (indices, item.rule.evaluate(mesh, indices) * 0.8)}
            if isinstance(item, ConstraintSet) and item.kind == "slide":
                region = self.doc.resolve(item.selection)
                if len(region):
                    chosen = sample(mesh, region, ms.spacing * ms.sigma)
                    return {"vectors": (chosen, mesh.normals[chosen] * 0.5)}
            if isinstance(item, Mode) and item.sites.ops and item.axis:
                sites = mode_sites(ms, item, mesh, resolve=self.doc.resolve)
                if sites.vectors is not None:
                    return {"vectors": (sites.sites, sites.vectors)}
                return {"frames": (sites.sites, sites.frames, np.asarray(item.stretch))}
        except (KeyError, ValueError) as exc:
            self._flash(f"{item.name}: {exc}")
        return {}

    def _axis_sample(self) -> np.ndarray:
        """~150 vertices spread over the face, nearest the nose first - where axes are shown."""
        if not hasattr(self, "_axis_indices"):
            mesh = self.doc.mesh
            center = mesh.landmark_position("prn") if "prn" in mesh.landmarks else mesh.centroid
            order = np.argsort(np.linalg.norm(mesh.vertices - center, axis=1))
            near = order[: max(len(order) // 4, 1)]
            self._axis_indices = spread_subset(mesh.vertices, near, 150)
        return self._axis_indices

    def _follow_shape(self) -> None:
        """The preview moved the mesh: overlays, glyphs and markers follow it."""
        self.selection_overlay.refresh()
        self.context_overlay.refresh()
        self.landmark_markers.reposition()
        if hasattr(self, "inspector"):
            self._draw_glyphs()

    def _update_landmarks(self) -> None:
        markers = self.landmark_markers
        if not self._show_landmarks:
            markers.clear()
            return
        highlighted = set()
        item = self.doc.get(self.active)
        if isinstance(item, Axis):
            base = split_rule(item.rule)[0]
            highlighted = {getattr(base, "start", None), getattr(base, "end", None),
                           *getattr(base, "center", ())}
        elif isinstance(item, (ConstraintSet, Mode)):
            selection = item.selection if isinstance(item, ConstraintSet) else item.sites
            for op in selection.ops:
                highlighted.update(getattr(op, "labels", ()))
        for label, vertex in self.doc.mesh.landmarks.items():
            markers.place(label, vertex, "marker_current" if label in highlighted else "marker")

    def _refresh_text(self) -> None:
        self.side.set_rows(panel_rows(self.doc, self.active, self.inspector, self.side,
                                      len(self.selection)))
        if self.prompt.active:
            status = prompt_line(self.prompt.label, self.prompt.buffer)
        else:
            focused = self.inspector.focused.label if self.inspector.focused else None
            status = status_text(self.doc, self.active, self.brush.radius, self.coloring.active.label,
                                 focused)
        self.hud.set_text("status", status)

    def _flash(self, message: str) -> None:
        self.hud.set_text("message", message)
        self._refresh_text()

    # -- prompt ------------------------------------------------------------------------------

    def _ask(self, label: str, allowed, action: Callable[[str], None], initial: str = "") -> None:
        self._prompt_action = action
        self.prompt.open(label, allowed, initial)

    def _on_prompt(self, text: str) -> None:
        action, self._prompt_action = self._prompt_action, None
        if action is not None:
            action(text)
        self._refresh_text()
