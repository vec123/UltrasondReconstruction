"""The model dropdown of the shape-space explorer, and switching to another model."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from src.models.gpmm.runners.sample_shapes.style import _CONTENT_X, _DROPDOWN_BAR_H, _DROPDOWN_PAD, _DROPDOWN_TOP

class DropdownMixin:
    """The model dropdown of runner.SampleShapesRunner, a mixin: it uses its state."""

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
        self.renderer.AddViewProp(self.dropdown_bar)

        self.dropdown_label = self._text_actor(13, bold=True)
        self.dropdown_label.GetTextProperty().SetVerticalJustificationToCentered()

        rows = min(self.config.dropdown.visible_rows, len(self.model_names))
        for _ in range(rows):
            actor, points = self._quad(display=True)
            actor.VisibilityOff()
            self.renderer.AddViewProp(actor)
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
