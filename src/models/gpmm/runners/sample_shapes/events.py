"""The events of the shape-space explorer: scrolling the slider panel, mouse, keys and resizing."""

from __future__ import annotations

import time

import numpy as np

from src.models.gpmm.runners.sample_shapes.style import (
    _CONTENT_X,
    _DROPDOWN_BAR_H,
    _DROPDOWN_TOP,
    _OUR_KEYS,
    _PANEL_X,
    _PANEL_Y,
    _SCROLLBAR_X,
)

class EventsMixin:
    """Scrolling and the event handlers of runner.SampleShapesRunner, a mixin: they use its state."""

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
