"""The widgets of the shape-space explorer: sliders, scroll bar, buttons, labels, and the theme."""

from __future__ import annotations

import vtk

from src.models.gpmm.runners.sample_shapes.style import (
    _COLOR_LABELS,
    _PANEL_X,
    _PANEL_Y,
    _SCROLLBAR_X,
    _THEMES,
)

class WidgetsMixin:
    """The widgets and the theme of runner.SampleShapesRunner, a mixin: they use its state."""

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
        self.renderer.AddViewProp(self.scroll_track)
        self.renderer.AddViewProp(self.scroll_thumb)

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
            self.renderer.AddViewProp(label)
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
        self.renderer.AddViewProp(actor)
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
