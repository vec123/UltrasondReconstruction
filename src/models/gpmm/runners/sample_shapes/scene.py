"""The surface of the shape-space explorer and its colour modes."""

from __future__ import annotations

from typing import Optional

import numpy as np
import vtk
from vtk.util import numpy_support

from src.models.gpmm.runners.sample_shapes.style import (
    _AXIS_VECTORS,
    _COLOR_ARRAY,
    _COLOR_BAR_TITLES,
    _COLOR_LABELS,
    _rgb,
)
from src.shared.vtk.create import create_polydata

class SceneMixin:
    """The scene and the colour modes of runner.SampleShapesRunner, a mixin: they use its state."""

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
            self.renderer.AddViewProp(self.scalar_bar)

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
        from src.shared.geometry.geometry import mesh_vertex_normals

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
