"""The window every tool opens: renderer, render window, interactor, camera."""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np
import vtk

from src.GUIs.core.theme import Theme

AXIS_VECTORS = {
    "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
}


class Viewer:
    """One renderer in one window, driven by a trackball-camera interactor.

    Components receive the viewer rather than its parts, so the renderer,
    interactor and ``render()`` they need always come from the same window.
    The default trackball style keeps rotate/pan/zoom; tools add behaviour as
    interactor *observers* (see ``input.py``), never by replacing the style.
    """

    def __init__(self, title: str, size: Sequence[int], *, offscreen: bool = False):
        """Create the window (not yet shown - see :meth:`start`).

        Args:
            title: window title.
            size: (width, height) in pixels.
            offscreen: render without a visible window, for tests and exports.
        """
        self.renderer = vtk.vtkRenderer()

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetSize(*size)
        self.render_window.SetWindowName(title)
        self.render_window.AddRenderer(self.renderer)
        if offscreen:
            self.render_window.SetOffScreenRendering(1)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        self._resize_callbacks: List[Callable[[int, int], None]] = []
        self.interactor.AddObserver("ConfigureEvent", self._on_configure)

    @property
    def size(self) -> Tuple[int, int]:
        width, height = self.render_window.GetSize()
        return int(width), int(height)

    def on_resize(self, callback: Callable[[int, int], None]) -> None:
        """Call ``callback(width, height)`` whenever the window is resized."""
        self._resize_callbacks.append(callback)

    def relayout(self) -> None:
        """Run every resize callback for the current size - after building a layout."""
        width, height = self.size
        for callback in self._resize_callbacks:
            callback(width, height)

    def _on_configure(self, interactor, event) -> None:
        self.relayout()

    def place_camera(self, centroid: np.ndarray, radius: float,
                     camera_axis: str, view_up: str, distance: float = 3.0) -> None:
        """Look at ``centroid`` from ``distance`` bounding radii along ``camera_axis``.

        Args:
            centroid: (3,) point to look at.
            radius: bounding radius of what is shown.
            camera_axis: key of AXIS_VECTORS the camera sits along, e.g. "+z".
            view_up: key of AXIS_VECTORS that points up on screen.
            distance: camera distance, in units of ``radius``.
        """
        axis = np.asarray(AXIS_VECTORS[camera_axis])
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*centroid)
        camera.SetPosition(*(np.asarray(centroid) + axis * radius * distance))
        camera.SetViewUp(*AXIS_VECTORS[view_up])
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()

    def look_along(self, center: np.ndarray, direction: np.ndarray, up: np.ndarray,
                   radius: float, distance: float = 3.0, zoom: float = 1.0) -> None:
        """Look at ``center`` from ``distance`` radii along ``direction`` (e.g. the way a face
        faces), with ``up`` pointing up on screen - for meshes not aligned with the world axes."""
        direction = np.asarray(direction, float) / np.linalg.norm(direction)
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(np.asarray(center, float) + direction * radius * distance))
        camera.SetViewUp(*np.asarray(up, float))
        self.renderer.ResetCamera()
        camera.Zoom(zoom)
        self.renderer.ResetCameraClippingRange()

    def apply_theme(self, theme: Theme) -> None:
        self.renderer.SetBackground(*theme.background)

    def render(self) -> None:
        self.render_window.Render()

    def initialize(self) -> None:
        """Prepare the interactor - after every observer and widget is attached."""
        self.interactor.Initialize()

    def start(self) -> None:
        """Show the window and block until it is closed."""
        self.interactor.Start()
