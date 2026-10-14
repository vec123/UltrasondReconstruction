"""A mesh drawn from live numpy buffers, so moving it never rebuilds VTK objects."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import vtk
from vtk.util import numpy_support

from src.vtk.create import create_polydata


class MeshView:
    """One mesh actor whose points and point fields are views onto numpy arrays.

    ``numpy_to_vtk(deep=False)`` makes the VTK arrays share memory with the
    numpy buffers held here, so updating the shape (or a field) is an in-place
    numpy write plus ``Modified()`` - cheap enough for every frame of a slider
    drag. The buffers must therefore never be rebound, only written into;
    :meth:`set_vertices` and :meth:`set_point_field` do exactly that.
    """

    def __init__(self, vertices: np.ndarray, faces: Optional[np.ndarray]):
        """Build the polydata, mapper and actor.

        Args:
            vertices: (N, 3) initial shape; copied into the live float32 buffer.
            faces: (F, 3) triangles, or None for a bare point cloud. With faces,
                point normals are computed on the fly so the surface shades
                smoothly as it moves.
        """
        self.vertices = np.array(vertices, dtype=np.float32, order="C")
        self.faces = faces
        self.polydata = create_polydata(self.vertices, faces)
        self._points = numpy_support.numpy_to_vtk(self.vertices, deep=False)
        self.polydata.GetPoints().SetData(self._points)
        self._fields: Dict[str, Tuple[np.ndarray, vtk.vtkDataArray]] = {}

        self.mapper = vtk.vtkPolyDataMapper()
        if faces is not None:
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(self.polydata)
            normals.ComputePointNormalsOn()
            normals.ComputeCellNormalsOff()
            normals.SplittingOff()
            normals.ConsistencyOn()
            self.mapper.SetInputConnection(normals.GetOutputPort())
        else:
            self.mapper.SetInputData(self.polydata)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetSpecular(0.15)
        self.actor.GetProperty().SetSpecularPower(20)

        self.representation = "surface"

    # -- geometry --------------------------------------------------------------

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    def set_vertices(self, source: np.ndarray) -> None:
        """Write a new (N, 3) shape into the live buffer."""
        np.copyto(self.vertices, source, casting="unsafe")
        self._points.Modified()
        self.polydata.Modified()

    # -- point fields ---------------------------------------------------------

    def add_point_field(self, name: str, values: Optional[np.ndarray] = None) -> np.ndarray:
        """Attach a live scalar point field and return its buffer.

        Args:
            name: array name, as colouring modes select it.
            values: (N,) initial values; zeros when None.

        Returns:
            The (N,) float32 buffer - write into it, then :meth:`field_modified`,
            or use :meth:`set_point_field`.

        Raises:
            ValueError: a field of this name exists, or values has the wrong length.
        """
        if name in self._fields:
            raise ValueError(f"point field {name!r} is already attached")
        buffer = np.zeros(self.n_vertices, dtype=np.float32)
        if values is not None:
            values = np.asarray(values).reshape(-1)
            if len(values) != self.n_vertices:
                raise ValueError(f"point field {name!r} has {len(values)} values "
                                 f"for {self.n_vertices} vertices")
            buffer[:] = values
        array = numpy_support.numpy_to_vtk(buffer, deep=False)
        array.SetName(name)
        self.polydata.GetPointData().AddArray(array)
        self._fields[name] = (buffer, array)
        return buffer

    def has_point_field(self, name: str) -> bool:
        return name in self._fields

    def point_field(self, name: str) -> np.ndarray:
        """The live buffer of field ``name``."""
        return self._fields[name][0]

    def set_point_field(self, name: str, values) -> None:
        """Overwrite field ``name`` in place (a scalar broadcasts)."""
        self._fields[name][0][:] = values
        self.field_modified(name)

    def field_modified(self, name: str) -> None:
        """Tell VTK the buffer of field ``name`` was written to directly."""
        self._fields[name][1].Modified()
        self.polydata.Modified()

    # -- display ---------------------------------------------------------------

    def set_representation(self, representation: str, point_size: float = 4.0) -> None:
        """'surface' (filled) or 'points' (raw vertices, which show through folds)."""
        prop = self.actor.GetProperty()
        if representation == "points":
            prop.SetRepresentationToPoints()
            prop.SetPointSize(point_size)
        elif representation == "surface":
            prop.SetRepresentationToSurface()
        else:
            raise ValueError(f"representation must be 'surface' or 'points', got {representation!r}")
        self.representation = representation

    def toggle_representation(self, point_size: float = 4.0) -> str:
        """Switch surface <-> points; returns the new representation."""
        self.set_representation("points" if self.representation == "surface" else "surface",
                                point_size)
        return self.representation
