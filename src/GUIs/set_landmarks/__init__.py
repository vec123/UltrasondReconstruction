"""The landmark picker behind ``scripts/SSM/place_landmarks.py``.

    app.py          SetLandmarksApp - the window, composed from src.GUIs.core
    checklist.py    LandmarkChecklist - names, placements, cursor, pending pick (no VTK)
    shape_space.py  ShapeSpace - linear / LDDMM samples and real subjects (no VTK)
    formatting.py   the panel, status and hint text, as pure functions
"""

from src.GUIs.set_landmarks.app import SetLandmarksApp, SetLandmarksResult
from src.GUIs.set_landmarks.checklist import AddResult, LandmarkChecklist
from src.GUIs.set_landmarks.shape_space import LoadedModel, ShapeSpace, load_shape_space

__all__ = [
    "SetLandmarksApp", "SetLandmarksResult",
    "AddResult", "LandmarkChecklist",
    "LoadedModel", "ShapeSpace", "load_shape_space",
]
