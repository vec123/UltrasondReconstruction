"""The deformation-mode editor behind ``scripts/GUIs/edit_deformation_modes.py``.

    app.py         DeformationModesApp - the window, composed from src.GUIs.core
    document.py    EditorDocument - current ModeSet, undo / redo, resolve and compile caches
    properties.py  PropertyBuilder - inspector rows (and their edits) per item kind
    formatting.py  outline, status and hint text, as pure functions
    config.py      DeformationModesConfig

The spec being edited, and everything that turns it into fields, is
``src.deformation_modes`` - shared with training, independent of this GUI.
"""

from src.GUIs.deformation_modes.app import DeformationModesApp, DeformationModesResult
from src.GUIs.deformation_modes.config import DeformationModesConfig
from src.GUIs.deformation_modes.document import EditorDocument, default_modeset

__all__ = [
    "DeformationModesApp", "DeformationModesResult", "DeformationModesConfig",
    "EditorDocument", "default_modeset",
]
