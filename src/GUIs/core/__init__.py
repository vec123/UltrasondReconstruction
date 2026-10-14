"""Building blocks shared by every interactive window.

Nothing here knows about shape models, landmarks or config files: each piece
takes plain arrays and callbacks, so a new tool is composed from them rather
than copied from an existing one. (The geometry behind deformation modes -
frames, direction rules, the midline, generated fields - is in
``src.deformation_modes``, which training uses too; the pieces here only draw
and edit its results.)

    Viewer            window, renderer, interactor, camera
    Theme / THEMES    colour palettes addressed by role
    MeshView          a mesh drawn from live numpy buffers (shape + point fields)
    SurfaceColoring   switchable colour modes (FlatColoring, ScalarColoring) + scalar bar
    Hud / TextSlot    screen-space text laid out by window edge
    VertexMarkers     named sphere markers pinned to mesh vertices
    ModeSliderPanel   scrolling sliders over a model's modes or a preview's fields
    VertexPicker      click -> vertex index
    KeyMap / KeyboardRouter / TextPrompt   keyboard actions and modal text entry
    VertexSelection / BrushTool / SelectionOverlay   region selection (Ctrl+click brush)
    VectorGlyphs / FrameGlyphs / PlaneGlyph          arrows, local frames, reference planes
    FieldPreview      live field preview: deform by amplitude, magnitude colouring, arrows
    Inspector / Property   keyboard-edited property list for whatever item is selected
    ListPanel / Row / Cell clickable text rows: buttons, steppers, hover, wheel
"""

from src.GUIs.core.coloring import (
    DIVERGING, SEQUENTIAL, ColorMode, FlatColoring, ScalarColoring, SurfaceColoring,
    make_scalar_bar, symmetric_range,
)
from src.GUIs.core.glyphs import FrameGlyphs, PlaneGlyph, VectorGlyphs
from src.GUIs.core.hud import Hud, TextSlot, make_text_actor
from src.GUIs.core.inspector import Inspector, Property
from src.GUIs.core.input import KeyBinding, KeyboardRouter, KeyMap, TextPrompt, VertexPicker
from src.GUIs.core.list_panel import BLANK, Cell, ListPanel, Row
from src.GUIs.core.markers import Marker, VertexMarkers
from src.GUIs.core.mesh_view import MeshView
from src.GUIs.core.preview import FieldPreview, spread_subset
from src.GUIs.core.selection import BrushTool, SelectionOverlay, Stroke, VertexSelection
from src.GUIs.core.slider_panel import ModeSliderPanel, PanelGeometry, SliderSource, Throttle
from src.GUIs.core.theme import THEMES, Theme, get_theme, next_theme, rgb
from src.GUIs.core.viewer import AXIS_VECTORS, Viewer

__all__ = [
    "AXIS_VECTORS", "Viewer",
    "THEMES", "Theme", "get_theme", "next_theme", "rgb",
    "MeshView",
    "ColorMode", "FlatColoring", "ScalarColoring", "SurfaceColoring",
    "DIVERGING", "SEQUENTIAL", "make_scalar_bar", "symmetric_range",
    "Hud", "TextSlot", "make_text_actor",
    "Marker", "VertexMarkers",
    "ModeSliderPanel", "PanelGeometry", "SliderSource", "Throttle",
    "KeyBinding", "KeyMap", "KeyboardRouter", "TextPrompt", "VertexPicker",
    "VertexSelection", "BrushTool", "SelectionOverlay", "Stroke",
    "VectorGlyphs", "FrameGlyphs", "PlaneGlyph",
    "FieldPreview", "spread_subset",
    "Inspector", "Property",
    "ListPanel", "Row", "Cell", "BLANK",
]
