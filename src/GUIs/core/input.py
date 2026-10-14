"""Mouse and keyboard input, layered on top of the default camera style.

Everything here is an interactor *observer* at a priority above the trackball
camera style's, never a replacement style: rotate/pan/zoom keep working
untouched, and an observer that handles an event sets the abort flag so the
style does not act on it too.

Keys arrive twice in VTK - a ``KeyPressEvent`` (carrying the key symbol) and
then a ``CharEvent`` (carrying the typed character). The style's own
single-letter bindings ('s' surface, 'w' wireframe, 'p' pick, 'q'/'e' quit,
...) run on ``CharEvent``, so consuming a key means aborting *both* events -
which is what :class:`KeyboardRouter` does for every key a :class:`KeyMap`
binds, and nothing else; unbound keys ('q' to quit, 'r' to reset the camera)
still reach the style.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Union

import vtk

from src.GUIs.core.viewer import Viewer

Log = Callable[..., None]


def _no_log(*args) -> None:
    pass


def _always() -> bool:
    return True


# -- mouse ----------------------------------------------------------------------


class VertexPicker:
    """Reports the mesh vertex under a left click.

    Picks on button-*down* rather than telling a click from a drag: an earlier
    version compared press and release positions, which made picking depend
    on both events firing and that bookkeeping being right - a real source of
    "clicking does nothing". Button-down is the same event the trackball
    camera starts a rotate from, so if the camera turns, picking fires. The
    cost is that a rotate-drag also reports a pick at its starting point;
    callers should make a pick provisional (e.g. needing Enter to commit).

    Two ways to find the vertex:

    * ``"point"`` (``vtkPointPicker``) - the vertex nearest the click *ray*,
      whatever its depth. Also hits raw point clouds and vertices on the far
      side, which is what the landmark picker's point view relies on.
    * ``"surface"`` (``vtkCellPicker``) - the front-most triangle under the
      click, then its vertex nearest the hit. Never reaches through the
      surface, which is what a region brush needs.
    """

    MODES = ("point", "surface")

    def __init__(self, viewer: Viewer, tolerance: float,
                 on_pick: Callable[[int], None],
                 on_miss: Callable[[int, int], None],
                 enabled: Callable[[], bool] = _always,
                 log: Log = _no_log, priority: float = 10.0, consume: bool = False,
                 mode: str = "point"):
        """Attach to ``viewer``'s interactor.

        Args:
            viewer: the window to pick in.
            tolerance: picker tolerance, a fraction of the window diagonal.
            on_pick: called with the picked vertex index.
            on_miss: called with the click's (x, y) when no vertex was under it.
            enabled: picking is skipped while this returns False (e.g. while
                a text prompt owns the keyboard, or unless a modifier is held).
            log: receives one diagnostic line per click.
            priority: observer priority, above the camera style's; of two
                pickers, the higher priority sees a click first.
            consume: stop a handled click from reaching lower-priority
                observers - the camera style, and any other picker - so a
                modifier-click does not also start a camera spin.
            mode: "point" or "surface" - see the class docstring.

        Raises:
            ValueError: unknown ``mode``.
        """
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.viewer = viewer
        self.on_pick = on_pick
        self.on_miss = on_miss
        self.enabled = enabled
        self.log = log
        self.consume = consume
        self.picker = vtk.vtkPointPicker() if mode == "point" else vtk.vtkCellPicker()
        self.picker.SetTolerance(tolerance)
        viewer.interactor.SetPicker(self.picker)
        self._tag = viewer.interactor.AddObserver("LeftButtonPressEvent", self._on_press, priority)

    def pick(self, x: int, y: int) -> Optional[int]:
        """The vertex under display position (x, y), or None."""
        hit = self.picker.Pick(x, y, 0, self.viewer.renderer)
        vertex = self.picker.GetPointId()
        self.log(f"  click at ({x}, {y}) -> hit={bool(hit)} vertex={vertex}")
        return int(vertex) if hit and vertex >= 0 else None

    def _on_press(self, interactor, event) -> None:
        if not self.enabled():
            return
        x, y = interactor.GetEventPosition()
        vertex = self.pick(x, y)
        if vertex is None:
            self.on_miss(x, y)
        else:
            self.on_pick(vertex)
        if self.consume:
            interactor.GetCommand(self._tag).SetAbortFlag(1)
        self.viewer.render()


# -- keyboard -------------------------------------------------------------------


@dataclass(frozen=True)
class KeyBinding:
    key: str                        # lower-case VTK key symbol, e.g. "s", "return", "prior"
    action: Callable[[], None]
    description: str = ""
    when: Callable[[], bool] = _always   # bound only while this holds


class KeyMap:
    """Key symbol -> action. A key is either bound (and consumed) or left to the camera style."""

    def __init__(self):
        self._bindings: Dict[str, KeyBinding] = {}

    def bind(self, keys: Union[str, Iterable[str]], action: Callable[[], None],
             description: str = "", when: Callable[[], bool] = _always) -> None:
        """Bind one key, or several keys to the same action.

        Raises:
            ValueError: a key is already bound.
        """
        for key in ([keys] if isinstance(keys, str) else keys):
            key = key.lower()
            if key in self._bindings:
                raise ValueError(f"key {key!r} is already bound to "
                                 f"{self._bindings[key].description or self._bindings[key].action}")
            self._bindings[key] = KeyBinding(key, action, description, when)

    def lookup(self, key: str) -> Optional[KeyBinding]:
        """The active binding for ``key`` (any case), or None."""
        binding = self._bindings.get(key.lower())
        if binding is None or not binding.when():
            return None
        return binding

    def __iter__(self):
        return iter(self._bindings.values())


class TextPrompt:
    """A one-line modal text field: while open it receives every keystroke.

    Holds no VTK state - :class:`KeyboardRouter` feeds it key symbols and
    characters - so it can be driven (and tested) without a window.
    """

    def __init__(self, on_confirm: Callable[[str], None], on_cancel: Callable[[], None],
                 on_change: Callable[[], None], max_length: int = 40,
                 allowed: Callable[[str], bool] = lambda c: c.isalnum() or c in "_- "):
        """Configure the prompt.

        Args:
            on_confirm: called with the typed text on Enter (prompt already closed).
            on_cancel: called on Escape (prompt already closed).
            on_change: called after every edit, e.g. to redraw the text.
            max_length: characters beyond this are ignored.
            allowed: which typed characters are accepted.
        """
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel
        self.on_change = on_change
        self.max_length = int(max_length)
        self.allowed = allowed
        self._default_allowed = allowed
        self.active = False
        self.buffer = ""
        self.label = ""

    def open(self, label: str = "", allowed: Optional[Callable[[str], bool]] = None,
             initial: str = "") -> None:
        """Start editing; ``label`` says what is being asked for, ``allowed`` overrides the
        accepted characters for this one entry, ``initial`` pre-fills the buffer."""
        self.active = True
        self.label = label
        self.allowed = allowed or self._default_allowed
        self.buffer = initial
        self.on_change()

    def handle_key(self, keysym: str) -> bool:
        """Editing keys: Enter confirms, Escape cancels, BackSpace deletes. True if handled."""
        if keysym in ("Return", "KP_Enter"):
            text, self.active, self.buffer = self.buffer, False, ""
            self.on_confirm(text)
        elif keysym == "Escape":
            self.active, self.buffer = False, ""
            self.on_cancel()
        elif keysym == "BackSpace":
            self.buffer = self.buffer[:-1]
            self.on_change()
        else:
            return False
        return True

    def handle_char(self, char: str) -> bool:
        """A typed character; control characters (handled by :meth:`handle_key`) are ignored."""
        if not char or ord(char) < 32 or len(self.buffer) >= self.max_length:
            return False
        if not self.allowed(char):
            return False
        self.buffer += char
        self.on_change()
        return True


class KeyboardRouter:
    """Sends each key to an open :class:`TextPrompt`, else to a :class:`KeyMap`.

    A key the map binds is consumed on both ``KeyPressEvent`` and the
    ``CharEvent`` that follows it, so the camera style never also acts on it
    (and the letter that *opened* a prompt is not typed into it). Any other
    key passes through to the style untouched.
    """

    def __init__(self, viewer: Viewer, keymap: KeyMap, prompt: Optional[TextPrompt] = None,
                 priority: float = 10.0):
        self.viewer = viewer
        self.keymap = keymap
        self.prompt = prompt
        self._swallow_char: Optional[str] = None
        interactor = viewer.interactor
        self._press_tag = interactor.AddObserver("KeyPressEvent", self._on_press, priority)
        self._char_tag = interactor.AddObserver("CharEvent", self._on_char, priority)

    def _abort(self, tag: int) -> None:
        self.viewer.interactor.GetCommand(tag).SetAbortFlag(1)

    def _on_press(self, interactor, event) -> None:
        keysym = interactor.GetKeySym() or ""
        self._swallow_char = None

        if self.prompt is not None and self.prompt.active:
            self._abort(self._press_tag)
            if self.prompt.handle_key(keysym):
                self.viewer.render()
            return

        binding = self.keymap.lookup(keysym)
        if binding is None:
            return
        self._abort(self._press_tag)
        # Only a single-character key symbol produces a CharEvent worth eating.
        self._swallow_char = keysym.lower() if len(keysym) == 1 else None
        binding.action()
        self.viewer.render()

    def _on_char(self, interactor, event) -> None:
        char = interactor.GetKeyCode() or ""
        swallow, self._swallow_char = self._swallow_char, None
        if swallow is not None and char.lower() == swallow:
            self._abort(self._char_tag)
            return

        if self.prompt is not None and self.prompt.active:
            self._abort(self._char_tag)
            if self.prompt.handle_char(char):
                self.viewer.render()
