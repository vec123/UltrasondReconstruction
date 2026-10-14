"""The editor's document: the current ModeSet, its undo history, and caches derived from it.

The ModeSet is immutable, so an edit is "replace the current value" and undo
is a stack of previous values - no command classes, nothing to get out of
sync. Resolved selections are cached by the (hashable) Selection itself, and
compiled fields by mode name together with the signature they were built
from, so staleness is a comparison rather than bookkeeping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.deformation_modes import (
    AwayFromMidline, Axis, Between, CompiledMode, ModeSet, Selection, SurfaceNormal,
    TemplateMesh, compile_mode, faces_fingerprint, load_modeset, load_template_mesh,
    mode_signature, save_modeset,
)

ItemRef = Tuple[str, str]                # (group, name); group in SETTINGS + ModeSet groups
SETTINGS: ItemRef = ("settings", "defaults")
GROUP_TITLES = {"settings": "SETTINGS", "axes": "AXES", "constraint_sets": "CONSTRAINT SETS",
                "modes": "MODES"}
GROUP_NOUNS = {"settings": "settings", "axes": "axis", "constraint_sets": "constraint set",
               "modes": "mode"}


def default_modeset(mesh: TemplateMesh, template: str, sigma: float) -> ModeSet:
    """A new, empty mode set with the axes almost every face mode needs."""
    axes = []
    if {"aR", "aL"} <= mesh.landmarks.keys():
        axes.append(Axis("lateral", Between("aR", "aL")))
    try:
        mesh.midline
        axes.append(Axis("away_from_midline", AwayFromMidline()))
    except ValueError:
        pass
    axes.append(Axis("normal", SurfaceNormal()))
    return ModeSet(template=template, n_vertices=mesh.n_vertices,
                   faces_sha1=faces_fingerprint(mesh.faces), sigma=float(sigma), axes=tuple(axes))


class EditorDocument:
    """One mode set being edited on one template mesh."""

    def __init__(self, modeset: ModeSet, mesh: TemplateMesh, path: Path, max_undo: int = 200):
        self.modeset = modeset
        self.mesh = mesh
        self.path = Path(path)
        self.dirty = False
        self._undo: List[Tuple[ModeSet, str]] = []
        self._redo: List[Tuple[ModeSet, str]] = []
        self._max_undo = int(max_undo)
        self._resolved: Dict[Selection, np.ndarray] = {}
        self._compiled: Dict[str, CompiledMode] = {}

    @classmethod
    def open(cls, template: Path, spec: Path, sigma_fraction: float) -> Tuple["EditorDocument", bool]:
        """Load the template and the spec (or start a new one); returns (document, created).

        Raises:
            SpecError: the spec exists but is invalid or was authored on another mesh.
        """
        mesh = load_template_mesh(template)
        if Path(spec).exists():
            return cls(load_modeset(spec, mesh), mesh, spec), False
        modeset = default_modeset(mesh, Path(template).name, sigma_fraction * mesh.radius)
        return cls(modeset, mesh, spec), True

    # -- editing ------------------------------------------------------------------

    def edit(self, modeset: ModeSet, description: str) -> bool:
        """Make ``modeset`` current (undoable); False if nothing changed."""
        if modeset == self.modeset:
            return False
        self._undo.append((self.modeset, description))
        del self._undo[:-self._max_undo]
        self._redo.clear()
        self.modeset = modeset
        self.dirty = True
        return True

    def undo(self) -> Optional[str]:
        """Step back; returns what was undone, or None."""
        if not self._undo:
            return None
        previous, description = self._undo.pop()
        self._redo.append((self.modeset, description))
        self.modeset, self.dirty = previous, True
        return description

    def redo(self) -> Optional[str]:
        if not self._redo:
            return None
        following, description = self._redo.pop()
        self._undo.append((self.modeset, description))
        self.modeset, self.dirty = following, True
        return description

    @property
    def history(self) -> List[str]:
        """What each undoable edit did, oldest first - the last one is what undo reverts."""
        return [description for _, description in self._undo]

    @property
    def redo_history(self) -> List[str]:
        """Undone edits that redo would bring back, next one last."""
        return [description for _, description in self._redo]

    # -- items ------------------------------------------------------------------------

    def items(self) -> List[ItemRef]:
        """Every editable item, in outline order: modes, constraint sets, axes, settings."""
        refs: List[ItemRef] = []
        for group in ("modes", "constraint_sets", "axes"):
            refs += [(group, item.name) for item in getattr(self.modeset, group)]
        return refs + [SETTINGS]

    def get(self, ref: ItemRef):
        return None if ref == SETTINGS else self.modeset.get(*ref)

    # -- derived data ------------------------------------------------------------------

    def resolve(self, selection: Selection) -> np.ndarray:
        """Vertex indices of ``selection`` (cached)."""
        if selection not in self._resolved:
            if len(self._resolved) > 256:
                self._resolved.clear()
            self._resolved[selection] = selection.resolve(self.mesh)
        return self._resolved[selection]

    def compiled(self, name: str) -> Optional[CompiledMode]:
        """The last field compiled for mode ``name`` - possibly stale, see :meth:`is_stale`."""
        return self._compiled.get(name)

    def is_stale(self, name: str) -> bool:
        entry = self._compiled.get(name)
        if entry is None or not self.modeset.has("modes", name):
            return True
        return entry.signature != mode_signature(self.modeset, self.modeset.mode(name))

    def compile(self, name: str) -> CompiledMode:
        """Solve mode ``name`` now (seconds, not milliseconds)."""
        entry = compile_mode(self.modeset, name, self.mesh, resolve=self.resolve)
        self._compiled[name] = entry
        return entry

    def forget(self, name: str) -> None:
        self._compiled.pop(name, None)

    @property
    def reference_length(self) -> Tuple[float, str]:
        """A length to express brush and kernel sizes in: outer-canthal width, else the radius."""
        if {"exL", "exR"} <= self.mesh.landmarks.keys():
            return float(np.linalg.norm(self.mesh.landmark_position("exL")
                                        - self.mesh.landmark_position("exR"))), "exL-exR"
        return self.mesh.radius, "R"

    # -- files ------------------------------------------------------------------------

    def save(self) -> Path:
        path = save_modeset(self.path, self.modeset)
        self.dirty = False
        return path
