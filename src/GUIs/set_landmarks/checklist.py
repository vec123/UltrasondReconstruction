"""The landmark checklist: which names exist, which are placed where, and the click waiting for a name.

Pure state - no VTK, no files - so the picker's rules (pick and name in
either order, Enter commits, re-opening a file never loses a name) live in one
place that can be tested without a window. The app turns its return values
into markers and status messages.
"""

from __future__ import annotations

import enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


class AddResult(enum.Enum):
    """What :meth:`LandmarkChecklist.add_name` did."""

    EMPTY = "empty"          # nothing usable was typed
    EXISTING = "existing"    # the name was already on the list; it is now current
    ADDED = "added"          # a new name was appended and made current


class LandmarkChecklist:
    """Names to place (in order), their vertex placements, the current name and the pending pick.

    Picking and naming are two separate actions linked by :meth:`commit_pending`:
    a click sets the *pending* vertex, moving the cursor sets the *current*
    name, and only a commit writes one under the other - so either order works.
    """

    def __init__(self, names: Sequence[str], n_vertices: int,
                 placements: Optional[Mapping[str, int]] = None):
        """Build a checklist.

        Args:
            names: checklist order; must be non-empty and unique.
            n_vertices: vertex count of the mesh; placements must index into it.
            placements: initial name -> vertex index; every name must be in ``names``.

        Raises:
            ValueError: empty or duplicate names, or an invalid placement.
        """
        if not names:
            raise ValueError("a landmark checklist needs at least one name")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate landmark names in {list(names)}")
        self.names: List[str] = list(names)
        self.n_vertices = int(n_vertices)
        self.placements: Dict[str, int] = {}
        for name, vertex in (placements or {}).items():
            if name not in self.names:
                raise ValueError(f"placement for {name!r}, which is not on the checklist")
            self.placements[name] = self._check_vertex(vertex)
        self.current_index = self._first_unplaced()
        self.pending_vertex: Optional[int] = None
        self.dirty = False

    @classmethod
    def from_saved(cls, configured_names: Sequence[str], saved: Mapping[str, int],
                   n_vertices: int) -> Tuple["LandmarkChecklist", List[Tuple[str, int]]]:
        """Merge a configured checklist with a previously saved placement file.

        Saved names missing from ``configured_names`` are appended after it, so
        re-opening a file never loses a name; saved placements that do not
        index into the mesh are dropped and reported rather than kept.

        Returns:
            (checklist, dropped) - ``dropped`` lists the (name, index) pairs left out.

        Raises:
            ValueError: no configured names and nothing saved to fall back on.
        """
        names = list(dict.fromkeys(configured_names))     # de-duplicated, order kept
        for name in saved:
            if name not in names:
                names.append(name)
        if not names:
            raise ValueError(
                "landmark_names is empty and landmarks_json has no entries to fall back on"
            )
        kept, dropped = {}, []
        for name, index in saved.items():
            if 0 <= index < n_vertices:
                kept[name] = int(index)
            else:
                dropped.append((name, index))
        return cls(names, n_vertices, kept), dropped

    # -- queries ---------------------------------------------------------------

    @property
    def current_name(self) -> str:
        return self.names[self.current_index]

    def is_placed(self, name: str) -> bool:
        return name in self.placements

    @property
    def n_placed(self) -> int:
        return len(self.placements)

    # -- the cursor ------------------------------------------------------------

    def select(self, index: int) -> Tuple[str, str]:
        """Make ``index`` (wrapped) the current name; returns (previous, new) current names."""
        previous = self.current_name
        self.current_index = int(index) % len(self.names)
        return previous, self.current_name

    def move(self, delta: int) -> Tuple[str, str]:
        """Move the cursor by ``delta`` names (wrapping); returns (previous, new)."""
        return self.select(self.current_index + delta)

    # -- the pending pick ------------------------------------------------------

    def set_pending(self, vertex: int) -> None:
        self.pending_vertex = self._check_vertex(vertex)

    def discard_pending(self) -> bool:
        """Drop the pending pick; False if there was none."""
        had_one = self.pending_vertex is not None
        self.pending_vertex = None
        return had_one

    def commit_pending(self) -> Optional[Tuple[str, int]]:
        """Write the pending vertex under the current name (overwriting).

        The cursor stays where it is: there is no canonical "next" landmark,
        so what to place next is left to the user.

        Returns:
            (name, vertex) placed, or None when nothing was pending.
        """
        if self.pending_vertex is None:
            return None
        name, vertex = self.current_name, self.pending_vertex
        self.placements[name] = vertex
        self.pending_vertex = None
        self.dirty = True
        return name, vertex

    # -- editing the list -------------------------------------------------------

    def clear_current(self) -> Optional[str]:
        """Remove the current name's placement; returns the name, or None if it had none."""
        name = self.current_name
        if name not in self.placements:
            return None
        del self.placements[name]
        self.dirty = True
        return name

    def add_name(self, raw: str) -> Tuple[AddResult, str]:
        """Add a typed name (spaces become underscores) and make it current.

        Returns:
            (result, name) - ``name`` is the cleaned name, "" for EMPTY.
        """
        name = raw.strip().replace(" ", "_")
        if not name:
            return AddResult.EMPTY, ""
        if name in self.names:
            self.select(self.names.index(name))
            return AddResult.EXISTING, name
        self.names.append(name)
        self.select(len(self.names) - 1)
        return AddResult.ADDED, name

    def mark_saved(self) -> None:
        self.dirty = False

    # -- helpers -----------------------------------------------------------------

    def _first_unplaced(self) -> int:
        for i, name in enumerate(self.names):
            if name not in self.placements:
                return i
        return 0

    def _check_vertex(self, vertex: int) -> int:
        vertex = int(vertex)
        if not 0 <= vertex < self.n_vertices:
            raise ValueError(f"vertex {vertex} is out of range for {self.n_vertices} vertices")
        return vertex
