"""Locating and reading the checkpoints CheckpointWriter leaves behind.

CheckpointWriter writes ``step_<N>.pt`` on its cadence and ``final.pt`` when the
run ends, into ``<output_dir>/checkpoints``. This module is the reader for that
layout: everything that knows those filenames lives here, so a change to the
naming is a change to one file.

It sits at package level rather than under ``apply/`` because BOTH entry points
read checkpoints -- apply, to reconstruct a finished model, and training, to
resume or warm-start from one (``init:`` in the experiment config). ``apply`` is
downstream of training, so training importing it from there would invert the
layering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

# CheckpointWriter's two names. final.pt is written on train end whatever step
# the run stopped at, so it is the one to prefer when asked for "the" checkpoint.
FINAL_NAME = "final.pt"
STEP_PATTERN = re.compile(r"^step_(\d+)\.pt$")


@dataclass(frozen=True)
class CheckpointRef:
    """Which .pt was chosen, and whether the config named it or picked it.

    ``pinned`` is the difference between "apply this RUN" and "apply THIS
    checkpoint", and it is the only thing that distinguishes them once the file
    has been found -- both end up at the same path when a run has only one. It
    decides whether the output directory is tagged: a pinned run is normally one
    of several being compared, so its results must not land on top of another
    step's.

    Attributes:
        path: the .pt file to read
        pinned: True when the configuration named this exact checkpoint, by step
            number or by file path, rather than leaving it to be resolved
    """

    path: Path
    pinned: bool

    @property
    def label(self) -> str:
        """A directory-safe name for this checkpoint: ``step_1300``, ``final``."""
        return self.path.stem

    @property
    def step(self) -> Optional[int]:
        """The step in the FILENAME, or None for final.pt and other names.

        Distinct from the step recorded INSIDE the file (see checkpoint_step):
        this one is available without reading it.
        """
        match = STEP_PATTERN.match(self.path.name)
        return int(match.group(1)) if match else None


def available_steps(directory) -> list:
    """The step numbers of every ``step_<N>.pt`` in a directory, ascending."""
    return sorted(int(m.group(1)) for p in Path(directory).iterdir()
                  for m in [STEP_PATTERN.match(p.name)] if m)


def resolve_checkpoint(path, step: Optional[int] = None) -> CheckpointRef:
    """A configured checkpoint location -> the .pt file to actually read.

    Three ways to say which one, in order of how specific they are:

      * ``step=N`` with a directory -> that directory's ``step_N.pt``. PINNED.
      * a .pt file path -> itself. PINNED.
      * a directory alone -> ``final.pt`` if present, else the highest-numbered
        ``step_N.pt``. NOT pinned: this is "whatever the run ended at", which is
        a different request and gives a different answer as the run progresses.

    Sorting is numeric, not lexicographic -- ``step_1000.pt`` sorts BEFORE
    ``step_200.pt`` as a string, so a plain max() over names would silently pick
    an early checkpoint on any run past a thousand steps.

    Args:
        path: a .pt file, or a directory containing checkpoints
        step: the exact training step to load, or None to resolve one

    Returns:
        CheckpointRef naming an existing file

    Raises:
        ValueError: if a step is given alongside a path that is already a file
        FileNotFoundError: if nothing matches
    """
    path = Path(path)

    if step is not None:
        if path.is_file():
            raise ValueError(
                f"step={step} was given alongside a checkpoint FILE ({path}), which "
                f"already names one. Point 'checkpoint' at the checkpoints/ directory, "
                f"or drop 'step'."
            )
        if not path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {path}")

        pinned = path / f"step_{step}.pt"
        if not pinned.is_file():
            available = available_steps(path)
            raise FileNotFoundError(
                f"No step_{step}.pt in {path}. "
                + (f"Available steps: {available}." if available
                   else f"Directory holds no step_<N>.pt at all.")
            )
        return CheckpointRef(pinned, pinned=True)

    if path.is_file():
        return CheckpointRef(path, pinned=True)

    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    final = path / FINAL_NAME
    if final.is_file():
        return CheckpointRef(final, pinned=False)

    available = available_steps(path)
    if not available:
        raise FileNotFoundError(
            f"No checkpoint in {path}: expected {FINAL_NAME} or step_<N>.pt "
            f"(is this a run's checkpoints/ directory?)"
        )
    return CheckpointRef(path / f"step_{available[-1]}.pt", pinned=False)


def load_checkpoint(ref, map_location="cpu") -> dict:
    """Read a checkpoint written by CheckpointWriter.

    Args:
        ref: a CheckpointRef, or anything resolve_checkpoint accepts
        map_location: passed to torch.load

    Returns:
        The checkpoint dict: {"step", "flow", "codes", "optimizer"}, plus "path"

    Raises:
        FileNotFoundError: if nothing readable is at the location
        ValueError: if the file holds no flow weights, i.e. it is not a
            checkpoint of this pipeline
    """
    if not isinstance(ref, CheckpointRef):
        ref = resolve_checkpoint(ref)

    state = torch.load(str(ref.path), map_location=map_location)

    if not isinstance(state, dict) or "flow" not in state:
        raise ValueError(
            f"{ref.path} is not a ResNetLDDMM checkpoint: expected a dict with a "
            f"'flow' entry, got {sorted(state) if isinstance(state, dict) else type(state).__name__}"
        )

    state["path"] = str(ref.path)
    return state


def checkpoint_step(state: dict) -> Optional[int]:
    """The training step a checkpoint was written at, or None if unrecorded."""
    step = state.get("step")
    return None if step is None else int(step)
