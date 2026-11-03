"""Reconstruction/target pairs from run output folders. Importing the package registers the built-in layouts."""

from src.SSM.correspondence.loader.pair_loader import (
    FRAMES,
    CorrespondencePair,
    PairLayout,
    available_layouts,
    build_layout,
    detect_layout,
    discover_cases,
    load_pair,
    register_layout,
)
from src.SSM.correspondence.loader import layouts  # noqa: F401  (registers the layouts)

__all__ = [
    "FRAMES",
    "CorrespondencePair",
    "PairLayout",
    "available_layouts",
    "build_layout",
    "detect_layout",
    "discover_cases",
    "load_pair",
    "register_layout",
]
