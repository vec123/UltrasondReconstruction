"""Correspondences between two surfaces: the result type, the methods and their registry.

Importing the package registers every method, so ``build_correspondence`` can
build any of them by name.
"""

from src.SSM.correspondence.base import (
    CORRESPONDENCE,
    CorrespondenceError,
    Correspondences,
    CorrespondenceStrategy,
    available_correspondences,
    build_correspondence,
    register_correspondence,
)
from src.SSM.correspondence.methods import (
    ConstrainedMatching,
    NearestNeighbourMatching,
    NoCorrespondence,
    SymmetricMatching,
)

__all__ = [
    "CORRESPONDENCE",
    "ConstrainedMatching",
    "CorrespondenceError",
    "Correspondences",
    "CorrespondenceStrategy",
    "NearestNeighbourMatching",
    "NoCorrespondence",
    "SymmetricMatching",
    "available_correspondences",
    "build_correspondence",
    "register_correspondence",
]
