"""Reconstruction/target pairs, read from the output folder of the run that made them.

Each run family writes its reconstruction and its target under different file
names and in different frames. A :class:`PairLayout` knows the folder structure
of one family. Everything downstream sees only the :class:`CorrespondencePair`
it returns, so a new family is one new layout class (see ``layouts.py``).

    load_pair(case_dir)                                  layout found from its marker file
    load_pair(case_dir, "direct_ssm_fit", {"fit": "gamma_1000"})
    discover_cases(root, "direct_flow_apply")            every case folder below root
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from src.shared.correspondence.cloud import PointCloud
from src.shared.core.registry import TypedRegistry
from src.shared.transforms.similarity import SimilarityTransform

#: frames a distance can be given in: the reconstruction's own, or the scan's (mm)
FRAMES = ("recon", "surface")


@dataclass(frozen=True, eq=False)
class CorrespondencePair:
    """One reconstruction and the target it was fitted to, in the same frame (the recon frame)."""

    case_id: str
    layout: str
    case_dir: Path
    recon: PointCloud
    target: PointCloud
    recon_path: Path
    target_path: Path
    #: maps the recon frame onto the surface frame (the scan, in mm); None when the run kept nothing to recover it from
    recon_to_surface: Optional[SimilarityTransform] = None
    #: the case's landmarks as the source run names them; None when it used none
    landmarks_path: Optional[Path] = None

    def to_recon_frame(self, distance: float, frame: str) -> float:
        """``distance``, given in ``frame``, in recon-frame units.

        Exact for a similarity: it scales every distance by the same factor.
        """
        if frame not in FRAMES:
            raise ValueError(f"frame must be one of {FRAMES}, got {frame!r}")
        if frame == "recon":
            return float(distance)
        if self.recon_to_surface is None:
            raise ValueError(
                f"{self.case_id}: no recon-to-surface transform, so a distance in the "
                f"'surface' frame cannot be converted; give it in the 'recon' frame"
            )
        return float(distance) / self.recon_to_surface.scale

    def summary(self) -> str:
        scale = "none" if self.recon_to_surface is None else f"{self.recon_to_surface.scale:.6g}"
        return (
            f"{self.case_id} [{self.layout}]: recon {len(self.recon)} points, "
            f"target {len(self.target)} points, recon->surface scale {scale}"
        )


class PairLayout(ABC):
    """The folder structure of one run family."""

    #: registry key, set by @register_layout
    name = "?"
    #: file, relative to a case folder, that identifies a case folder of this layout
    marker: ClassVar[str] = ""

    @classmethod
    def is_case(cls, case_dir: Path) -> bool:
        return (Path(case_dir) / cls.marker).is_file()

    @abstractmethod
    def load(self, case_dir: Path, case_id: str) -> CorrespondencePair:
        """Reads the pair of one case folder."""


PAIR_LAYOUT = TypedRegistry("pair layout")
register_layout = PAIR_LAYOUT.register


def available_layouts() -> List[str]:
    return PAIR_LAYOUT.names()


def build_layout(name: str, kwargs: Optional[Dict[str, Any]] = None) -> PairLayout:
    return PAIR_LAYOUT.build(name, kwargs)


def detect_layout(case_dir) -> str:
    """The one registered layout whose marker file is in ``case_dir``."""
    found = [name for name in available_layouts() if PAIR_LAYOUT.get(name).is_case(case_dir)]
    if len(found) != 1:
        raise ValueError(
            f"{case_dir}: {len(found)} layouts match ({found}); exactly one must. "
            f"Markers: {({name: PAIR_LAYOUT.get(name).marker for name in available_layouts()})}"
        )
    return found[0]


def discover_cases(root, layout: str = "auto") -> List[Path]:
    """Every case folder below ``root`` that is a case of ``layout`` (of any layout for "auto")."""
    root = Path(root)
    names = available_layouts() if layout == "auto" else [PAIR_LAYOUT.get(layout).name]
    found = {marker.parent for name in names
             for marker in root.rglob(PAIR_LAYOUT.get(name).marker)
             if PAIR_LAYOUT.get(name).is_case(marker.parent)}
    return sorted(found)


def load_pair(case_dir, layout: str = "auto", kwargs: Optional[Dict[str, Any]] = None,
              root=None) -> CorrespondencePair:
    """The pair of one case folder.

    ``case_id`` is the folder's path relative to ``root`` when ``root`` is
    given, and its name otherwise. Folder names alone are not unique in every
    layout: each fused-scan case of direct_SSM_fit is called ``fused_scan_ssm``.
    """
    case_dir = Path(case_dir)
    name = detect_layout(case_dir) if layout == "auto" else layout
    case_id = case_dir.relative_to(root).as_posix() if root is not None else case_dir.name
    return build_layout(name, kwargs).load(case_dir, case_id)
