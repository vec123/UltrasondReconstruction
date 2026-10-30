"""Config for scripts/align_meshes.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------
# scripts/align_meshes.py
# --------------------------------------------------------------------------

@dataclass
class MeshSource:
    """A mesh, given as a file, as a model's mean shape, or as both.

    The two are complementary rather than alternatives:

    * ``mesh_vtp`` supplies the geometry *and everything attached to it* - the
      per-vertex fields written by scripts/mean_shape.py travel through the
      transform with the points, so the aligned pair keeps its heatmaps.
    * ``shape_model`` supplies the landmarks, which are what the alignment is
      sanity-checked against, and the mean shape when no mesh is given.

    Give both to get both. They are then required to agree vertex for vertex,
    since they are supposed to be the same shape.
    """

    mesh_vtp: Optional[Path] = None
    # .mat or Deformetrica atlas.
    shape_model: Optional[Path] = None
    normalize: bool = False          # shape_model only
    # Split every polygon into triangles on load. A mesh clipped in ParaView
    # comes back with mixed cell sizes, which normals and any downstream
    # surface method would rather not see. Point data is carried over.
    triangulate: bool = True

    def __post_init__(self):
        if self.mesh_vtp is None and self.shape_model is None:
            raise ValueError("set at least one of mesh_vtp or shape_model")


@dataclass
class AlignMeshesOutputs:
    dir: Path
    transform: str = "similarity_transform.npz"
    aligned: str = "aligned_source.vtp"
    target: str = "target.vtp"
    write_target: bool = True        # keep both meshes in one frame, in one folder

    def path(self, key: str) -> Path:
        return self.dir / getattr(self, key)


@dataclass
class FocusConfig:
    """Where the alignment should spend its accuracy.

    An unweighted ICP minimises the average distance over the whole surface,
    which is the right thing only if every part matters equally. It does not
    here: a fetal and a neonatal face differ everywhere, and spreading that
    difference evenly leaves the one landmark a nonrigid step most needs
    anchored - the nose - as wrong as the cheeks.

    Weighting the correspondences concentrates the fit without dropping the
    rest of the surface, so the transform stays well determined.
    """

    # Landmark of the *target* model to centre the weighting on.
    landmark: str = "prn"
    # Falloff of the weight away from it, in target units - a Gaussian, so
    # there is no hard edge for the ICP to chatter across.
    radius: float = 0.003
    # Weight at the centre relative to the far field. 1 disables the focus.
    weight: float = 8.0
    # Use a point-data array of the target as the weight instead, e.g. one of
    # mean_shape.py's 'heatmap_<landmark>' fields, which follow the surface
    # rather than a Euclidean radius. Overrides landmark/radius when set.
    field: Optional[str] = None


@dataclass
class AlignMeshesConfig:
    """Similarity alignment of two meshes with no correspondence between them.

    Solves ``x_target ~= s R x_source + t`` from geometry alone, as the initial
    guess for a nonrigid step - it will not remove a shape difference, only the
    placement, orientation and scale.
    """

    source: MeshSource               # the mesh that moves
    target: MeshSource               # the frame to move it into
    outputs: AlignMeshesOutputs
    allow_scale: bool = True         # similarity (True) vs rigid-only (False)
    # null weights every correspondence equally.
    focus: Optional[FocusConfig] = field(default_factory=FocusConfig)
    # Principal axes match in 24 proper rotations; all are screened and this
    # many are refined with ICP. Lowering it risks keeping a near-miss that
    # would have lost to a slower starter.
    refine_count: int = 8
    iterations: int = 100            # ICP iterations per candidate
    # Fraction of the worst-matched source points dropped each iteration. The
    # two surfaces do not cover the same region, so an unmatched rim would drag
    # the transform outward if every point had to be explained.
    trim_ratio: float = 0.1
    tolerance: float = 1.0e-12       # ICP convergence: change in mean distance
    verbose: bool = True
