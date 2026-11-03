"""User-specified deformation modes: the geometry and fields behind them, free of any GUI.

Shared by the deformation-mode editor (``src/GUIs/deformation_modes``) and,
later, by training, so that what the editor previews is exactly what the flow
gets. Nothing here imports VTK at module level.

    mesh.py          TemplateMesh - vertices, faces, point fields, plus derived quantities
                     (normals, landmark vertices, midline, mirror map, geodesic balls)
    landmarks.py     landmark vertices from heatmap_<label> point fields
    symmetry.py      MidlinePlane (fit from bilateral landmarks) and MirrorMap
    frames.py        orthonormal local frames (tangent, bitangent, normal) at points
    directions.py    direction rules: unit vectors per point from template landmarks
    kernel_field.py  KernelField - the smoothest field meeting displacement, anchor,
                     slide, strain and rigid constraints
    spec.py          ModeSet / Axis / ConstraintSet / Mode / Selection recipes (immutable)
    io.py            YAML read / write and validation of a ModeSet
    compile.py       a Mode -> its constraints -> its KernelField and field on the mesh
    transport.py     modes carried onto a posed (similarity) copy of the template, for training
"""

from src.shared.deformation_modes.geometry.directions import (
    AwayFrom, AwayFromMidline, Between, DirectionRule, Fixed, MidlineNormal, Reversed,
    SurfaceNormal, Tangential,
)
from src.shared.deformation_modes.geometry.frames import FRAME_AXES, local_frames
from src.shared.deformation_modes.field.kernel_field import (
    KernelExpansion, KernelField, StrainConstraint, ValueConstraint, anchors, displacement,
    rigid, sliding, strain, thin,
)
from src.shared.deformation_modes.geometry.landmarks import landmark_vertices
from src.shared.deformation_modes.geometry.mesh import TemplateMesh, load_template_mesh
from src.shared.deformation_modes.geometry.symmetry import MidlinePlane, MirrorMap
from src.shared.deformation_modes.spec import (
    CONSTRAINT_KINDS, MODE_KINDS, Axis, Brush, ConstraintSet, FieldThreshold, Mirror, Mode,
    ModeSet, NearLandmarks, Selection, faces_fingerprint,
)
from src.shared.deformation_modes.io import SpecError, check_mesh, load_modeset, save_modeset
from src.shared.deformation_modes.field.compile import CompiledMode, compile_mode, mode_signature, mode_sites
from src.shared.deformation_modes.field.transport import Similarity, TransportedMode, fit_similarity, transport_modes

__all__ = [
    "TemplateMesh", "load_template_mesh", "landmark_vertices",
    "MidlinePlane", "MirrorMap",
    "FRAME_AXES", "local_frames",
    "DirectionRule", "Between", "SurfaceNormal", "MidlineNormal", "AwayFromMidline",
    "AwayFrom", "Fixed", "Tangential", "Reversed",
    "KernelField", "KernelExpansion", "ValueConstraint", "StrainConstraint",
    "displacement", "anchors", "sliding", "strain", "rigid", "thin",
    "ModeSet", "Axis", "ConstraintSet", "Mode", "Selection",
    "Brush", "NearLandmarks", "FieldThreshold", "Mirror", "CONSTRAINT_KINDS", "MODE_KINDS",
    "faces_fingerprint", "SpecError", "check_mesh", "load_modeset", "save_modeset",
    "CompiledMode", "compile_mode", "mode_signature", "mode_sites",
    "Similarity", "TransportedMode", "fit_similarity", "transport_modes",
]
