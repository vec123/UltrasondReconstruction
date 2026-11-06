"""The GPMM config classes, one tree per stage (src/models/gpmm/registrations.py) or GUI (scripts/GUIs/).

Fields without a default are required in the YAML file; everything else may be
omitted and falls back to the value declared here.
"""

from __future__ import annotations

# NOTE: annotations stay in the typing.List / typing.Dict spelling rather than
# the PEP 585 list[...] / dict[...] one. The config client resolves them with
# typing.get_type_hints(), which cannot evaluate the builtin generics on
# Python 3.8 - the version of the torch environment this project runs in.

from .align_landmarks import AlignLandmarksConfig, AlignLandmarksOutputs  # noqa: F401
from .align_meshes import AlignMeshesConfig, AlignMeshesOutputs, FocusConfig, MeshSource  # noqa: F401
from .mean_shape import MeanShapeConfig, MeanShapeFields, MeanShapeOutputFiles, MeanShapeOutputs  # noqa: F401
from .model import AtlasConfig, GpmmModelConfig, GpmmModelOutputConfig, ModelConfig  # noqa: F401
from .nicp import (  # noqa: F401
    ComponentSpec,
    CorrespondenceSpec,
    MatchingConfig,
    ModelOptimizationConfig,
    NicpConfig,
    NicpFitSpec,
    NicpVariant,
    _format_gamma,
    deep_merge,
    resolve_fit_specs,
)
from .nicp_fit import (  # noqa: F401
    NicpComputeConfig,
    NicpFitConfig,
    NicpInputs,
    NicpOutputFiles,
    NicpOutputs,
    ToModelSpace,
)
from .pipeline import (  # noqa: F401
    GmmToTemplateConfig,
    LaplacianConfig,
    PipelineConfig,
    PipelineInputs,
    PipelineOutputFiles,
    PipelineOutputs,
    RegionConfig,
    RigidConfig,
    SpectralConfig,
    StepsConfig,
)
from .place_landmarks import DEFAULT_LANDMARK_NAMES, PlaceLandmarksConfig  # noqa: F401
from .sample_shapes import (  # noqa: F401
    DropdownConfig,
    ModelLibraryConfig,
    SampleShapesConfig,
    SampleShapesExport,
    SampleShapesView,
    ShapeModelSource,
    SlidersConfig,
)
from .ssm_posterior import NoiseModelConfig, SsmPosteriorConfig  # noqa: F401
