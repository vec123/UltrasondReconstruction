"""Runner classes: each one takes a config object and executes one script.

sim_tf, upsample_atlas and vtu_2_vtp are not re-exported here: those runners
only exist under ``src.SSM.runners.old`` (not yet migrated alongside the
others), so importing them here would break this package entirely. Import
them from ``src.SSM.runners.old.<module>`` directly if/when needed, or
migrate them here properly.
"""

from src.SSM.runners.align_landmarks import AlignLandmarksResult, AlignLandmarksRunner
from src.SSM.runners.align_meshes import AlignMeshesResult, AlignMeshesRunner
from src.SSM.runners.mean_shape import MeanShapeResult, MeanShapeRunner
from src.SSM.runners.nicp import NicpResult, NicpRunner, RunResult
from src.SSM.runners.pipeline import PipelineResult, PipelineRunner
from src.SSM.runners.place_landmarks import PlaceLandmarksResult, PlaceLandmarksRunner
from src.SSM.runners.sample_shapes import SampleShapesResult, SampleShapesRunner

__all__ = [
    "AlignLandmarksRunner",
    "AlignLandmarksResult",
    "AlignMeshesRunner",
    "AlignMeshesResult",
    "MeanShapeRunner",
    "MeanShapeResult",
    "NicpRunner",
    "NicpResult",
    "RunResult",
    "PipelineRunner",
    "PipelineResult",
    "PlaceLandmarksRunner",
    "PlaceLandmarksResult",
    "SampleShapesRunner",
    "SampleShapesResult",
]
