"""Runner classes: each one takes a config object and executes one script.

The sim_tf, upsample_atlas and vtu_2_vtp runners are gone: they were never
migrated out of ``src.models.gpmm.runners.old``, and that folder was removed. Recover
them from the git history if they are needed again.
"""

from src.models.gpmm.runners.align_landmarks import AlignLandmarksResult, AlignLandmarksRunner
from src.models.gpmm.runners.align_meshes import AlignMeshesResult, AlignMeshesRunner
from src.models.gpmm.runners.mean_shape import MeanShapeResult, MeanShapeRunner
from src.models.gpmm.runners.nicp import NicpResult, NicpRunner, RunResult
from src.models.gpmm.runners.pipeline import PipelineResult, PipelineRunner
from src.models.gpmm.runners.sample_shapes import SampleShapesResult, SampleShapesRunner

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
    "SampleShapesRunner",
    "SampleShapesResult",
]
