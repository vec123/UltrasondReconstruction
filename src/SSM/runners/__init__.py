"""Runner classes: each one takes a config object and executes one script.

The sim_tf, upsample_atlas and vtu_2_vtp runners are gone: they were never
migrated out of ``src.SSM.runners.old``, and that folder was removed. Recover
them from the git history if they are needed again.
"""

from src.SSM.runners.align_landmarks import AlignLandmarksResult, AlignLandmarksRunner
from src.SSM.runners.align_meshes import AlignMeshesResult, AlignMeshesRunner
from src.SSM.runners.mean_shape import MeanShapeResult, MeanShapeRunner
from src.SSM.runners.nicp import NicpResult, NicpRunner, RunResult
from src.SSM.runners.pipeline import PipelineResult, PipelineRunner
from src.SSM.runners.sample_shapes import SampleShapesResult, SampleShapesRunner
# The landmark picker moved to src.GUIs.set_landmarks; these names keep old imports working.
from src.GUIs.set_landmarks import SetLandmarksApp as PlaceLandmarksRunner
from src.GUIs.set_landmarks import SetLandmarksResult as PlaceLandmarksResult

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
