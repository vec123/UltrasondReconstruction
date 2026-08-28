"""Runner classes: each one takes a config object and executes one script."""

from src.runners.align_landmarks import AlignLandmarksResult, AlignLandmarksRunner
from src.runners.align_meshes import AlignMeshesResult, AlignMeshesRunner
from src.runners.mean_shape import MeanShapeResult, MeanShapeRunner
from src.runners.nicp import NicpResult, NicpRunner, RunResult
from src.runners.pipeline import PipelineResult, PipelineRunner
from src.runners.place_landmarks import PlaceLandmarksResult, PlaceLandmarksRunner
from src.runners.sample_shapes import SampleShapesResult, SampleShapesRunner
from src.runners.sim_tf import SimTransformResult, SimTransformRunner
from src.runners.upsample_atlas import UpsampleAtlasResult, UpsampleAtlasRunner
from src.runners.vtu_2_vtp import VtuToVtpResult, VtuToVtpRunner

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
    "SimTransformRunner",
    "SimTransformResult",
    "UpsampleAtlasRunner",
    "UpsampleAtlasResult",
    "VtuToVtpRunner",
    "VtuToVtpResult",
]
