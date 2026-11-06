"""What the morphing pipeline runner returns, and the result.json it writes beside its fits."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.models.gpmm.model.model import ShapeInstance
from src.models.gpmm.nicp import Correspondences
from src.models.gpmm.outputs import PipelineOutputFiles
from src.shared.correspondence.cloud import PointCloud as LandmarkCloud
from src.shared.results import Result, similarity_record, write_results
from src.shared.transforms.similarity import SimilarityTransform

#: registry key for the step 1 result
REFINED_KEY = "refined"
_FILES = PipelineOutputFiles()


def write_fit_results(out_dir, keys, shape_model_path):
    """The result.json (src/shared/results.py) of a run's folder: one record per fit key whose file exists.

    recon      nicp_refined_vertices_<key>.vtp, or refined_vertices.vtp for "refined".
    target     rigid_alignment.vtp: surface.vtp moved into the model frame.
    frame      the similarity fitted from new_target_landmarks.vtp (model frame) onto
               orig_langmarks.vtp (scan mm). The run moved both with one transform, so the fit is exact.
    landmarks  orig_langmarks.vtp.
    """
    folder = Path(out_dir)
    target = folder / _FILES.rigid_alignment
    if not target.is_file():
        return
    surface_landmarks = folder / _FILES.original_landmarks
    recon_landmarks = folder / _FILES.transformed_landmarks
    transform = None
    if surface_landmarks.is_file() and recon_landmarks.is_file():
        transform = SimilarityTransform.from_correspondences(
            LandmarkCloud.from_vtp(recon_landmarks).points, LandmarkCloud.from_vtp(surface_landmarks).points,
            stage="landmarks recon->surface")
    results = {}
    for key in keys:
        recon = folder / (_FILES.refined_vertices if key == REFINED_KEY else _FILES.nicp_fit.format(name=key))
        if recon.is_file():
            results[key] = Result(str(recon), str(target), recon_to_scan=similarity_record(transform),
                                  landmarks=str(surface_landmarks) if surface_landmarks.is_file() else None,
                                  model={"kind": "gpmm", "path": str(shape_model_path)})
    if results:
        write_results(folder, results)


@dataclass
class PipelineResult:
    shape: ShapeInstance
    surface_vertices: np.ndarray
    aligned_surface: np.ndarray
    refined_vertices: np.ndarray
    transform: SimilarityTransform
    matches: Optional[Correspondences]
    template_vertices: Optional[np.ndarray] = None
    gmm_template_transform: Optional[SimilarityTransform] = None
    shapes: dict = field(default_factory=dict)      # key -> ShapeInstance
    vertices: dict = field(default_factory=dict)    # key -> (3N,) reconstruction
    deformed_vertices: Optional[np.ndarray] = None
    written_files: list = field(default_factory=list)
