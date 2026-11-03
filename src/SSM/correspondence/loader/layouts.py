"""The output-folder layouts that a reconstruction/target pair can be read from.

To add a layout: subclass PairLayout, set ``marker``, write ``load``, and
register the class under a name. Nothing else changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np

from src.SSM.correspondence.cloud import PointCloud
from src.SSM.correspondence.loader.pair_loader import (
    CorrespondencePair,
    PairLayout,
    register_layout,
)
from src.config.SSM.schema.pipeline import PipelineOutputFiles
from src.transforms.similarity import SimilarityTransform

_SSM_FILES = PipelineOutputFiles()


def _similarity_from_json(path: Path) -> SimilarityTransform:
    """A ``{scale, rotation, translation}`` file (x' = s R x + t), as scripts/SSM/target_to_ssm_frame.py writes it."""
    with open(path) as f:
        data = json.load(f)
    matrix = np.eye(4)
    matrix[:3, :3] = float(data["scale"]) * np.asarray(data["rotation"], dtype=np.float64)
    matrix[:3, 3] = np.asarray(data["translation"], dtype=np.float64)
    return SimilarityTransform(matrix, stage=str(path))


@register_layout("direct_ssm_fit")
@dataclass(frozen=True)
class DirectSSMFitLayout(PairLayout):
    """A case folder of the SSM pipeline runner (src/SSM/runners/pipeline.py).

    recon      the fit named ``fit``: nicp_refined_vertices_<fit>.vtp, or
               refined_vertices.vtp for "refined" (the landmark step's result).
    target     rigid_alignment.vtp, which is surface.vtp moved into the model
               frame. Not surface.vtp: that is still in scan mm.
    frame      the similarity fitted from new_target_landmarks.vtp (model
               frame) onto orig_langmarks.vtp (scan mm). The pipeline moved
               both with one transform, so the fit is exact.
    landmarks  orig_langmarks.vtp.
    """

    fit: str

    marker: ClassVar[str] = _SSM_FILES.rigid_alignment
    refined_key: ClassVar[str] = "refined"     # REFINED_KEY in src/SSM/runners/pipeline.py

    def _recon_file(self) -> str:
        if self.fit == self.refined_key:
            return _SSM_FILES.refined_vertices
        return _SSM_FILES.nicp_fit.format(name=self.fit)

    def load(self, case_dir: Path, case_id: str) -> CorrespondencePair:
        recon_path = case_dir / self._recon_file()
        target_path = case_dir / _SSM_FILES.rigid_alignment
        if not recon_path.is_file():
            fits = sorted(p.name for p in case_dir.glob(_SSM_FILES.nicp_fit.format(name="*"))
                          if "_iter_" not in p.name and not p.name.endswith("_colors.vtp"))
            raise FileNotFoundError(f"{recon_path} does not exist; fits in this folder: {fits}")

        surface_landmarks = case_dir / _SSM_FILES.original_landmarks
        recon_landmarks = case_dir / _SSM_FILES.transformed_landmarks
        recon_to_surface = None
        if surface_landmarks.is_file() and recon_landmarks.is_file():
            recon_to_surface = SimilarityTransform.from_correspondences(
                PointCloud.from_vtp(recon_landmarks).points,
                PointCloud.from_vtp(surface_landmarks).points,
                stage="landmarks recon->surface",
            )

        return CorrespondencePair(
            case_id=case_id,
            layout=self.name,
            case_dir=case_dir,
            recon=PointCloud.from_vtp(recon_path),
            target=PointCloud.from_vtp(target_path),
            recon_path=recon_path,
            target_path=target_path,
            recon_to_surface=recon_to_surface,
            landmarks_path=surface_landmarks if surface_landmarks.is_file() else None,
        )


@register_layout("direct_flow_apply")
@dataclass(frozen=True)
class DirectFlowApplyLayout(PairLayout):
    """A member folder of a resnet_flow apply run (src/resnet_lddmm/resnet_flow/apply/runner.py).

    recon      world/deformed.vtp.
    target     the cohort member the flow was applied to (apply.json ->
               cohort_member.path), with its own normals. Not
               world/sample_points.vtp: that has the same points, but no
               normals and no fields.
    frame      the cohort's similarity.json (x_mm = s R x + t), one folder
               above the member's train/ or test/ folder. None if it is absent.
    landmarks  the ``landmarks`` entry of the training config, when it has one.
    """

    marker: ClassVar[str] = "apply.json"
    recon_file: ClassVar[str] = "world/deformed.vtp"
    similarity_file: ClassVar[str] = "similarity.json"
    target_normals_field: ClassVar[str] = "normal"

    def load(self, case_dir: Path, case_id: str) -> CorrespondencePair:
        with open(case_dir / self.marker) as f:
            manifest = json.load(f)
        member = manifest.get("cohort_member")
        if not member:
            raise ValueError(f"{case_dir / self.marker} has no cohort_member; only cohort applies are supported")

        recon_path = case_dir / self.recon_file
        target_path = Path(member["path"])

        similarity_path = target_path.parent.parent / self.similarity_file
        recon_to_surface = _similarity_from_json(similarity_path) if similarity_path.is_file() else None

        landmarks_path: Optional[Path] = None
        train_config = manifest.get("train_config")
        if train_config and Path(train_config).is_file():
            with open(train_config) as f:
                landmarks = json.load(f).get("landmarks")
            landmarks_path = Path(landmarks) if landmarks else None

        return CorrespondencePair(
            case_id=case_id,
            layout=self.name,
            case_dir=case_dir,
            recon=PointCloud.from_vtp(recon_path),
            target=PointCloud.from_vtp(target_path, normals_field=self.target_normals_field),
            recon_path=recon_path,
            target_path=target_path,
            recon_to_surface=recon_to_surface,
            landmarks_path=landmarks_path,
        )
