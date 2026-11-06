"""The output-folder layouts that a reconstruction/target pair can be read from.

``result`` reads the result.json that a model stage writes (src/shared/results.py),
so it reads any model's output. The other layouts read the folders of run
families written before result.json existed; a folder that has one is read
by ``result`` only.

To add a layout: subclass PairLayout, set ``marker``, write ``load``, and
register the class under a name. Nothing else changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np

from src.shared.correspondence.cloud import PointCloud
from src.evaluation.loader.pair_loader import (
    CorrespondencePair,
    PairLayout,
    register_layout,
)
from src.models.gpmm.outputs import PipelineOutputFiles
from src.models.lddmm.outputs import ApplyOutputFiles
from src.shared.results import RESULT_FILE, read_results
from src.shared.transforms.similarity import SimilarityTransform

_SSM_FILES = PipelineOutputFiles()
_FLOW_FILES = ApplyOutputFiles()


@register_layout("result")
@dataclass(frozen=True)
class ResultLayout(PairLayout):
    """A folder with a result.json (src/shared/results.py): the output of any model stage.

    ``result`` names the record when the folder holds more than one (a GPMM
    run writes one per fit, e.g. "map_1x" or "refined").
    """

    result: Optional[str] = None

    marker: ClassVar[str] = RESULT_FILE

    def load(self, case_dir: Path, case_id: str) -> CorrespondencePair:
        results = read_results(case_dir)
        if self.result is None and len(results) != 1:
            raise ValueError(f"{case_dir / self.marker} holds {sorted(results)}; name one with result:")
        if self.result is not None and self.result not in results:
            raise ValueError(f"{case_dir / self.marker} has no {self.result!r}; it holds {sorted(results)}")
        record = results[self.result] if self.result is not None else next(iter(results.values()))
        return CorrespondencePair(
            case_id=case_id,
            layout=self.name,
            case_dir=case_dir,
            recon=PointCloud.from_vtp(record.recon),
            target=PointCloud.from_vtp(record.target, normals_field=record.target_normals),
            recon_path=Path(record.recon),
            target_path=Path(record.target),
            recon_to_surface=record.transform(),
            landmarks_path=Path(record.landmarks) if record.landmarks else None,
        )


def _has_result(case_dir) -> bool:
    return (Path(case_dir) / RESULT_FILE).is_file()


@register_layout("direct_ssm_fit")
@dataclass(frozen=True)
class DirectSSMFitLayout(PairLayout):
    """A case folder of the SSM pipeline runner (src/models/gpmm/runners/pipeline.py).

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
    refined_key: ClassVar[str] = "refined"     # REFINED_KEY in src/models/gpmm/runners/pipeline.py

    @classmethod
    def is_case(cls, case_dir: Path) -> bool:
        return super().is_case(case_dir) and not _has_result(case_dir)

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
    """A member folder of a resnet_flow apply run (src/models/lddmm/resnet_flow/apply/runner.py).

    recon      world/deformed.vtp.
    target     the cohort member the flow was applied to (apply.json ->
               cohort_member.path), with its own normals. Not
               world/sample_points.vtp: that has the same points, but no
               normals and no fields.
    frame      the cohort's similarity.json (x_mm = s R x + t), one folder
               above the member's train/ or test/ folder. None if it is absent.
    landmarks  the ``landmarks`` entry of the training config, when it has one.
    """

    marker: ClassVar[str] = _FLOW_FILES.manifest
    recon_file: ClassVar[str] = f"world/{_FLOW_FILES.deformed}"
    similarity_file: ClassVar[str] = "similarity.json"
    target_normals_field: ClassVar[str] = "normal"

    @classmethod
    def is_case(cls, case_dir: Path) -> bool:
        return super().is_case(case_dir) and not _has_result(case_dir)

    def load(self, case_dir: Path, case_id: str) -> CorrespondencePair:
        with open(case_dir / self.marker) as f:
            manifest = json.load(f)
        member = manifest.get("cohort_member")
        if not member:
            raise ValueError(f"{case_dir / self.marker} has no cohort_member; only cohort applies are supported")

        recon_path = case_dir / self.recon_file
        target_path = Path(member["path"])

        similarity_path = target_path.parent.parent / self.similarity_file
        recon_to_surface = SimilarityTransform.from_json(similarity_path) if similarity_path.is_file() else None

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


@register_layout("direct_flow_eval")
@dataclass(frozen=True)
class DirectFlowEvalLayout(PairLayout):
    """The eval folder of a direct_flow apply member (src/evaluation/stages/recon_target_distance.py).

    recon      eval/recon_w_distance.vtp: world/deformed.vtp moved into the
               landmark frame (mm) by a similarity fitted on the heatmap
               landmarks.
    target     eval/target_w_distance.vtp, moved by the same similarity. Two kinds:
               a mesh (the fused scan, as scripts/studies/eval_cohort_apply.py
               scores it) uses its own face normals; a point cloud must be the
               cohort member, whose ``normal`` array an eval written before
               transform_polydata_matrix did not rotate (or scaled by s), so
               the member's own normals are rotated with the similarity that is
               recovered from the recon.
    frame      eval -> world (that similarity, inverted), then the cohort's
               similarity.json: the scan in mm.
    landmarks  the same as for the apply member.
    """

    marker: ClassVar[str] = "target_w_distance.vtp"
    recon_file: ClassVar[str] = "recon_w_distance.vtp"
    #: largest misfit, in eval units, of the similarity recovered from the recon
    tolerance: ClassVar[float] = 1e-3

    @classmethod
    def is_case(cls, case_dir: Path) -> bool:
        case_dir = Path(case_dir)
        # The parent is an apply member, written before or after result.json existed.
        return (case_dir / cls.marker).is_file() and (case_dir.parent / DirectFlowApplyLayout.marker).is_file()

    def load(self, case_dir: Path, case_id: str) -> CorrespondencePair:
        world = DirectFlowApplyLayout().load(case_dir.parent, case_id)
        recon_path = case_dir / self.recon_file
        target_path = case_dir / self.marker
        recon = PointCloud.from_vtp(recon_path)
        target = PointCloud.from_vtp(target_path)

        world_to_eval = SimilarityTransform.from_correspondences(
            world.recon.points, recon.points, stage="world->eval"
        )
        self._check_similar(case_dir, "recon", world_to_eval, world.recon.points, recon.points)

        if target.faces is None:
            self._check_similar(case_dir, "target", world_to_eval, world.target.points, target.points)
            target = PointCloud(
                target.points,
                normals=world.target.normals @ world_to_eval.rotation.T,
                fields={name: target.field(name) for name in target.field_names},
            )
        recon_to_surface = (None if world.recon_to_surface is None
                            else world.recon_to_surface.compose(world_to_eval.inverse()))

        return CorrespondencePair(
            case_id=case_id,
            layout=self.name,
            case_dir=case_dir,
            recon=recon,
            target=target,
            recon_path=recon_path,
            target_path=target_path,
            recon_to_surface=recon_to_surface,
            landmarks_path=world.landmarks_path,
        )

    def _check_similar(self, case_dir, side, transform, world_points, eval_points):
        """The eval points must be the world points under ``transform``."""
        if len(world_points) == len(eval_points):
            misfit = np.abs(transform.apply_points(world_points) - eval_points).max()
            if misfit <= self.tolerance:
                return
            detail = f"misfit {misfit:.3g}"
        else:
            detail = f"{len(eval_points)} eval points, {len(world_points)} world points"
        raise ValueError(
            f"{case_dir}: the eval {side} is not the world {side} under one similarity "
            f"({detail}); the eval and world files do not belong together"
        )
