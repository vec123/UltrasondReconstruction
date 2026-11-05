"""What a model stage's output is, recorded next to its files: a reconstruction of a target.

A model stage writes ``result.json`` into its output folder, one record per result
(a GPMM fit writes one per fit, a cohort apply one per member folder):

    {"<result name>": {
        "recon": "nicp_refined_vertices_map.vtp",       # relative to the folder, or absolute
        "target": "rigid_alignment.vtp",
        "target_normals": null,                         # the target's point array of normals; null: from its faces
        "recon_to_scan": {"scale": s, "rotation": R, "translation": t},   # x_scan = s R x + t; null if unknown
        "landmarks": "orig_langmarks.vtp",              # in scan mm; null if none
        "model": {"kind": "gpmm", "path": "..."}}}      # a ShapeModel spec (src/shared/shape_model.py)

So the evaluation reads any model's output the same way (the ``result`` layout of
src/evaluation/loader/layouts.py), and no reader has to know a run family's file
names. Which stage, config and case made the folder is in its manifest.json
(src/shared/core/pipeline/manifest.py).
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np

from src.shared.transforms.similarity import SimilarityTransform
from src.shared.vtk.transform import decompose_similarity_transform

RESULT_FILE = "result.json"


@dataclass
class Result:
    recon: str
    target: str
    target_normals: Optional[str] = None
    recon_to_scan: Optional[dict] = None
    landmarks: Optional[str] = None
    model: Optional[dict] = None

    def transform(self, stage="recon->scan") -> Optional[SimilarityTransform]:
        """recon_to_scan as a SimilarityTransform, or None."""
        if self.recon_to_scan is None:
            return None
        matrix = np.eye(4)
        matrix[:3, :3] = float(self.recon_to_scan["scale"]) * np.asarray(self.recon_to_scan["rotation"], dtype=np.float64)
        matrix[:3, 3] = np.asarray(self.recon_to_scan["translation"], dtype=np.float64)
        return SimilarityTransform(matrix, stage=stage)


def similarity_record(transform: Optional[SimilarityTransform]) -> Optional[dict]:
    """The recon_to_scan record of a SimilarityTransform (None stays None)."""
    if transform is None:
        return None
    s, R, t = decompose_similarity_transform(transform.matrix)
    return {"scale": float(s), "rotation": np.asarray(R).tolist(), "translation": np.asarray(t).tolist()}


def _relative(folder, path):
    if path is None:
        return None
    path = os.path.abspath(path)
    inside = os.path.commonpath([os.path.abspath(folder), path]) == os.path.abspath(folder)
    return os.path.relpath(path, folder) if inside else path


def write_results(folder, results: Dict[str, Result]):
    """Writes ``results`` as the folder's result.json; paths inside the folder are stored relative to it."""
    records = {}
    for name, result in results.items():
        record = asdict(result)
        for key in ("recon", "target", "landmarks"):
            record[key] = _relative(folder, record[key])
        records[name] = record
    with open(os.path.join(folder, RESULT_FILE), "w") as f:
        json.dump(records, f, indent=2)


def read_results(folder) -> Dict[str, Result]:
    """The records of the folder's result.json, their paths absolute."""
    with open(os.path.join(folder, RESULT_FILE)) as f:
        records = json.load(f)
    results = {}
    for name, record in records.items():
        for key in ("recon", "target", "landmarks"):
            if record.get(key) is not None:
                record[key] = os.path.normpath(os.path.join(folder, record[key]))
        results[name] = Result(**record)
    return results
