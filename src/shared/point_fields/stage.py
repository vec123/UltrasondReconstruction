"""Stage: add point fields to one surface mesh.

    stage: point_fields
    input: ${FUSED_SCAN}
    output: ""                 # empty: in place
    fields:                    # applied in order
      - kind: normals
        weighting: unit
      - kind: edge
        combine: [support]

Each entry of ``fields`` is a kind of src/shared/point_fields/kinds.py with
its settings. All entries are checked before the mesh is read. Every array the
mesh has stays; a kind writes over an array of the same name. For many meshes,
give the YAML a ``targets:`` block (src/shared/core/pipeline/targets.py).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.shared.point_fields.kinds import KIND, build_field
from src.shared.vtk.io import load_vtp, save_vtp


@dataclass
class Config:
    input: str  # the mesh (.vtp)
    fields: List[Dict[str, Any]] = field(default_factory=list)  # {kind: <name>, <its settings>}, in order
    output: str = ""  # empty: write over input

    @property
    def output_path(self):
        return self.output or self.input


def run(cfg: Config):
    if not cfg.fields:
        raise ValueError("point_fields needs at least one entry under 'fields'.")
    kinds = [build_field(entry, cfg.input, "point_fields") for entry in cfg.fields]
    polydata = load_vtp(cfg.input)
    print(f"[point_fields] {cfg.input}: {polydata.GetNumberOfPoints()} points")
    for entry, kind in zip(cfg.fields, kinds):
        print(f"[point_fields]   {entry[KIND]}: {kind.apply(polydata)}")
    save_vtp(polydata, cfg.output_path)
    print(f"[point_fields] wrote {cfg.output_path}")
