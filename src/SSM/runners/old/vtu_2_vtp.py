"""Runner for the VTU -> VTP conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.schema import VtuToVtpConfig
from src.geometry.vtk import (
    convert_vtu_to_vtp_vtk,
    filter_largest_component,
    load_vtu,
    save_vtp_mesh,
)


@dataclass
class VtuToVtpResult:
    output_vtp: Path
    polydata: object


class VtuToVtpRunner:
    """Loads a .vtu volume mesh, converts it to a surface .vtp and saves it."""

    def __init__(self, config: VtuToVtpConfig):
        self.config = config

    def run(self) -> VtuToVtpResult:
        cfg = self.config

        if not cfg.input_vtu.is_file():
            raise FileNotFoundError(f"input_vtu does not exist: {cfg.input_vtu}")

        self._log("load")
        vtu = load_vtu(str(cfg.input_vtu))

        self._log("convert")
        vtp = convert_vtu_to_vtp_vtk(vtu)

        if cfg.filter_largest_component:
            self._log("filter largest component")
            vtp = filter_largest_component(vtp)

        self._log("save")
        cfg.output_vtp.parent.mkdir(parents=True, exist_ok=True)
        save_vtp_mesh(vtp, str(cfg.output_vtp))

        return VtuToVtpResult(output_vtp=cfg.output_vtp, polydata=vtp)

    def _log(self, *args):
        if self.config.verbose:
            print(*args)
