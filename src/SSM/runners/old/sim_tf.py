"""Runner that replays a saved similarity transform onto a set of meshes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config.schema import SimTransformConfig
from src.fitting.rigid import invert_similarity_transform
from src.geometry.vtk import load_vtp_mesh, save_vtp_mesh, transform_polydata


@dataclass
class SimTransformResult:
    R: np.ndarray
    t: np.ndarray
    s: float
    inverse: bool
    written_files: list = field(default_factory=list)


class SimTransformRunner:
    """Applies a stored (R, t, s) - or its inverse - to whole .vtp meshes.

    Meshes go through vtkTransformPolyDataFilter, so topology, point data and
    cell data survive; only the geometry moves.
    """

    def __init__(self, config: SimTransformConfig):
        self.config = config

    def run(self) -> SimTransformResult:
        cfg = self.config

        R, t, s = self.load_transform()
        if cfg.inverse:
            R, t, s = invert_similarity_transform(R, t, s)
            self._log("applying the INVERSE transform")
        self._log(f"scale={s:.6g}")

        paths = self.resolve_inputs()
        if not paths:
            raise FileNotFoundError(
                f"no meshes to transform (inputs.dir={cfg.inputs.dir}, "
                f"pattern={cfg.inputs.pattern!r})"
            )

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        suffix = cfg.suffix or ""
        written = []
        for path in paths:
            dest = cfg.output_dir / f"{path.stem}{suffix}{path.suffix}"
            if dest.resolve() == path.resolve():
                raise ValueError(
                    f"refusing to overwrite the input mesh {path}; "
                    f"set a suffix or a different output_dir"
                )
            polydata = transform_polydata(load_vtp_mesh(str(path)), R, t, s)
            save_vtp_mesh(polydata, str(dest))
            self._log(f"  {path.name} -> {dest.name}")
            written.append(dest)

        self._log(f"transformed {len(written)} mesh(es) into {cfg.output_dir}")
        return SimTransformResult(R=R, t=t, s=s, inverse=cfg.inverse, written_files=written)

    # -- steps -------------------------------------------------------------

    def load_transform(self):
        path = self.config.transform
        if not path.is_file():
            raise FileNotFoundError(f"transform does not exist: {path}")

        data = np.load(str(path), allow_pickle=False)
        keys = list(data.keys())

        if "R" not in keys:
            raise KeyError(f"{path} has no 'R' array (found {keys})")
        # pipeline.py writes 't', align_landmarks.py writes 'T'
        t_key = next((k for k in ("t", "T") if k in keys), None)
        if t_key is None:
            raise KeyError(f"{path} has no translation array 't'/'T' (found {keys})")

        R = np.asarray(data["R"], dtype=np.float64)
        t = np.asarray(data[t_key], dtype=np.float64).flatten()
        s = float(np.asarray(data["s"]).flatten()[0]) if "s" in keys else 1.0

        if R.shape != (3, 3) or t.shape != (3,):
            raise ValueError(f"{path}: expected R (3,3) and t (3,), got {R.shape} and {t.shape}")

        self._log(f"loaded transform from {path}")
        if "direction" in keys:
            self._log(f"  recorded direction: {data['direction']}")
        return R, t, s

    def resolve_inputs(self) -> list:
        inputs = self.config.inputs

        if inputs.files:
            missing = [p for p in inputs.files if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"input mesh(es) not found: {missing}")
            return list(inputs.files)

        if not inputs.dir.is_dir():
            raise NotADirectoryError(f"inputs.dir does not exist: {inputs.dir}")

        excluded = set(inputs.exclude)
        paths = []
        for path in sorted(inputs.dir.glob(inputs.pattern)):
            if path.name in excluded:
                self._log(f"  skipping {path.name} (excluded)")
                continue
            # don't pick up the results of an earlier run
            if self.config.suffix and path.stem.endswith(self.config.suffix):
                self._log(f"  skipping {path.name} (already carries the suffix)")
                continue
            paths.append(path)
        return paths

    def _log(self, *args):
        if self.config.verbose:
            print(*args)
