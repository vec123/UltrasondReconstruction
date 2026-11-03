"""The model source and build context, and the small readers for spec dictionaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from src.SSM.model.mean_domain import MeanDomain


# --------------------------------------------------------------------------
# reading a spec that may be a mapping or a config dataclass
# --------------------------------------------------------------------------

def _get(spec, key: str, default=None):
    """``spec[key]`` or ``spec.key``, whichever the object supports.

    The same accommodation ``build_kernel`` makes, and for the same reason: the
    builder must take both a plain dict written in a test and the config
    dataclass of section 13, without depending on ``src.SSM.config``. An explicit
    ``null`` in YAML reads as the default, which is what makes
    ``n_components: null`` mean "no cap" rather than an error.
    """
    if spec is None:
        return default
    value = spec.get(key, default) if hasattr(spec, "get") else getattr(spec, key, default)
    return default if value is None else value


def _component(spec, default_name: str) -> Tuple[str, Dict[str, Any]]:
    """``(name, kwargs)`` from ``"name"``, ``{"name": ..., "kwargs": ...}`` or None."""
    if spec is None:
        return default_name, {}
    if isinstance(spec, str):
        return spec, {}
    return str(_get(spec, "name", default_name)), dict(_get(spec, "kwargs", {}) or {})


def _as_float(value, what: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, got {value!r}")


# --------------------------------------------------------------------------
# the model source
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSource:
    """What a model file yields, whatever its format.

    An Adapter over :mod:`src.SSM.model.sources.load`, which already puts a ``.mat`` and
    a Deformetrica atlas behind one call. Holding the result as a value object
    matters because it is read three times during one build - for the mean, for
    the triangulation and for the ``pca`` basis - and both backends are
    expensive enough (an atlas shoots its population, ~90 s uncached) that
    loading per use would dominate the build.
    """

    path: Path
    mean: np.ndarray                    # (3n,)
    eigenfunctions: np.ndarray          # (3n, r)
    eigenvalues: np.ndarray             # (r,)
    landmarks: dict
    faces: Optional[np.ndarray]         # (F, 3) or None

    @classmethod
    def load(cls, path, normalize: bool = True, atlas_options=None,
             verbose: bool = False) -> "ModelSource":
        if Path(path).suffix.lower() == ".npz":
            # A built archive as the source of another build. Its (mean, Phi,
            # Lambda) give Phi Lambda Phi^T, the same covariance in either
            # inner product, so a ``pca`` term built from it is exact.
            from src.SSM.model.model_memory_io import load_model_file

            model = load_model_file(path)
            return cls(
                path=Path(path),
                mean=np.ascontiguousarray(model.mean.reshape(-1)),
                eigenfunctions=model.eigenfunctions,
                eigenvalues=model.eigenvalues.reshape(-1),
                landmarks=model.landmarks or {},
                faces=None if model.faces is None else np.ascontiguousarray(model.faces),
            )

        from src.SSM.model.sources.load import load_shape_model, load_shape_model_triangulation

        mean, eigenfunctions, eigenvalues, landmarks = load_shape_model(
            path, normalize=normalize, atlas_options=atlas_options, verbose=verbose
        )
        faces = load_shape_model_triangulation(path)
        return cls(
            path=Path(path),
            mean=np.ascontiguousarray(np.asarray(mean, dtype=np.float64).reshape(-1)),
            eigenfunctions=np.asarray(eigenfunctions, dtype=np.float64),
            eigenvalues=np.asarray(eigenvalues, dtype=np.float64).reshape(-1),
            landmarks=landmarks or {},
            faces=None if faces is None else np.ascontiguousarray(
                np.asarray(faces, dtype=np.int64)
            ),
        )

    @property
    def n_vertices(self) -> int:
        return self.mean.size // 3

    def describe(self) -> str:
        faces = "no faces" if self.faces is None else f"{len(self.faces)} faces"
        return (
            f"{self.path.name} ({self.n_vertices} vertices, "
            f"{self.eigenvalues.size} modes, {faces})"
        )


@dataclass(frozen=True)
class BuildContext:
    """Everything an injector may need, and nothing it may not."""

    domain: MeanDomain
    source: ModelSource
    verbose: bool = False
