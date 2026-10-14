"""Where in the shape model the on-screen shape sits: a linear sample, an LDDMM sample, or a real subject.

A landmark is a vertex *index*, so it stays valid anywhere in shape space;
this module lets the picker show the model away from its mean, where a point
of interest may be easier to see, and check that a placement still sits on
its anatomy there. Two parametrizations, toggled at runtime:

* linear - mean + sum_k alpha_k phi_k, PCA on ambient vertex coordinates.
  One matrix product per sample, so a slider drag redraws every frame.
* lddmm - atlas models only. PCA on the initial momenta instead; every sample
  is reached by shooting the geodesic, so it cannot fold the surface through
  itself the way the linear sum can. Costs an integration per sample.

Switching resets the coefficients to zero: a coefficient in one basis has no
meaning in the other. Separately, :meth:`ShapeSpace.step_subject` shoots one of
the atlas's real training subjects - the most faithful "different sample",
carrying none of a slider combination's risk of looking like no real fetus.

No VTK here: the app copies :meth:`ShapeSpace.current_shape` into its mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from src.config.SSM.schema import PlaceLandmarksConfig
from src.SSM.model.deformetrica import load_momentum_model, shoot
from src.SSM.model.load import is_atlas_path, load_shape_model, load_shape_model_triangulation

Log = Callable[..., None]

#: Minimum seconds between redraws while dragging an LDDMM slider: each
#: sample is a geodesic integration, so a drag is throttled and the release
#: always gets the last word.
LDDMM_DRAG_INTERVAL = 0.15


class ShapeSpace:
    """The active parametrization, its coefficients, and the subject being browsed (if any).

    ``alpha``, ``sigma`` and ``variance_share`` always describe the *active*
    parametrization - this object is the :class:`~src.GUIs.core.SliderSource`
    the slider panel edits. ``alpha`` is rebound (not resized) on a toggle, so
    readers must not cache it.
    """

    def __init__(self, mean: np.ndarray, linear_basis: np.ndarray, linear_sigma: np.ndarray,
                 linear_variance_share: np.ndarray, momentum_model=None,
                 lddmm_sigma: Optional[np.ndarray] = None,
                 lddmm_variance_share: Optional[np.ndarray] = None,
                 lddmm_time_points: int = 6):
        """Start at the mean, in the linear parametrization.

        Args:
            mean: (N, 3) mean shape.
            linear_basis: (3N, K) PCA eigenfunctions exposed on the panel.
            linear_sigma: (K,) their standard deviations.
            linear_variance_share: (K,) fraction of total variance per mode.
            momentum_model: ``MomentumShapeModel`` for the LDDMM parametrization
                and subject browsing, or None when the model is not an atlas.
            lddmm_sigma, lddmm_variance_share: the momentum modes' counterparts.
            lddmm_time_points: geodesic shooting resolution.
        """
        self.mean = np.asarray(mean, dtype=np.float64)
        self.linear_basis = linear_basis
        self.linear_sigma = linear_sigma
        self.linear_variance_share = linear_variance_share
        self.momentum_model = momentum_model
        self.lddmm_sigma = lddmm_sigma
        self.lddmm_variance_share = lddmm_variance_share
        self.lddmm_time_points = int(lddmm_time_points)

        self.parametrization = "linear"
        self.sigma = linear_sigma
        self.variance_share = linear_variance_share
        self.alpha = np.zeros(len(linear_sigma))

        # None: showing the alpha sample. An int: showing that real subject,
        # ignoring alpha until a coefficient changes again.
        self.subject_index: Optional[int] = None
        self._subject_cache: Dict[int, np.ndarray] = {}

    # -- queries ---------------------------------------------------------------

    @property
    def has_lddmm(self) -> bool:
        return self.momentum_model is not None

    @property
    def n_subjects(self) -> int:
        return self.momentum_model.atlas.n_subjects if self.has_lddmm else 0

    @property
    def drag_interval(self) -> float:
        """Seconds to wait between redraws while dragging, for the active parametrization."""
        return LDDMM_DRAG_INTERVAL if self.parametrization == "lddmm" else 0.0

    def label(self) -> str:
        """Short description of what is on screen, for the status line."""
        if self.subject_index is not None:
            return f"subject {self.subject_index + 1}/{self.n_subjects}"
        return self.parametrization

    # -- the shape on screen -----------------------------------------------------

    def sample(self) -> np.ndarray:
        """(N, 3) the shape the current coefficients stand for.

        The LDDMM momentum is built from *only* the exposed modes (alpha's
        length), slicing the model's modes to match, rather than calling its
        reconstruct()/momentum(): those multiply against every mode it has and
        would fail as soon as the panel exposes fewer modes than the model's rank.
        """
        if self.parametrization == "lddmm":
            model = self.momentum_model
            k = len(self.alpha)
            momentum = model.mean_momentum
            if k:
                momentum = momentum + model.modes[:, :k] @ self.alpha
            return shoot(model.atlas.control_points, momentum.reshape(-1, 3),
                         model.atlas.template, model.atlas.kernel_width,
                         self.lddmm_time_points)
        if len(self.alpha) == 0:
            return self.mean
        return self.mean + (self.linear_basis @ self.alpha).reshape(-1, 3)

    def current_shape(self) -> np.ndarray:
        """The browsed subject if one is active, else the alpha sample."""
        if self.subject_index is not None:
            return self._subject_cache[self.subject_index]
        return self.sample()

    # -- navigation ------------------------------------------------------------

    def leave_subject(self) -> None:
        """Back to the alpha sample - called whenever a coefficient is touched."""
        self.subject_index = None

    def reset(self) -> None:
        """Back to the mean: no subject, every coefficient zero (in place)."""
        self.subject_index = None
        self.alpha[:] = 0.0

    def toggle_parametrization(self) -> str:
        """linear <-> lddmm, coefficients reset to zero; returns the new parametrization.

        Raises:
            RuntimeError: the model has no atlas momenta, so no LDDMM parametrization.
        """
        if not self.has_lddmm:
            raise RuntimeError("this model has no atlas momenta to shoot")
        if self.parametrization == "linear":
            self.parametrization = "lddmm"
            self.sigma, self.variance_share = self.lddmm_sigma, self.lddmm_variance_share
        else:
            self.parametrization = "linear"
            self.sigma, self.variance_share = self.linear_sigma, self.linear_variance_share
        self.subject_index = None
        self.alpha = np.zeros(len(self.sigma))
        return self.parametrization

    def step_subject(self, delta: int, log: Optional[Log] = None) -> int:
        """Show the real subject ``delta`` steps from the current one (wrapping); returns its index.

        From the alpha sample, +1 starts at the first subject and -1 at the last.
        Shot shapes are cached: revisiting subjects while comparing is the point.

        Raises:
            RuntimeError: the model has no atlas subjects.
        """
        if not self.has_lddmm:
            raise RuntimeError("this model has no atlas momenta")
        atlas = self.momentum_model.atlas
        start = (-1 if delta > 0 else 0) if self.subject_index is None else self.subject_index
        index = int(start + delta) % atlas.n_subjects
        if index not in self._subject_cache:
            if log is not None:
                log(f"  shooting subject {index + 1}/{atlas.n_subjects}")
            self._subject_cache[index] = shoot(atlas.control_points, atlas.momenta[index],
                                               atlas.template, atlas.kernel_width,
                                               self.lddmm_time_points)
        self.subject_index = index
        return index


@dataclass
class LoadedModel:
    """What the picker needs from a shape model file."""

    space: ShapeSpace
    faces: Optional[np.ndarray]
    n_modes_total: int


def load_shape_space(cfg: PlaceLandmarksConfig, log: Log) -> LoadedModel:
    """Read the model ``cfg.shape_model`` into a :class:`ShapeSpace`.

    The panel exposes ``cfg.sliders.mode_count`` leading modes (all when
    None). The LDDMM parametrization is loaded eagerly - just an SVD on the
    momenta - so its availability is known (and logged) up front; it is
    optional, so a failure to load it is logged, not raised.

    Raises:
        FileNotFoundError: ``cfg.shape_model`` does not exist.
    """
    if not cfg.shape_model.exists():
        raise FileNotFoundError(f"shape_model does not exist: {cfg.shape_model}")

    mean, eigenfunctions, eigenvalues, _ = load_shape_model(
        cfg.shape_model, normalize=cfg.model.normalize,
        atlas_options=cfg.model.atlas.to_options(), verbose=cfg.verbose,
    )
    mean = np.asarray(mean, dtype=np.float64).reshape(-1, 3)
    faces = load_shape_model_triangulation(cfg.shape_model)

    eigenvalues = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    n_modes = _panel_modes(cfg, len(eigenvalues))
    total = eigenvalues.sum()
    linear_basis = np.ascontiguousarray(np.asarray(eigenfunctions, dtype=np.float64)[:, :n_modes])
    linear_sigma = np.sqrt(eigenvalues[:n_modes])
    linear_share = eigenvalues[:n_modes] / total if total > 0 else np.zeros(n_modes)

    log(f"model: {cfg.shape_model}")
    log(f"  {len(mean)} vertices, {0 if faces is None else len(faces)} triangles, "
        f"{len(eigenvalues)} modes ({n_modes} on the panel)")

    momentum_model, lddmm_sigma, lddmm_share = _load_lddmm(cfg, log)
    space = ShapeSpace(mean, linear_basis, linear_sigma, linear_share,
                       momentum_model, lddmm_sigma, lddmm_share, cfg.lddmm_time_points)
    return LoadedModel(space=space, faces=faces, n_modes_total=len(eigenvalues))


def _panel_modes(cfg: PlaceLandmarksConfig, available: int) -> int:
    if cfg.sliders.mode_count is None:
        return available
    return min(int(cfg.sliders.mode_count), available)


def _load_lddmm(cfg: PlaceLandmarksConfig, log: Log):
    """(momentum_model, sigma, variance_share), or three Nones when unavailable."""
    if not is_atlas_path(cfg.shape_model):
        log("  LDDMM unavailable: not a Deformetrica atlas (no momenta to shoot)")
        return None, None, None
    try:
        model = load_momentum_model(cfg.shape_model, kernel_width=cfg.model.atlas.kernel_width,
                                    time_points=cfg.lddmm_time_points, verbose=False)
    except Exception as exc:                       # noqa: BLE001 - optional feature, not fatal
        log(f"  LDDMM unavailable: {type(exc).__name__}: {exc}")
        return None, None, None

    k = _panel_modes(cfg, model.n_modes)
    log(f"  LDDMM available: {model.n_modes} momentum modes ({k} on the panel), "
        f"{cfg.lddmm_time_points} shooting time points")
    return model, np.sqrt(model.eigenvalues[:k]), model.variance_share[:k]
