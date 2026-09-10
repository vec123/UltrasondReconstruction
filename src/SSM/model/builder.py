"""Spec in, model out. One function, and nothing else constructs a GPMM.

    from src.SSM.model.builder import build_model

    model = build_model({
        "domain": {"source": "src/SSM/FetusModel_x4", "metric": "euclidean"},
        "kernel": {"name": "sum", "children": [
            {"name": "pca"},
            {"name": "scale", "kwargs": {"scale": 0.3},
             "children": [{"name": "gaussian", "kwargs": {"sigma": 15.0}}]},
        ]},
        "truncation": {"variance_threshold": 0.99},
    }, verbose=True)

**Why this is a Builder and not a registry call.** ``build_kernel`` already
turns ``{"name": "gaussian", "kwargs": {"sigma": 15.0}}`` into a kernel with no
help - a config can name any parameterised kernel and get it. What a config
*cannot* do is carry a 93081 x 163 array, and three kernels need one:

    pca                basis (3n, r)             from the model source
    laplace_beltrami   eigenvectors, eigenvalues from the domain mesh
    localized          weights (n,)              from a weight field

Resolving those needs the domain and the model source - context the registry
does not have and should not acquire. That context is this module, and the
resolution is table-driven (:data:`INJECTORS`), so a fourth kernel that needs an
array is one function with a decorator, not an edit to the walker.

**The sequence** (``docs/gpmm_architecture.md`` section 11):

    1. source       .mat or atlas -> mean, Phi, Lambda, landmarks, faces
    2. mean         which reference shape the deformation is added to
    3. domain       MeanDomain(points, faces, mass, metric)
    4. kernel       resolve the spec tree, injecting arrays at the three leaves
    5. operator     KernelOperator(kernel, domain)
    6. decompose    exact_factor when the kernel carries its own factor,
                    otherwise the named or auto-chosen approximator
    7. truncate     strictest of n_components / variance / min_eigenvalue
    8. inference    prepared against kernel.analytic_part()
    9. assemble     StatisticalShapeModel, noise = discarded variance

**Not here: caching.** :func:`build_model` is pure - a spec and the files it
names go in, a model comes out - which is what lets ``io.py`` wrap it with a
content-addressed cache (section 12) instead of having to reach inside it. A
builder that wrote to disk could not be used by a hyperparameter search that
must not.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.SSM.kernels.algebra import IsotropicLift, Sum
from src.SSM.kernels.base import Kernel, build_kernel
from src.SSM.kernels.data import Pca
from src.SSM.kernels.spectral import LaplaceBeltrami
from src.SSM.lowrank.operator import KernelOperator, decompose
from src.SSM.lowrank.truncation import KEEP_ALL, Decomposition, Truncation
from src.SSM.model.continuous_inference import build_inference
from src.SSM.model.mean_domain import MeanDomain
from src.SSM.model.model import StatisticalShapeModel

#: Above this many rows the auto-selection stops offering ``dense_eigh``:
#: 8000^2 x 8 bytes is 512 MB for the matrix alone, and the fetus model at
#: 5743 vertices (17229 rows) is already past it. Naming the number here rather
#: than burying it in a branch makes it adjustable for a machine with more.
DENSE_ROW_LIMIT = 8_000

#: Fallback column budget for ``pivoted_cholesky`` when the truncation states no
#: cap. The factor is ``N x max_rank``: 300 columns is 223 MB at baby scale.
DEFAULT_MAX_RANK = 300


# --------------------------------------------------------------------------
# reading a spec that may be a mapping or a config dataclass
# --------------------------------------------------------------------------

def _get(spec, key: str, default=None):
    """``spec[key]`` or ``spec.key``, whichever the object supports.

    The same accommodation ``build_kernel`` makes, and for the same reason: the
    builder must take both a plain dict written in a test and the config
    dataclass of section 13, without depending on ``src.config``. An explicit
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

    An Adapter over :mod:`src.SSM.model.load`, which already puts a ``.mat`` and
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
        from src.SSM.model.load import load_shape_model, load_shape_model_triangulation

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


# --------------------------------------------------------------------------
# array injection - the three kernels a config cannot fully describe
# --------------------------------------------------------------------------

#: kernel name -> ``f(kwargs, context) -> kwargs`` with the arrays filled in.
INJECTORS: Dict[str, Callable[[Dict[str, Any], BuildContext], Dict[str, Any]]] = {}


def register_injector(name: str):
    """Files a resolver under a kernel name. Mirrors ``register_kernel``."""

    def decorator(function):
        if name in INJECTORS:
            raise ValueError(f"an injector for kernel '{name}' is already registered")
        INJECTORS[name] = function
        return function

    return decorator


@register_injector("pca")
def inject_pca(kwargs: Dict[str, Any], ctx: BuildContext) -> Dict[str, Any]:
    """``basis = Phi sqrt(Lambda)`` from the build's source, or another model's.

    ``source`` names where the statistics come from: omitted (or any of
    ``data`` / ``self`` / ``atlas``) means this build's own source, and a path
    means a different model - which is how a prior learned on one population is
    put on another mesh, provided the two are in vertex-wise correspondence.

    ``n_components`` truncates the basis before it becomes a kernel. Cheaper
    than truncating afterwards, because the rank of ``pca`` bounds the rank of
    everything downstream.
    """
    kwargs = dict(kwargs)
    if "basis" in kwargs:
        return kwargs                       # already resolved; a search reuses it

    where = kwargs.pop("source", "data")
    n_components = kwargs.pop("n_components", None)

    if isinstance(where, str) and where.lower() in ("data", "self", "atlas", "source"):
        source = ctx.source
    else:
        source = ModelSource.load(where, verbose=ctx.verbose)

    Phi, lam = source.eigenfunctions, source.eigenvalues
    if Phi.shape[0] != 3 * ctx.domain.n:
        raise ValueError(
            f"the pca basis has {Phi.shape[0]} rows ({Phi.shape[0] // 3} vertices) but the "
            f"domain has {ctx.domain.n}. A borrowed basis must be in vertex-wise "
            f"correspondence with the mesh it is used on."
        )
    if n_components is not None:
        keep = int(n_components)
        Phi, lam = Phi[:, :keep], lam[:keep]

    kwargs["basis"] = Pca.from_eigendecomposition(Phi, lam).basis
    return kwargs


@register_injector("laplace_beltrami")
def inject_laplace_beltrami(kwargs: Dict[str, Any], ctx: BuildContext) -> Dict[str, Any]:
    """Solves the mesh eigenproblem, so a config need only ask for ``n_eig``."""
    kwargs = dict(kwargs)
    if "eigenvectors" in kwargs:
        return kwargs

    n_eig = int(kwargs.pop("n_eig", 100))
    if ctx.domain.faces is None:
        raise ValueError(
            "kernel 'laplace_beltrami' needs a triangulation to solve its eigenproblem, "
            "and this domain has none. Either the source carries no faces or they were "
            "not passed through."
        )

    spectral = LaplaceBeltrami.from_mesh(ctx.domain.points, ctx.domain.faces, n_eig=n_eig)
    kwargs["eigenvectors"] = spectral.eigenvectors
    kwargs["eigenvalues"] = spectral.eigenvalues
    return kwargs


@register_injector("localized")
def inject_localized(kwargs: Dict[str, Any], ctx: BuildContext) -> Dict[str, Any]:
    """Per-vertex weights, from an array, a ``.npy``, a mesh field or a config."""
    kwargs = dict(kwargs)
    weights = kwargs.pop("weight_field", None)
    if "weights" in kwargs:
        weights = kwargs.pop("weights")
    if weights is None:
        raise ValueError(
            "kernel 'localized' needs per-vertex weights: pass 'weights' (an array, a "
            ".npy, or a .vtp plus 'weights_field') or 'weight_field' (a landmark "
            "weight-field YAML config)."
        )

    field_name = kwargs.pop("weights_field", None)
    resolved = _load_weights(weights, field_name, ctx)
    if resolved.size != ctx.domain.n:
        raise ValueError(
            f"{resolved.size} localisation weights for a domain of {ctx.domain.n} vertices"
        )
    kwargs["weights"] = resolved
    return kwargs


def _load_weights(value, field_name: Optional[str], ctx: BuildContext) -> np.ndarray:
    """An array, a ``.npy``/``.npz``, a mesh point field, or a weight-field config."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float64).reshape(-1)

    path = Path(str(value))
    suffix = path.suffix.lower()

    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64).reshape(-1)

    if suffix == ".npz":
        with np.load(path) as data:
            key = field_name or ("weights" if "weights" in data else list(data)[0])
            return np.asarray(data[key], dtype=np.float64).reshape(-1)

    if suffix in (".vtp", ".vtk"):
        return _read_point_field(path, field_name)

    if suffix in (".yaml", ".yml"):
        return _compute_weight_field(path, ctx)

    raise ValueError(
        f"cannot read localisation weights from {path}: expected .npy, .npz, .vtp or a "
        f"weight-field .yaml"
    )


def _read_point_field(path: Path, field_name: Optional[str]) -> np.ndarray:
    """One named point-data array off a mesh - what the weight-field script writes."""
    from vtk.util.numpy_support import vtk_to_numpy

    from src.vtk.io import load_vtp

    if not field_name:
        raise ValueError(
            f"reading weights from {path.name} needs 'weights_field' naming the point "
            f"array to read (the weight-field script writes 'landmark_weight')"
        )
    point_data = load_vtp(str(path)).GetPointData()
    array = point_data.GetArray(field_name)
    if array is None:
        available = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]
        raise ValueError(f"{path.name} has no point field {field_name!r}; it has {available}")
    return np.asarray(vtk_to_numpy(array), dtype=np.float64).reshape(-1)


def _compute_weight_field(path: Path, ctx: BuildContext) -> np.ndarray:
    """Runs a landmark weight-field config. The one torch-touching path here.

    Imported inside the function deliberately: ``src.weight_field`` pulls torch,
    and the point of the numpy boundary (section 14) is that building a model
    never needs it. A config that asks for this opts in; nothing else pays.
    """
    try:
        from src.resnet_lddmm.resnet_flow.io import load_shape
        from src.weight_field.config import load_yaml_mapping
        from src.weight_field.landmark_weight_field import (
            LandmarkWeightFieldCfg,
            compute_landmark_weight_field,
        )
    except ImportError as error:
        raise ImportError(
            f"computing a weight field from {path.name} needs src.weight_field, which "
            f"imports torch. Either install it, or bake the field onto a mesh once with "
            f"scripts/weight_fields/write_landmark_weight_field.py and point 'weights' at "
            f"that .vtp instead ({error})"
        ) from error

    cfg = LandmarkWeightFieldCfg.from_dict(load_yaml_mapping(str(path)))
    weights = compute_landmark_weight_field(load_shape(cfg.vtp), cfg)
    return np.asarray(weights.detach().cpu().numpy(), dtype=np.float64).reshape(-1)


# --------------------------------------------------------------------------
# the kernel tree
# --------------------------------------------------------------------------

def resolve_kernel_spec(spec, ctx: BuildContext):
    """A spec tree with every array filled in, ready for ``build_kernel``.

    Resolution and construction stay separate on purpose. Construction already
    exists, is recursive, and validates kwargs against each class's own
    signature; re-implementing it here to inject three arrays would fork that
    validation. So this returns a spec, and exactly one code path builds.
    """
    if isinstance(spec, Kernel):
        return spec

    name = _get(spec, "name")
    if not name:
        raise ValueError(f"kernel spec has no 'name': {spec!r}")

    children = [resolve_kernel_spec(child, ctx) for child in (_get(spec, "children") or [])]
    single = _get(spec, "child")
    if single is not None:
        children.append(resolve_kernel_spec(single, ctx))

    kwargs = dict(_get(spec, "kwargs", {}) or {})
    injector = INJECTORS.get(name)
    if injector is not None:
        kwargs = injector(kwargs, ctx)

    resolved = {"name": name, "kwargs": kwargs}
    if children:
        resolved["children"] = children
    return resolved


def build_kernel_with_context(spec, domain: MeanDomain, source: ModelSource,
                              verbose: bool = False) -> Kernel:
    """The kernel a spec names, with ``pca``/``laplace_beltrami``/``localized`` resolved."""
    return build_kernel(resolve_kernel_spec(spec, BuildContext(domain, source, verbose)))


def variance_report(kernel: Kernel, domain) -> List[Tuple[str, float]]:
    """Mean per-coordinate variance of each top-level term, largest first.

    Worth printing on every build. On the baby model the data kernel's mean
    per-coordinate variance is 4.16e-08 while its largest eigenvalue is 7.00e-04,
    so a smooth term whose ``scale`` was set from ``lambda_max`` - the
    obvious-looking choice - is **16836x** too large, and the population then
    contributes nothing visible to the result. That failure is silent in every
    other diagnostic: the model builds, samples and fits, it has just stopped
    being a statistical shape model. Two numbers side by side make it obvious.
    """
    terms = kernel.children if isinstance(kernel, Sum) else [kernel]
    rows = [(term.describe(), float(np.asarray(term.diag(domain)).mean())) for term in terms]
    return sorted(rows, key=lambda row: -row[1])


# --------------------------------------------------------------------------
# choosing how to decompose and how to extend
# --------------------------------------------------------------------------

def choose_approximator(kernel: Kernel, domain, requested=None,
                        truncation: Truncation = KEEP_ALL) -> Tuple[str, Dict[str, Any]]:
    """The named method, or the one the operator's own structure dictates.

    Auto-selection is not a micro-optimisation. A ``pca`` kernel already carries
    ``Phi sqrt(Lambda)``, so ``exact_factor`` is one thin SVD - 3.2 s on the baby
    model, reproducing its eigenvalues to 4e-15. Any other route to the same
    answer is either a 65 GB allocation or an approximation of something already
    known in closed form.
    """
    name, kwargs = _component(requested, "auto")
    if name != "auto":
        return name, kwargs

    structure = kernel.structure
    rows = kernel.block_dim * domain.n

    if structure.is_lowrank and structure.rank and structure.rank <= rows:
        return "exact_factor", {}
    if rows <= DENSE_ROW_LIMIT:
        return "dense_eigh", {}

    max_rank = truncation.n_components or DEFAULT_MAX_RANK
    return "pivoted_cholesky", {"max_rank": int(max_rank), "tolerance": 1e-3}


def choose_inference(kernel: Kernel, domain, requested=None) -> Tuple[str, Dict[str, Any]]:
    """The named strategy, or the best one this kernel actually supports.

    ``kernel_regression`` needs an analytic part to regress onto, which a purely
    tabulated kernel (``pca`` alone) does not have. Rather than let it raise, the
    auto path falls back to ``barycentric`` where there is a triangulation and to
    ``none`` where there is not. An explicitly named strategy is honoured and
    allowed to fail, because a config that asked for continuity should hear that
    it cannot have it.
    """
    name, kwargs = _component(requested, "auto")
    if name != "auto":
        return name, kwargs

    if kernel.analytic_part() is not None:
        return "kernel_regression", {}
    if getattr(domain, "faces", None) is not None:
        return "barycentric", {}
    return "none", {}


def residual_noise(decomposition: Decomposition, block_dim: int) -> float:
    """The discarded variance as a *per-coordinate* variance - the ppca epsilon.

    ``residual_variance`` is a trace, and converting it to a per-coordinate
    variance depends on the measure the decomposition was taken in - exactly the
    class of factor-of-n error that ``rescale_eigenvalues`` exists to undo, so it
    is worth doing explicitly rather than by eye:

    * ``mass``: the trace is ``sum_i m_i sum_c K(3i+c, 3i+c)`` and the masses sum
      to one, so it is already an area-weighted average over vertices - divide by
      ``block_dim`` alone, to get one coordinate's share.
    * ``euclidean``: the trace is a plain sum over all ``3n`` coordinates -
      divide by all of them.
    """
    residual = float(decomposition.residual_variance)
    if decomposition.inner_product == "mass":
        return residual / max(int(block_dim), 1)
    return residual / max(decomposition.n_rows, 1)


# --------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------

def _resolve_mean(spec, source: ModelSource) -> np.ndarray:
    """The reference shape the deformation is added to."""
    if spec is None or (isinstance(spec, str) and spec.lower() in ("data", "source", "self")):
        return source.mean

    if isinstance(spec, (list, tuple, np.ndarray)):
        return np.asarray(spec, dtype=np.float64).reshape(-1)

    if isinstance(spec, str) and spec.lower() == "zero":
        raise ValueError(
            "mean: 'zero' is not meaningful here. A GPMM's *deformation* mean is already "
            "zero - u ~ GP(0, k) - and this field names the reference shape the "
            "deformation is added to, which cannot be the origin. Use 'data', or a path "
            "to a .vtp / .npy holding the reference points."
        )

    path = Path(str(spec))
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if path.suffix.lower() in (".vtp", ".vtk"):
        from src.vtk.extract import get_vtp_vertices
        from src.vtk.io import load_vtp

        return np.asarray(get_vtp_vertices(load_vtp(str(path))), dtype=np.float64).reshape(-1)
    raise ValueError(f"mean must be 'data' or a path to a .vtp / .npy, got {spec!r}")


def _truncation(spec) -> Truncation:
    if isinstance(spec, Truncation):
        return spec
    if spec is None:
        return KEEP_ALL
    n_components = _get(spec, "n_components")
    return Truncation(
        n_components=None if n_components is None else int(n_components),
        variance_threshold=_as_float(_get(spec, "variance_threshold", 1.0), "variance_threshold"),
        min_eigenvalue=_as_float(_get(spec, "min_eigenvalue", 0.0), "min_eigenvalue"),
    )


def build_model(spec, *, verbose: bool = False) -> StatisticalShapeModel:
    """A spec and the files it names, in; a ``StatisticalShapeModel``, out.

    The returned model is the type the pipeline already consumes, so
    ``ShapeInstance``, ``ShapeSpace`` and every metric take it unchanged - the
    kernel, domain and inference ride along as extra knowledge rather than as a
    different interface.

    **One thing changes meaning under the default.** With ``domain.mass`` on
    (the default, and the deliberate deviation from the theory's uniform
    ``d mu`` documented in ``mean_domain.py``), the eigenfunctions are
    orthonormal in the *mass* inner product: ``Phi^T M Phi = I``, not
    ``Phi^T Phi = I``. The model records that in ``inner_product``, and
    ``Phi Lambda^{1/2}`` - the array sampling and fitting actually use - is
    identical either way, so samples are unaffected. Fitting used to assume the
    Euclidean case too, projecting with a bare ``Phi^T r``; it now goes through
    ``ShapeInstance.closed_form_solution``, which solves
    ``(Phi^T Phi + gamma I) da = Phi^T r`` and is therefore the Euclidean
    least-squares answer whichever inner product Phi is orthonormal in.
    """
    started = time.perf_counter()
    domain_spec = _get(spec, "domain", {})

    # 1. source
    source_path = _get(domain_spec, "source") or _get(spec, "source")
    if source_path is None:
        raise ValueError(
            "a model spec needs domain.source: a BabyFM .mat file or a Deformetrica "
            "atlas directory"
        )
    source = ModelSource.load(
        source_path,
        normalize=bool(_get(spec, "normalize", True)),
        atlas_options=_get(spec, "atlas"),
        verbose=verbose,
    )

    # 2-3. mean and domain
    mean = _resolve_mean(_get(spec, "mean", "data"), source)
    faces = source.faces if _get(domain_spec, "faces", True) is not False else None
    domain = MeanDomain.from_mean(
        mean, faces, metric=str(_get(domain_spec, "metric", "euclidean"))
    )
    use_mass = bool(_get(domain_spec, "mass", True))

    # 4. kernel. Defaulting to the empirical kernel alone reproduces the
    #    source's own PDM, which makes "no kernel stated" the legacy model
    #    rather than an error - and gives every comparison a baseline.
    kernel_spec = _get(spec, "kernel", {"name": "pca"})
    kernel = build_kernel_with_context(kernel_spec, domain, source, verbose=verbose)
    # Scalar kernels describe one independent covariance per coordinate. Lift
    # them before decomposition so every model still exposes the repository's
    # vertex-major (3n, L) basis contract.
    if kernel.block_dim == 1:
        kernel = IsotropicLift(kernel)

    if verbose:
        print("build_model")
        print(f"  source      {source.describe()}")
        print(f"  domain      {domain.describe()}")
        print(f"  kernel      {kernel.describe()}")
        _print_variance(variance_report(kernel, domain))

    # 5-7. operator, decomposition, truncation
    operator = KernelOperator(kernel, domain)
    truncation = _truncation(_get(spec, "truncation"))
    method, method_kwargs = choose_approximator(
        kernel, domain, _get(spec, "approximation"), truncation
    )

    clock = time.perf_counter()
    decomposition = decompose(
        operator, method, method_kwargs,
        truncation=truncation,
        mass=domain.mass if use_mass else None,
    )
    elapsed = time.perf_counter() - clock

    if verbose:
        detail = " ".join(f"{key}={value}" for key, value in method_kwargs.items())
        print(f"  decompose   {method} {detail}".rstrip())
        print(f"              rank {decomposition.rank}, {decomposition.explained:.2%} of "
              f"variance, {elapsed:.1f} s")

    # 8. continuous inference
    name, kwargs = choose_inference(kernel, domain, _get(spec, "inference"))
    inference = build_inference(name, kwargs).prepare(domain, kernel, decomposition.basis)

    # 9. assemble
    noise = residual_noise(decomposition, kernel.block_dim)
    model = StatisticalShapeModel(
        mean=mean,
        eigenfunctions=decomposition.basis,
        eigenvalues=decomposition.values,
        landmarks=source.landmarks,
        faces=faces,
        domain=domain,
        kernel=kernel,
        inference=inference,
        noise=noise,
        inner_product=decomposition.inner_product,
    )

    if verbose:
        caveat = (" (Phi^T M Phi = I; Phi^T r is NOT the projection)"
                  if decomposition.inner_product == "mass" else "")
        print(f"  inference   {inference.describe()}")
        print(f"  noise       {noise:.3e} per coordinate "
              f"({1.0 - decomposition.explained:.2%} of the variance discarded)")
        print(f"  basis       {decomposition.inner_product}-orthonormal{caveat}")
        print(f"  total       {time.perf_counter() - started:.1f} s")

    return model


def _print_variance(rows: List[Tuple[str, float]]) -> None:
    if not rows:
        return
    largest = rows[0][1]
    for index, (label, value) in enumerate(rows):
        if not index:
            share = ""
        elif value > 0:
            share = f"   ({largest / value:,.0f}x smaller than the first term)"
        else:
            share = "   (contributes nothing)"
        label_column = "variance" if index == 0 else "        "
        print(f"  {label_column}    {label[:44]:46s}{value:.3e}{share}")
