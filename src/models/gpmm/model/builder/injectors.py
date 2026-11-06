"""Table-driven resolution of the kernels that need an array: pca, laplace_beltrami and localized."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional
import numpy as np
from src.models.gpmm.kernels.base import Kernel, build_kernel
from src.models.gpmm.kernels.data import Pca
from src.models.gpmm.kernels.spectral import LaplaceBeltrami
from src.models.gpmm.model.mean_domain import MeanDomain

from .spec import BuildContext, ModelSource, _get


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

    from src.shared.vtk.io import load_vtp

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

    Imported inside the function deliberately: ``src.shared.weight_field`` pulls torch,
    and the point of the numpy boundary (section 14) is that building a model
    never needs it. A config that asks for this opts in; nothing else pays.
    """
    try:
        from src.shared.geometry.shape import load_shape
        from src.shared.core.config import load_yaml
        from src.shared.weight_field.landmark_weight_field import (
            LandmarkWeightFieldCfg,
            compute_landmark_weight_field,
        )
    except ImportError as error:
        raise ImportError(
            f"computing a weight field from {path.name} needs src.shared.weight_field, which "
            f"imports torch. Either install it, or bake the field onto a mesh once with "
            f"configs/surface/template/landmark_weight_field.yaml and point 'weights' at "
            f"that .vtp instead ({error})"
        ) from error

    cfg = LandmarkWeightFieldCfg.from_dict(load_yaml(str(path)))
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
