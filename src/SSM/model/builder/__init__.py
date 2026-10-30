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

from .build import _print_variance, _resolve_mean, build_model, variance_report  # noqa: F401
from .choices import (  # noqa: F401
    DEFAULT_MAX_RANK,
    DENSE_ROW_LIMIT,
    _truncation,
    choose_approximator,
    choose_inference,
    residual_noise,
)
from .injectors import (  # noqa: F401
    INJECTORS,
    _compute_weight_field,
    _load_weights,
    _read_point_field,
    build_kernel_with_context,
    inject_laplace_beltrami,
    inject_localized,
    inject_pca,
    register_injector,
    resolve_kernel_spec,
)
from .spec import BuildContext, ModelSource, _as_float, _component, _get  # noqa: F401
