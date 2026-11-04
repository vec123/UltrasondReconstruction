"""``build_model``: spec in, model out."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from src.models.gpmm.kernels.algebra import IsotropicLift, Sum
from src.models.gpmm.kernels.base import Kernel
from src.models.gpmm.lowrank.operator import KernelOperator, decompose
from src.models.gpmm.model.inference.continuous_inference import build_inference
from src.models.gpmm.model.mean_domain import MeanDomain
from src.models.gpmm.model.model import StatisticalShapeModel

from .choices import _truncation, choose_approximator, choose_inference, residual_noise
from .injectors import build_kernel_with_context
from .spec import ModelSource, _get


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
        from src.shared.vtk.extract import get_vtp_vertices
        from src.shared.vtk.io import load_vtp

        return np.asarray(get_vtp_vertices(load_vtp(str(path))), dtype=np.float64).reshape(-1)
    raise ValueError(f"mean must be 'data' or a path to a .vtp / .npy, got {spec!r}")


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
