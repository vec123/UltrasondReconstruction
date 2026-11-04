"""Defaults the builder picks from the model size: approximator, inference, noise and truncation."""

from __future__ import annotations

from typing import Any, Dict, Tuple
from src.models.gpmm.kernels.base import Kernel
from src.models.gpmm.lowrank.truncation import Decomposition, KEEP_ALL, Truncation

from .spec import _as_float, _component, _get


#: Above this many rows the auto-selection stops offering ``dense_eigh``:
#: 8000^2 x 8 bytes is 512 MB for the matrix alone, and the fetus model at
#: 5743 vertices (17229 rows) is already past it. Naming the number here rather
#: than burying it in a branch makes it adjustable for a machine with more.
DENSE_ROW_LIMIT = 8_000


#: Fallback column budget for ``pivoted_cholesky`` when the truncation states no
#: cap. The factor is ``N x max_rank``: 300 columns is 223 MB at baby scale.
DEFAULT_MAX_RANK = 300


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
