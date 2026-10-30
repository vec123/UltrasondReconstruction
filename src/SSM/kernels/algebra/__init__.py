"""Combinators - the part that makes a GPMM more than a covariance.

§Gaussian Process Morphable Models builds its total covariance as

    k(p, p') = k_PCA(p, p') + k_smooth(p, p')      or      k_PCA * k_smooth

and the theory notes the third variant explicitly: when ``k_smooth`` is scalar
the product is the **block-scaled** ``K_ij = k_smooth(p_i, p_j) * K_PCA,ij``,
not a Hadamard product against a lifted ``k_smooth I_3``. Those are different
matrices - the lift zeroes the off-diagonal entries inside every 3x3 block, and
using it would quietly destroy the directional structure the data kernel exists
to carry. :class:`Product` distinguishes the two cases; see its docstring.

Each combinator is itself a :class:`~src.SSM.kernels.base.Kernel`, so an
expression nests to any depth and needs no special evaluator:

    pca + 0.3 * Gaussian(sigma=15)
    Localized(Matern52(rho=8), weights) + White(1e-8)
    Symmetry(pca * Gaussian(sigma=20), mirror)

What the two forms buy is different, and worth choosing deliberately:

**Sum** *adds* directions. The 163 modes of a PCA model span a fixed subspace;
anything outside it is unreachable, however implausible the target. Adding a
smooth kernel makes the model able to reach shapes the population never showed,
and incidentally makes ``K`` non-singular so a posterior stops being degenerate.

**Product** *localises*. It decorrelates distant vertices while keeping the
population's local correlation structure, which is what breaks the "every mode
is global" property of PCA. That is the useful one for fitting partial data, and
it is also the expensive one - see the rank note on :class:`Product`.

The rank consequence is worth stating once, because it surprises people: a
smooth kernel has numerical rank of order ``area / sigma^2``, so a product with
it multiplies the number of modes rather than adding to it. Localising a
163-mode model at a tenth of its extent gives an object of rank ~10^4 before
truncation. That is the freedom being bought, not a defect - but the truncation
level, the regularisation weight and the chi-square bound all mean different
things afterwards.
"""

from __future__ import annotations

from src.SSM.kernels.base import expand_indices  # noqa: F401  (unused in the old module; kept so the import still happens)

from .common import _check_children, _namespaced, _split_namespaced  # noqa: F401
from .lift import IsotropicLift  # noqa: F401
from .localized import Localized  # noqa: F401
from .product_kernel import MAX_EXACT_PRODUCT_RANK, Product, _repeat_blocks, _row_khatri_rao  # noqa: F401
from .scale import Scale  # noqa: F401
from .sum_kernel import Sum, _lift_if_needed  # noqa: F401
from .symmetry import Symmetry  # noqa: F401
