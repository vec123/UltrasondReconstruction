"""Covariance kernels for Gaussian Process Morphable Models.

A GPMM models the deformation from a reference shape as ``u ~ GP(0, k)``
(§Gaussian Process Morphable Models of ``theory/shape_deformation_models.tex``).
This package is the ``k``: a registry of covariance functions and an algebra for
combining them, so a model's prior is written down rather than only estimated.

    from src.SSM.kernels import Gaussian, Pca, Localized

    data   = Pca.from_eigendecomposition(Phi, eigenvalues)
    prior  = data + 0.3 * Gaussian(sigma=15.0)          # reach beyond the population
    local  = data * Gaussian(sigma=15.0)                # break the global modes

Four axes vary independently and nothing special-cases their combinations:

    stationary   gaussian | matern12/32/52 | rational_quadratic | wendland
    data         pca | diagonal_field
    spectral     laplace_beltrami
    algebra      sum | product | scale | localized | symmetry | isotropic_lift

Everything is numpy and nothing here imports torch: a model is built once,
offline, and cached, and no gradient is needed on that path. The stationary
kernels evaluate against the namespace of whatever arrays they are handed (see
``_array.py``), so the same code differentiates under torch when
``src/SSM/hyper`` learns kernel parameters - without this package depending on
it.

Kernels are evaluated on a *domain*, which supplies the reference points and
the metric. Anything carrying a ``points`` attribute will do, which is what
makes these testable without a mesh; ``src.SSM.model.domain.ReferenceDomain`` is
the real one and adds the quadrature measure, the triangulation and the option
of geodesic distances - under which every stationary kernel here becomes an
intrinsic one, unchanged.

Two properties of the interface are worth knowing before using it:

* ``structure`` records whether a kernel is dense, low-rank, sparse or diagonal,
  and the combinators propagate it. That is how the low-rank layer picks an
  exact cheap route instead of an eigensolver - ``pca`` and
  ``laplace_beltrami`` carry their factor with them.
* ``analytic_part()`` returns the sub-kernel that can be evaluated away from the
  sample points. ``pca`` and ``laplace_beltrami`` are tabulated and have none;
  a sum's analytic part is its parameterised terms. The continuous extension of
  §Practical notes regresses onto exactly that part, which is what keeps it
  non-circular.
"""

from src.SSM.kernels.algebra import (
    MAX_EXACT_PRODUCT_RANK,
    IsotropicLift,
    Localized,
    Product,
    Scale,
    Sum,
    Symmetry,
)
from src.SSM.kernels.base import (
    DEFAULT_CHUNK_BYTES,
    DENSE,
    DIAGONAL,
    KERNELS,
    SPARSE,
    Kernel,
    KernelDomain,
    KernelStructure,
    Structure,
    available_kernels,
    build_kernel,
    cross_distances,
    expand_indices,
    lowrank,
    pairwise_distances,
    product_structure,
    register_kernel,
    resolve_indices,
    sum_structure,
)
from src.SSM.kernels.data import DiagonalField, Pca
from src.SSM.kernels.spectral import LaplaceBeltrami
from src.SSM.kernels.stationary import (
    Constant,
    Gaussian,
    Matern12,
    Matern32,
    Matern52,
    RationalQuadratic,
    Stationary,
    Wendland,
    White,
)

__all__ = [
    # interface
    "Kernel",
    "KernelDomain",
    "KernelStructure",
    "Structure",
    "DENSE",
    "SPARSE",
    "DIAGONAL",
    "lowrank",
    "sum_structure",
    "product_structure",
    "DEFAULT_CHUNK_BYTES",
    # registry
    "KERNELS",
    "register_kernel",
    "available_kernels",
    "build_kernel",
    # helpers
    "resolve_indices",
    "expand_indices",
    "pairwise_distances",
    "cross_distances",
    # stationary
    "Stationary",
    "Gaussian",
    "Matern12",
    "Matern32",
    "Matern52",
    "RationalQuadratic",
    "Wendland",
    "White",
    "Constant",
    # data
    "Pca",
    "DiagonalField",
    # spectral
    "LaplaceBeltrami",
    # algebra
    "Sum",
    "Product",
    "Scale",
    "Localized",
    "Symmetry",
    "IsotropicLift",
    "MAX_EXACT_PRODUCT_RANK",
]
