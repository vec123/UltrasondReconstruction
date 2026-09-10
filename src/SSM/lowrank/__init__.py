"""Low-rank decomposition of a covariance operator.

Turns a kernel into the ``(Psi, Lambda)`` a shape model is made of -
Eq. ``sample_gmm_discrete`` of ``theory/shape_deformation_models.tex``:

    u = Psi Lambda^{1/2} z ,    z ~ N(0, I_L)

    from src.SSM.kernels import Gaussian, Pca
    from src.SSM.lowrank import KernelOperator, Truncation, decompose

    op = KernelOperator(Pca.from_eigendecomposition(Phi, lam) + 0.3 * Gaussian(sigma=15), domain)
    result = decompose(op, "pivoted_cholesky", {"max_rank": 300},
                       truncation=Truncation(variance_threshold=0.99),
                       mass=domain.mass)

Four methods, and the choice is dictated by the operator rather than by taste:

    exact_factor       the kernel already carries its factor (pca, spectral,
                       and any sum of such) - nothing to solve
    dense_eigh         forms the matrix; exact, and the yardstick for the rest.
                       Fetus-sized only: 2.4 GB there, 69 GB on the baby model
    randomized         needs only matmat, so it works on any operator; costs
                       (2q+1) full passes. Best when a product is cheap
    nystroem           m inducing points; the only method that also yields the
                       extension formula, and the cheap route through a
                       Hadamard product
    pivoted_cholesky   touches r kernel columns and knows its own trace error;
                       the default where the matrix cannot be formed

Nothing here imports ``src.SSM.kernels`` beyond a few index helpers: the
algorithms consume a :class:`LinearOperator`, so they decompose any positive
semi-definite operator and are testable against a hand-built matrix with no mesh
involved. :class:`KernelOperator` is the adapter that makes a kernel one.

Two things are easy to get wrong and are handled in one place rather than per
method:

* **The quadrature measure.** On a mesh the eigenproblem is
  ``K M psi = lambda psi`` with ``M`` the barycentric vertex areas, not the
  theory's uniform ``1/n`` (see ``operator.py``). Pass ``mass=`` to
  :func:`decompose` and the symmetrisation and its inverse both happen; the
  resulting basis is ``M``-orthonormal, which ``Decomposition.inner_product``
  records.
* **Truncation.** One :class:`Truncation` object with three criteria, strictest
  wins - the convention ``AtlasConfig`` already uses. The discarded variance is
  reported as ``residual_variance``, which is the ``epsilon`` of
  Eq. ``ppca_gen_model``.
"""

from src.SSM.lowrank.cholesky import PivotedCholesky, pivoted_cholesky_factor
from src.SSM.lowrank.dense import DenseEigh, ExactFactor, cholesky_factor
from src.SSM.lowrank.nystroem import Nystroem, nystrom_extend, select_inducing
from src.SSM.lowrank.operator import (
    APPROXIMATORS,
    Approximator,
    DenseOperator,
    IndexableOperator,
    KernelOperator,
    LinearOperator,
    MassWeighted,
    available_approximators,
    build_approximator,
    decompose,
    mass_weighted,
    mass_weights,
    register_approximator,
    sorted_eigenpairs,
    to_mass_orthonormal,
)
from src.SSM.lowrank.randomized import Randomized
from src.SSM.lowrank.truncation import KEEP_ALL, Decomposition, Truncation

__all__ = [
    # operators
    "LinearOperator",
    "IndexableOperator",
    "KernelOperator",
    "DenseOperator",
    "MassWeighted",
    "mass_weights",
    "mass_weighted",
    "to_mass_orthonormal",
    # interface
    "Approximator",
    "APPROXIMATORS",
    "register_approximator",
    "available_approximators",
    "build_approximator",
    "decompose",
    "sorted_eigenpairs",
    # results
    "Decomposition",
    "Truncation",
    "KEEP_ALL",
    # methods
    "DenseEigh",
    "ExactFactor",
    "cholesky_factor",
    "Randomized",
    "Nystroem",
    "select_inducing",
    "nystrom_extend",
    "PivotedCholesky",
    "pivoted_cholesky_factor",
]
