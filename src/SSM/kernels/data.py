"""Kernels that carry measured statistics rather than a formula.

The ``k_PCA`` of §Gaussian Process Morphable Models:

    k_PCA(p, p') = (1/(N-1)) sum_i u^i(p) u^i(p')^T        i.e.   K = (1/(N-1)) U U^T

which is the empirical covariance of the population and is what makes a GPMM a
*statistical* shape model rather than only a smoothness prior.

**The population is not needed to build it.** Only the second moment enters, and
an existing PCA model already stores that in factored form: ``K = Phi Lambda
Phi^T``, so ``basis = Phi sqrt(Lambda)`` and the kernel is exact with no
decomposition to perform. This is what lets the baby model - whose individual
subjects are not in the repository - become a GPMM anyway. It is also the
cheapest possible input to the low-rank layer, since the factor *is* the answer.

**What it cannot do** is evaluate off the sample points. ``k_PCA(x*, x')``
needs each subject's deformation at ``x*``, and the subjects are known only at
the mesh vertices, so :meth:`supports_cross` is ``False`` and
:meth:`analytic_part` is ``None``. That is a property of empirical covariance,
not a missing feature: continuous evaluation goes through the extension
operator in ``src/SSM/model/extension.py``, which regresses onto the *analytic*
part of the surrounding kernel. It is precisely why a GPMM meant to be sampled
at new resolutions needs at least one parameterised term.

**Rank.** With ``N`` subjects the kernel has rank at most ``N-1`` - 163 on the
baby model, 183 on the fetus atlas. Anything outside that span is unreachable,
and every mode is global: no mode moves the nose without moving the jaw. Adding
or multiplying a parameterised kernel is how that is escaped, which is the whole
point of the algebra in ``algebra.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Optional

import numpy as np

from src.SSM.kernels.base import (
    DIAGONAL,
    Kernel,
    KernelStructure,
    expand_indices,
    log_params,
    lowrank,
    read_log,
    register_kernel,
    resolve_indices,
)


@register_kernel("pca")
@dataclass(frozen=True, eq=False)
class Pca(Kernel):
    """``scale * B B^T`` with ``B`` the population's factored covariance.

    ``basis`` is ``(3n, r)`` in the vertex-major layout used everywhere in
    ``src/SSM`` - row ``3i + c`` is component ``c`` of vertex ``i``. Build it
    with :meth:`from_eigendecomposition` (an existing PCA model) or
    :meth:`from_population` (subject shapes in correspondence) rather than by
    hand; both put the ``1/(N-1)`` in the right place, which is the easiest
    thing to get wrong here.

    ``scale`` multiplies the whole covariance and is the parameter that matters
    when this kernel is one term of a sum: it sets how much of the total
    variance the data explains against the smooth terms. It is separate from
    ``basis`` so that :meth:`with_parameters` costs nothing - a hyperparameter
    search re-scales a float, it does not rebuild a 93081-row matrix.

    Registered so it can appear in a spec tree by name, but ``basis`` comes from
    the model source, not from YAML: ``src/SSM/model/builder.py`` injects it
    when it resolves the spec.
    """

    basis: np.ndarray
    scale: float = 1.0
    block_dim = 3

    def __post_init__(self):
        basis = np.ascontiguousarray(np.asarray(self.basis, dtype=np.float64))
        if basis.ndim != 2:
            raise ValueError(f"basis must be 2-D (3n, r), got shape {basis.shape}")
        if basis.shape[0] % 3:
            raise ValueError(f"basis has {basis.shape[0]} rows, which is not a multiple of 3")
        if self.scale < 0:
            raise ValueError(f"scale must be >= 0, got {self.scale}")
        object.__setattr__(self, "basis", basis)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_eigendecomposition(cls, eigenfunctions, eigenvalues, scale: float = 1.0) -> "Pca":
        """From a model that is already decomposed: ``B = Phi sqrt(Lambda)``.

        Exact and free - no covariance is formed and nothing is estimated. The
        rank is the number of modes kept, so a model truncated at build time
        stays truncated here.

        Pass the eigenvalues as ``src.SSM.model.load.load_mat_ssm`` returns
        them, **not** the raw ``.mat`` field. The file stores them in a
        different inner product from the one its eigenfunctions are normalised
        in, and ``rescale_eigenvalues`` corrects that by a factor of
        ``n_vertices``. Skipping it leaves this kernel ~3e4 too small against
        any parameterised term it is summed with, which presents as "the smooth
        kernel does nothing".
        """
        Phi = np.asarray(eigenfunctions, dtype=np.float64)
        lam = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
        if Phi.shape[1] != lam.size:
            raise ValueError(f"{lam.size} eigenvalues for {Phi.shape[1]} eigenfunctions")
        if np.any(lam < 0):
            raise ValueError("eigenvalues must be non-negative; got a negative variance")
        return cls(basis=Phi * np.sqrt(lam)[None, :], scale=scale)

    @classmethod
    def from_population(
        cls,
        displacements,
        scale: float = 1.0,
        n_components: Optional[int] = None,
        already_centred: bool = False,
    ) -> "Pca":
        """From ``(3n, N)`` subject displacements in correspondence.

        ``K = (1/(N-1)) U U^T`` factors as ``B B^T`` with ``B = U/sqrt(N-1)``
        directly - no SVD is needed unless ``n_components`` asks for a
        truncation, in which case a thin SVD supplies the leading modes.

        ``already_centred`` skips the mean subtraction for a caller that has
        already done it. Centring the wrong way round is the other classic error
        in this constructor: the covariance is of the deviations from the mean
        shape, so an uncentred ``U`` adds a rank-one term that makes the mean
        itself look like the dominant mode of variation.
        """
        U = np.asarray(displacements, dtype=np.float64)
        if U.ndim != 2 or U.shape[0] % 3:
            raise ValueError(f"displacements must be (3n, N); got shape {U.shape}")
        n_subjects = U.shape[1]
        if n_subjects < 2:
            raise ValueError(f"need at least 2 subjects to form a covariance, got {n_subjects}")

        if not already_centred:
            U = U - U.mean(axis=1, keepdims=True)

        B = U / math.sqrt(n_subjects - 1)

        if n_components is not None and n_components < B.shape[1]:
            left, singular, _ = np.linalg.svd(B, full_matrices=False)
            keep = int(n_components)
            B = left[:, :keep] * singular[:keep][None, :]

        return cls(basis=B, scale=scale)

    # -- evaluation --------------------------------------------------------

    @property
    def structure(self) -> KernelStructure:
        return lowrank(self.basis.shape[1])

    @property
    def n_vertices(self) -> int:
        return self.basis.shape[0] // 3

    def factor(self, domain, rows=None) -> np.ndarray:
        rows = resolve_indices(domain, rows)
        B = self.basis[expand_indices(rows, 3)]
        return B if self.scale == 1.0 else B * math.sqrt(self.scale)

    def block(self, domain, rows=None, cols=None) -> np.ndarray:
        rows = resolve_indices(domain, rows)
        cols = resolve_indices(domain, cols)
        Br = self.basis[expand_indices(rows, 3)]
        Bc = self.basis[expand_indices(cols, 3)]
        return self.scale * (Br @ Bc.T)

    def matmat(self, domain, V, rows=None, cols=None, chunk_bytes=None):
        """Two thin products, never the ``3n x 3n`` matrix.

        ``(B_r B_c^T) V = B_r (B_c^T V)`` costs ``O(3n r k)`` against ``O(9n^2 k)``
        for the formed matrix - on the baby model, 30 MFLOP instead of 8.7 TFLOP.
        """
        rows = resolve_indices(domain, rows)
        cols = resolve_indices(domain, cols)
        V = np.asarray(V)
        squeeze = V.ndim == 1
        if squeeze:
            V = V[:, None]

        Br = self.basis[expand_indices(rows, 3)]
        Bc = self.basis[expand_indices(cols, 3)]
        if V.shape[0] != Bc.shape[0]:
            raise ValueError(f"V has {V.shape[0]} rows, expected {Bc.shape[0]}")

        out = self.scale * (Br @ (Bc.T @ V))
        return out[:, 0] if squeeze else out

    def diag(self, domain, rows=None) -> np.ndarray:
        rows = resolve_indices(domain, rows)
        B = self.basis[expand_indices(rows, 3)]
        return self.scale * np.einsum("ij,ij->i", B, B)

    # -- off-domain --------------------------------------------------------

    def supports_cross(self) -> bool:
        return False

    def analytic_part(self) -> Optional[Kernel]:
        return None

    # -- parameters --------------------------------------------------------

    def free_parameters(self) -> Dict[str, float]:
        return log_params(scale=self.scale)

    def with_parameters(self, theta) -> "Pca":
        return replace(self, scale=read_log(theta, "scale", self.scale))

    def describe(self) -> str:
        return (
            f"pca(rank={self.basis.shape[1]}, n_vertices={self.n_vertices}, "
            f"scale={self.scale:.4g}) [{self.structure}]"
        )


@register_kernel("diagonal_field")
@dataclass(frozen=True, eq=False)
class DiagonalField(Kernel):
    """Per-vertex variance: ``k(p_i, p_j) = values[i] * delta_ij``.

    A white kernel whose strength varies over the surface - so a region can be
    given its own independent freedom without any correlation to its
    neighbours. Its use is as the ``w`` in a heteroscedastic noise term, or to
    let a badly-imaged region move without the model claiming to know how.

    Neighbouring vertices are uncorrelated under it, so on its own it generates
    per-vertex noise rather than a deformation. To make a region move
    *coherently*, use ``algebra.Localized`` around a smooth kernel instead -
    that is almost always what is wanted.

    Like ``white``, it has no informative value off the sample points, so it
    contributes no analytic part.
    """

    values: np.ndarray
    block_dim = 1

    def __post_init__(self):
        values = np.ascontiguousarray(np.asarray(self.values, dtype=np.float64).reshape(-1))
        if np.any(values < 0):
            raise ValueError("values are variances and must be >= 0")
        object.__setattr__(self, "values", values)

    @property
    def structure(self) -> KernelStructure:
        return DIAGONAL

    def block(self, domain, rows=None, cols=None) -> np.ndarray:
        rows = resolve_indices(domain, rows)
        cols = resolve_indices(domain, cols)
        mask = (rows[:, None] == cols[None, :]).astype(np.float64)
        return mask * self.values[cols][None, :]

    def diag(self, domain, rows=None) -> np.ndarray:
        rows = resolve_indices(domain, rows)
        return self.values[rows].copy()

    def matmat(self, domain, V, rows=None, cols=None, chunk_bytes=None):
        rows = resolve_indices(domain, rows)
        cols = resolve_indices(domain, cols)
        V = np.asarray(V)
        squeeze = V.ndim == 1
        if squeeze:
            V = V[:, None]

        # Only the vertices appearing in both index sets contribute anything.
        lookup = {int(c): i for i, c in enumerate(cols)}
        out = np.zeros((rows.size, V.shape[1]), dtype=np.float64)
        for i, r in enumerate(rows):
            j = lookup.get(int(r))
            if j is not None:
                out[i] = self.values[r] * V[j]
        return out[:, 0] if squeeze else out

    def analytic_part(self) -> Optional[Kernel]:
        return None

    def describe(self) -> str:
        return (
            f"diagonal_field(n={self.values.size}, "
            f"range=[{self.values.min():.4g}, {self.values.max():.4g}]) [{self.structure}]"
        )
