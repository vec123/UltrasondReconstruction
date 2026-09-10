"""Kernels built from the Laplace-Beltrami spectrum of the reference surface.

``K = sum_i f(mu_i) phi_i phi_i^T`` over the leading eigenpairs of the
Laplace-Beltrami operator, with ``f`` a spectral filter. Because ``mu_i``
measures how oscillatory ``phi_i`` is, a decreasing ``f`` is a smoothness
prior stated in the frequency domain rather than through a distance: low modes
are given variance, high modes are damped, and a sample is smooth by
construction.

Two things distinguish it from the stationary family:

* It is **intrinsic**. The eigenfunctions live on the surface, so the
  correlation between two points follows the geometry rather than the straight
  line through the ambient space - across a cleft, two points that are close in
  R^3 do not move together. A ``gaussian`` on a geodesic domain buys some of
  this; the spectrum buys all of it, and without an all-pairs distance matrix.
* It is **exactly low-rank** by construction, at the number of eigenpairs
  requested. The factor is available with no decomposition to perform, which
  makes it cheap to sum with other kernels.

The price is that it is tabulated: the eigenfunctions exist at the mesh
vertices and nowhere else, so like ``pca`` it offers no analytic part and
cannot itself be evaluated at a new point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Optional

import numpy as np

from src.SSM.kernels.base import (
    Kernel,
    KernelStructure,
    log_params,
    lowrank,
    read_log,
    register_kernel,
    resolve_indices,
)

FILTERS = ("heat", "inverse")


@register_kernel("laplace_beltrami")
@dataclass(frozen=True, eq=False)
class LaplaceBeltrami(Kernel):
    """A spectral prior on the reference surface.

    ``eigenvectors`` is ``(n, k)`` and ``eigenvalues`` ``(k,)``; build them with
    :meth:`from_mesh` rather than by hand. Scalar (``block_dim = 1``), so it
    stands for ``k(p,p') I_3`` and lifts when summed with a matrix-valued
    kernel.

    **Filters.**

    ``heat``
        ``f(mu) = scale * exp(-t mu)`` - the heat kernel at time ``t``. A
        genuine diffusion: ``t`` is the time the surface is smoothed for, and
        larger ``t`` means longer-range, smoother correlation. Well behaved at
        ``mu = 0``, so it needs no guard.

    ``inverse``
        ``f(mu) = scale / (mu + epsilon)^power`` - the thin-plate family, and
        the one that matches the usual "penalise curvature" intuition.
        ``power = 1`` is the membrane energy, ``2`` the bending energy.

        Mind ``mu_0``. The first eigenvalue is zero (its eigenfunction is
        constant), so this filter gives the constant mode variance
        ``scale/epsilon^power`` - i.e. an almost unbounded global translation
        that swamps every other mode. ``epsilon`` bounds it, and
        ``skip_constant`` drops it entirely, which is usually what is wanted
        when a similarity transform is being fitted separately anyway.
    """

    eigenvectors: np.ndarray          # (n, k), M-orthonormal
    eigenvalues: np.ndarray           # (k,), ascending, mu_0 ~ 0
    filter: str = "heat"
    t: float = 1.0                    # heat only
    power: float = 2.0                # inverse only
    epsilon: float = 1e-3             # inverse only
    scale: float = 1.0
    skip_constant: bool = False
    block_dim = 1

    def __post_init__(self):
        vectors = np.ascontiguousarray(np.asarray(self.eigenvectors, dtype=np.float64))
        values = np.ascontiguousarray(np.asarray(self.eigenvalues, dtype=np.float64).reshape(-1))
        if vectors.ndim != 2 or vectors.shape[1] != values.size:
            raise ValueError(
                f"eigenvectors {vectors.shape} and {values.size} eigenvalues do not match"
            )
        if self.filter not in FILTERS:
            raise ValueError(f"filter must be one of {list(FILTERS)}, got {self.filter!r}")
        if self.scale < 0:
            raise ValueError(f"scale must be >= 0, got {self.scale}")
        if self.filter == "heat" and self.t <= 0:
            raise ValueError(f"t must be positive, got {self.t}")
        if self.filter == "inverse" and self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        object.__setattr__(self, "eigenvectors", vectors)
        object.__setattr__(self, "eigenvalues", np.maximum(values, 0.0))

    # -- construction ------------------------------------------------------

    @classmethod
    def from_mesh(cls, points, faces, n_eig: int = 100, **kwargs) -> "LaplaceBeltrami":
        """Solves the eigenproblem on the reference mesh.

        Uses ``compute_spectral_operators_fast``, deliberately, and **not**
        ``compute_spectral_operators``: the latter silently decimates any mesh
        past 10000 vertices and discards its faces, so on the baby model it
        would return eigenvectors indexed against a different vertex set than
        the domain. That mismatch is not detectable downstream - the shapes are
        simply wrong - so it is worth the note.

        The heavy dependencies (``robust_laplacian``, ``pyamg``) are imported
        here rather than at module level, so ``src.SSM.kernels`` stays
        importable without them.
        """
        from src.geometry.spectral_operators import compute_spectral_operators_fast

        points = np.asarray(points, dtype=np.float64)
        _, _, evals, evecs = compute_spectral_operators_fast(
            points, None if faces is None else np.asarray(faces), k_eig=int(n_eig)
        )
        return cls(eigenvectors=np.asarray(evecs), eigenvalues=np.asarray(evals), **kwargs)

    # -- the filter --------------------------------------------------------

    def spectrum(self) -> np.ndarray:
        """``f(mu_i)`` - the variance this kernel assigns to each mode."""
        mu = self.eigenvalues
        if self.filter == "heat":
            weights = np.exp(-self.t * mu)
        else:
            weights = 1.0 / np.power(mu + self.epsilon, self.power)
        weights = self.scale * weights
        if self.skip_constant and weights.size:
            weights = weights.copy()
            weights[0] = 0.0
        return weights

    # -- evaluation --------------------------------------------------------

    @property
    def structure(self) -> KernelStructure:
        return lowrank(int(np.count_nonzero(self.spectrum())))

    def factor(self, domain, rows=None) -> np.ndarray:
        rows = resolve_indices(domain, rows)
        weights = self.spectrum()
        keep = weights > 0
        return self.eigenvectors[np.ix_(rows, np.flatnonzero(keep))] * np.sqrt(weights[keep])[None, :]

    def block(self, domain, rows=None, cols=None) -> np.ndarray:
        Fr = self.factor(domain, rows)
        Fc = self.factor(domain, cols)
        return Fr @ Fc.T

    def matmat(self, domain, V, rows=None, cols=None, chunk_bytes=None):
        Fr = self.factor(domain, rows)
        Fc = self.factor(domain, cols)
        V = np.asarray(V)
        squeeze = V.ndim == 1
        if squeeze:
            V = V[:, None]
        if V.shape[0] != Fc.shape[0]:
            raise ValueError(f"V has {V.shape[0]} rows, expected {Fc.shape[0]}")
        out = Fr @ (Fc.T @ V)
        return out[:, 0] if squeeze else out

    def diag(self, domain, rows=None) -> np.ndarray:
        F = self.factor(domain, rows)
        return np.einsum("ij,ij->i", F, F)

    # -- off-domain --------------------------------------------------------

    def supports_cross(self) -> bool:
        return False

    def analytic_part(self) -> Optional[Kernel]:
        return None

    # -- parameters --------------------------------------------------------

    def free_parameters(self) -> Dict[str, float]:
        if self.filter == "heat":
            return log_params(t=self.t, scale=self.scale)
        return log_params(power=self.power, epsilon=self.epsilon, scale=self.scale)

    def with_parameters(self, theta) -> "LaplaceBeltrami":
        if self.filter == "heat":
            return replace(
                self,
                t=read_log(theta, "t", self.t),
                scale=read_log(theta, "scale", self.scale),
            )
        return replace(
            self,
            power=read_log(theta, "power", self.power),
            epsilon=read_log(theta, "epsilon", self.epsilon),
            scale=read_log(theta, "scale", self.scale),
        )

    def describe(self) -> str:
        detail = f"t={self.t:.4g}" if self.filter == "heat" else (
            f"power={self.power:.4g}, epsilon={self.epsilon:.4g}"
        )
        return (
            f"laplace_beltrami({self.filter}, {detail}, scale={self.scale:.4g}, "
            f"n_eig={self.eigenvalues.size}) [{self.structure}]"
        )
