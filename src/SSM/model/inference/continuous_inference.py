"""Evaluating the model anywhere on the mean shape, not only at its vertices.

Given **any** point - addressed by its coordinates on the mean shape, not
necessarily one of the ``n`` points the model was discretised at - produce the
deformation there. That is what makes a GPMM resolution-independent: fit at a
decimated resolution and reconstruct at full resolution, or transfer a model
between two meshes of the same anatomy.

Named after §Practical notes on continuous inference, the section of
``theory/shape_deformation_models.tex`` it implements, rather than after the
Nystrom *extension*, which turns out to be one case of the operator (see
:class:`KernelRegression`).

**The gap.** §Practical notes says plainly that *"the central object for both
inference methods is k(x,x') whose analytical form is required in either
case."* Every parameterised kernel has that form. ``k_PCA`` does not, and
cannot: ``k_PCA(x*, x') = (1/(N-1)) sum_i u^i(x*) u^i(x')^T`` needs each
subject's deformation *at the new point*, and the subjects are known only at the
mesh vertices. So a data-driven GPMM has no closed-form value off its own
discretisation, and no amount of care with the eigenproblem changes that.

**The fix, from the theory's other half.** A field known at ``n`` points is
extended by Gaussian-process regression - Eq. ``posterior_inference_kernel``.
Regressing the *basis* onto the *analytic part* of the kernel gives an operator
that is exact where the model already knows the answer and smooth everywhere
else, and that is what :class:`KernelRegression` is. It is not a substitute for
the theory's extension formula; it contains it.

Three strategies, registered so the config picks one by name:

    kernel_regression   the above. Default whenever the kernel has an analytic
                        part, which is any model with a parameterised term
    barycentric         piecewise-linear interpolation on the triangulation.
                        C^0, O(1) per query, and the only option for a model
                        whose kernel is purely tabulated
    none                off-domain evaluation is an error
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.linalg as sla

from src.core.registry import TypedRegistry
from src.SSM.kernels.base import expand_indices, resolve_indices

INFERENCE = TypedRegistry("inference")
register_inference = INFERENCE.register


def available_inference() -> List[str]:
    return INFERENCE.names()


def build_inference(name: str, kwargs: Optional[Dict[str, Any]] = None) -> "ContinuousInference":
    return INFERENCE.build(name, kwargs)


class ContinuousInference(ABC):
    """Turns coordinates on the mean shape into a deformation.

    Two-stage by necessity: :meth:`prepare` does work proportional to the model
    (a factorisation, or building a spatial index) and :meth:`basis_at` is then
    cheap per query. The model calls ``prepare`` once, at build time, and holds
    the result.
    """

    name = "?"

    @abstractmethod
    def prepare(self, domain, kernel, basis: np.ndarray) -> "ContinuousInference":
        """Binds to a model. Returns ``self`` so it can be chained onto a build."""

    @abstractmethod
    def basis_at(self, points) -> np.ndarray:
        """``(3M, L)`` - the model's basis evaluated at ``M`` new points."""

    @property
    def is_ready(self) -> bool:
        return getattr(self, "_ready", False)

    def _require_ready(self):
        if not self.is_ready:
            raise RuntimeError(
                f"inference '{self.name}' has not been prepared; call prepare(domain, "
                f"kernel, basis) first (the model builder does this)."
            )

    def deformation_at(self, points, alpha) -> np.ndarray:
        """``(M, 3)`` displacement at each query point - the primary question."""
        alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
        return (self.basis_at(points) @ alpha).reshape(-1, 3)

    def shape_at(self, points, alpha) -> np.ndarray:
        """``(M, 3)`` deformed position: the query point plus its displacement.

        A query point is given by its coordinates *on the mean shape*, so the
        mean at that point is the point itself and the instance is
        ``p + u(p)`` - the discrete ``mean + Phi alpha``, read continuously.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return points + self.deformation_at(points, alpha)

    def describe(self) -> str:
        return f"{self.name}({'ready' if self.is_ready else 'unprepared'})"

    def __repr__(self) -> str:
        return self.describe()


# --------------------------------------------------------------------------

@register_inference("kernel_regression")
class KernelRegression(ContinuousInference):
    """GP regression of the basis onto the kernel's analytic part.

    With ``k_s = kernel.analytic_part()`` and ``Z`` a set of ``m`` domain
    points,

        E(x*) = k_s(x*, Z) (K_s(Z,Z) + sigma^2 I)^{-1}
        Psi(x*) = E(x*) Psi_Z

    Four properties make this the right construction rather than one option
    among several:

    1. **Exact where the model already knows the answer** - subject to a caveat
       worth stating up front. At ``sigma^2 = 0`` and ``Z`` the whole domain,
       ``k_s(p_j, Z) K_s^{-1} = e_j^T``, so ``Psi(p_j) = Psi_j`` identically and
       the extension reproduces the discretisation it came from.

       That holds *in exact arithmetic and while ``K_s`` is invertible*, which
       is a real restriction rather than a formality: a Gaussian whose length
       scale is several times the point spacing is numerically singular, and
       that is exactly the regime a smoothness prior is chosen for. When the
       Cholesky fails the solve falls back to least squares, the operator
       **smooths instead of interpolating**, and :attr:`interpolates` is set to
       ``False`` so the caller can see it. Neither answer is wrong; they are
       different, and which one is in force should not have to be inferred from
       a reconstruction that does not match the model.
    2. **It contains the Nystrom extension.** If ``Psi``'s columns are
       eigenvectors of ``K_s`` with eigenvalues ``lambda_i``, then
       ``K_s^{-1} psi_i = psi_i / lambda_i`` and the formula collapses to
       ``psi_i(x*) = k_s(x*)^T psi_i / lambda_i`` - §Practical notes verbatim,
       which is also what ``lowrank.nystrom_extend`` computes. One operator, two
       readings.
    3. **The result is as smooth as ``k_s``.** ``Psi(x*) = k_s(x*, Z) A`` is a
       finite linear combination of ``k_s(z_j, .)``, so a Gaussian ``k_s`` gives
       an analytic basis. Piecewise-linear interpolation cannot; that difference
       is the reason this is the default.
    4. **It is the RKHS representer form** ``u(x) = sum_j k(x, z_j) a_j`` of
       §RKHS Parameterisation - the same function space LDDMM parameterises its
       velocity field in.

    **Why the analytic part and not the whole kernel.** Regressing onto a kernel
    containing ``pca`` would need ``k_PCA(x*, .)``, which is the value being
    solved for. ``Sum(pca, gaussian).analytic_part()`` is ``gaussian``, and
    using it is what keeps the construction non-circular.

    **Inducing points.** ``(K_s + sigma^2 I)^{-1}`` over the whole domain is an
    ``n x n`` solve - 7.7 GB at the baby model's 31027 vertices. ``n_inducing``
    regresses from a covering subset instead, chosen by farthest-point sampling
    so it spans the surface rather than clustering where the mesh is dense.
    ``None`` uses every point, which is exact and fetus-sized. Accuracy is set
    by whether ``m`` points resolve the kernel's length scale, so a short
    ``sigma`` needs more of them.

    ``sigma_sq`` is relative to the mean diagonal of ``K_s``, so it means the
    same thing whatever units the model is in. It is a smoothing knob, not only
    a numerical guard: ``0`` interpolates the discrete basis exactly, larger
    values smooth mesh-scale noise out of the extended modes.
    """

    def __init__(self, sigma_sq: float = 1e-6, n_inducing: Optional[int] = 1024,
                 selection: str = "farthest", seed: Optional[int] = None):
        if sigma_sq < 0:
            raise ValueError(f"sigma_sq must be >= 0, got {sigma_sq}")
        if n_inducing is not None and n_inducing < 1:
            raise ValueError(f"n_inducing must be >= 1 or None, got {n_inducing}")
        self.sigma_sq = float(sigma_sq)
        self.n_inducing = None if n_inducing is None else int(n_inducing)
        self.selection = selection
        self.seed = seed
        #: Set by :meth:`prepare`: True when the solve was a genuine inverse and
        #: the operator reproduces the basis at the sample points, False when it
        #: degraded to least squares and smooths instead. ``None`` until prepared.
        self.interpolates: Optional[bool] = None
        self._ready = False

    def prepare(self, domain, kernel, basis) -> "KernelRegression":
        analytic = kernel.analytic_part()
        if analytic is None:
            raise ValueError(
                "this model's kernel is entirely tabulated (pca / laplace_beltrami / "
                "localized without a weight_fn), so it has no value at a new point to "
                "regress against. Add a parameterised term - a gaussian or matern, even "
                "a small one - or use inference 'barycentric', which interpolates on the "
                "triangulation instead and is C0."
            )

        basis = np.ascontiguousarray(np.asarray(basis, dtype=np.float64))
        all_points = resolve_indices(domain, None)

        if self.n_inducing is None or self.n_inducing >= all_points.size:
            inducing = all_points
        else:
            from src.SSM.lowrank.operator import KernelOperator
            from src.SSM.lowrank.nystroem import select_inducing

            inducing = select_inducing(
                KernelOperator(analytic, domain), self.n_inducing, self.selection, self.seed
            )

        K_zz = np.asarray(analytic.block(domain, inducing, inducing), dtype=np.float64)
        K_zz = 0.5 * (K_zz + K_zz.T)
        if self.sigma_sq:
            K_zz = K_zz + self.sigma_sq * float(np.trace(K_zz)) / K_zz.shape[0] * np.eye(K_zz.shape[0])

        Phi_z = basis[expand_indices(inducing, 3)]          # (3m, L)
        n_modes = Phi_z.shape[1]

        if analytic.block_dim == 1:
            # A scalar kernel acts on each coordinate independently, so the
            # solve is m x m with 3L right-hand sides rather than 3m x 3m with
            # L - nine times less work for the same answer.
            rhs = Phi_z.reshape(inducing.size, 3 * n_modes)
        else:
            rhs = Phi_z

        self._weights, self.interpolates = _solve_psd(K_zz, rhs)
        self._kernel = analytic
        self._domain = domain
        self._inducing = inducing
        self._n_modes = n_modes
        self._ready = True
        return self

    def basis_at(self, points) -> np.ndarray:
        self._require_ready()
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        cross = np.asarray(
            self._kernel.cross(self._domain, points, self._inducing), dtype=np.float64
        )
        out = cross @ self._weights

        if self._kernel.block_dim == 1:
            # (M, 3L) with column c*L + l  ->  (3M, L) vertex-major.
            return out.reshape(points.shape[0], 3, self._n_modes).reshape(-1, self._n_modes)
        return out

    def describe(self) -> str:
        if not self.is_ready:
            return f"kernel_regression(sigma_sq={self.sigma_sq:.3g}, unprepared)"
        return (
            f"kernel_regression({self._kernel.describe()}, m={self._inducing.size}, "
            f"sigma_sq={self.sigma_sq:.3g}, "
            f"{'interpolating' if self.interpolates else 'smoothing'})"
        )


@register_inference("barycentric")
class Barycentric(ContinuousInference):
    """Piecewise-linear interpolation of the basis on the triangulation.

    The basis is a per-vertex vector field on a triangulated surface, so it is
    already defined between the vertices in the finite-element sense: locate the
    triangle a query point falls in and combine its three rows by the
    barycentric weights. Exact at a vertex, and ``O(log F)`` per query once the
    cell locator is built.

    Use it when the kernel is entirely tabulated and there is nothing to regress
    against. Its limitation is regularity, not accuracy: the result is only C^0,
    so a sample has creases along the edges of the mesh. That is the honest
    regularity of a triangulated population - but if the model *has* a smooth
    term, ``kernel_regression`` gives a smooth answer for the same information,
    and is preferable.

    Queries go through VTK one at a time, so this is the cheaper strategy per
    query but not a vectorised one; a very large batch is better served by
    ``kernel_regression``.

    Accuracy at a vertex is ~1e-7 relative rather than machine precision, and
    the reason is worth recording so it is not chased: ``src.vtk.create`` casts
    points to float32 on the way into VTK, so the cell locator answers in
    float32 and the barycentric weights inherit that. It is far below any
    modelling error here, but it is a floor, not noise - a check demanding 1e-12
    from this strategy will fail forever.
    """

    def __init__(self, snap_tolerance: float = 0.0):
        self.snap_tolerance = float(snap_tolerance)
        self._ready = False

    def prepare(self, domain, kernel, basis) -> "Barycentric":
        if getattr(domain, "faces", None) is None:
            raise ValueError(
                "inference 'barycentric' interpolates on a triangulation and the domain "
                "has no faces. Supply them, or use 'kernel_regression'."
            )
        import vtk

        from src.vtk.create import create_polydata

        self._basis = np.ascontiguousarray(np.asarray(basis, dtype=np.float64))
        self._n_modes = self._basis.shape[1]
        self._polydata = create_polydata(
            np.ascontiguousarray(domain.points, dtype=np.float64), domain.faces
        )
        self._locator = vtk.vtkCellLocator()
        self._locator.SetDataSet(self._polydata)
        self._locator.BuildLocator()
        self._vtk = vtk
        self._ready = True
        return self

    def basis_at(self, points) -> np.ndarray:
        self._require_ready()
        vtk = self._vtk
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)

        blocks = self._basis.reshape(-1, 3, self._n_modes)
        out = np.empty((3 * points.shape[0], self._n_modes), dtype=np.float64)
        cell = vtk.vtkGenericCell()
        closest = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)

        for i, point in enumerate(points):
            self._locator.FindClosestPoint(list(point), closest, cell, cell_id, sub_id, dist2)
            ids = [cell.GetPointIds().GetId(k) for k in range(3)]
            corners = np.array([cell.GetPoints().GetPoint(k) for k in range(3)])
            weights = _barycentric(np.asarray(closest), corners)
            out[3 * i : 3 * i + 3] = np.tensordot(weights, blocks[ids], axes=(0, 0))

        return out

    def describe(self) -> str:
        if not self.is_ready:
            return "barycentric(unprepared)"
        return f"barycentric(faces={self._polydata.GetNumberOfCells()})"


@register_inference("none")
class NoInference(ContinuousInference):
    """Off-domain evaluation is an error. The Null Object of this axis.

    The right default for a model loaded from a file with no kernel attached:
    it can still be sampled and fitted at its own vertices, and only says so
    when asked for something it genuinely cannot answer.
    """

    def prepare(self, domain, kernel, basis) -> "NoInference":
        self._ready = True
        return self

    def basis_at(self, points) -> np.ndarray:
        raise NotImplementedError(
            "this model has inference 'none', so it is defined only at its own "
            "vertices. Set inference to 'kernel_regression' (needs a parameterised "
            "term in the kernel) or 'barycentric' (needs a triangulation)."
        )

    def describe(self) -> str:
        return "none"


# --------------------------------------------------------------------------

def _barycentric(point: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Barycentric weights of ``point`` in the triangle ``corners`` (3, 3).

    Computed here rather than through ``vtkTriangle.BarycentricCoords``, whose
    Python overload resolution picks a 2-D signature for 3-vectors on at least
    some VTK builds. Six lines of numpy is both portable and easier to check.

    A degenerate triangle has no barycentric frame; its corners are weighted
    equally, which puts the answer at the centroid rather than producing a nan
    that would travel silently into the reconstructed shape.
    """
    e1 = corners[1] - corners[0]
    e2 = corners[2] - corners[0]
    rel = point - corners[0]
    d11, d12, d22 = e1 @ e1, e1 @ e2, e2 @ e2
    r1, r2 = rel @ e1, rel @ e2
    denominator = d11 * d22 - d12 * d12
    if abs(denominator) < 1e-30:
        return np.full(3, 1.0 / 3.0)
    b1 = (d22 * r1 - d12 * r2) / denominator
    b2 = (d11 * r2 - d12 * r1) / denominator
    return np.array([1.0 - b1 - b2, b1, b2])


def _solve_psd(A: np.ndarray, B: np.ndarray):
    """``(A^{-1} B, exact)`` for a symmetric PSD ``A``, degrading gracefully.

    Cholesky when it can, least squares when it cannot - and the caller is told
    which, because the difference is not cosmetic.

    A smooth kernel's inducing block is ill-conditioned by construction: a
    Gaussian whose length scale is several times the point spacing is
    numerically singular for any interesting ``m``, which is precisely the
    regime a smoothness prior is chosen for. When that happens the least-squares
    solution is the minimum-norm one, and the regression **smooths rather than
    interpolates** - it no longer reproduces the discrete basis at the sample
    points. That is a defensible answer, often a desirable one, but it is a
    different answer, and a caller that believes it is interpolating should find
    out here rather than from a reconstruction that does not match the model.
    """
    try:
        solved = sla.cho_solve(sla.cho_factor(A, lower=True, check_finite=False), B,
                               check_finite=False)
        return solved, True
    except (sla.LinAlgError, np.linalg.LinAlgError):
        return np.linalg.lstsq(A, B, rcond=None)[0], False
