"""The generated field: the smoothest Gaussian-RKHS vector field that meets a set of linear constraints.

Every constraint a mode can state is a linear functional of the velocity
field ``v``, of one of two kinds:

    value   d . v(y) = c              displacement (3 rows per site), anchor
                                      (v = 0, 3 rows), slide (n . v = 0, 1 row)
    strain  <S, sym(Dv)(y)> = c       prescribed stretch in a local frame
                                      (6 rows per site), rigid (sym(Dv) = 0)

In the RKHS of the Gaussian kernel ``k(x, y) = exp(-|x - y|^2 / 2 sigma^2) I``,
the minimal-norm field meeting functionals ``L_k(v) = c_k`` is
``v = sum_k lambda_k w_k`` with ``w_k`` the representer of ``L_k`` and
``(G + diag(nu_k)) lambda = c``, ``G_kl = L_k(w_l)``. This is the explicit,
static form of IMODAL's implicit modules of order 0 and 1 - the same
constraints, solved once for the current geometry. With ``r = x - y``:

    value representer    w(x) = k(x, y) d
    strain representer   w(x) = k(x, y) / sigma^2 * S r

    G[value i, value j]   = k_ij d_i . d_j
    G[value i, strain j]  = -k_ij / sigma^2 * d_i^T S_j (y_j - y_i)
    G[strain i, strain j] = k_ij (<S_i, S_j>_F / sigma^2 - (S_i r).(S_j r) / sigma^4),  r = y_j - y_i

Strain rows are scaled by sigma internally (``S -> sigma S``, ``c -> sigma c``),
which puts every block of G at O(1) and keeps the solve well conditioned
whatever the template's units. ``nu`` softens all constraints slightly, so
conflicting or duplicated rows (an anchor on a displacement site) still solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np
import scipy.linalg

if TYPE_CHECKING:
    from src.deformation_modes.symmetry import MirrorMap

# Orthonormal basis (Frobenius) of symmetric 3x3 matrices: the 3 axial
# stretches, then the 3 shears.
_S = 1.0 / np.sqrt(2.0)
SYM_BASIS = np.array([
    [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
    [[0, _S, 0], [_S, 0, 0], [0, 0, 0]],
    [[0, 0, _S], [0, 0, 0], [_S, 0, 0]],
    [[0, 0, 0], [0, 0, _S], [0, _S, 0]],
], dtype=np.float64)


def _as_points(points) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points


@dataclass(frozen=True)
class ValueConstraint:
    """Rows ``directions[k] . v(points[k]) = targets[k]``."""

    points: np.ndarray       # (m, 3)
    directions: np.ndarray   # (m, 3)
    targets: np.ndarray      # (m,)
    weight: float = 1.0
    label: str = "value"

    def __post_init__(self):
        object.__setattr__(self, "points", _as_points(self.points))
        object.__setattr__(self, "directions", _as_points(self.directions))
        object.__setattr__(self, "targets", np.asarray(self.targets, np.float64).reshape(-1))
        m = len(self.points)
        if len(self.directions) != m or len(self.targets) != m:
            raise ValueError(f"{self.label}: {m} points, {len(self.directions)} directions, "
                             f"{len(self.targets)} targets - one of each per row")
        if self.weight <= 0:
            raise ValueError(f"{self.label}: weight must be > 0, got {self.weight}")


@dataclass(frozen=True)
class StrainConstraint:
    """Rows ``<matrices[k], sym(Dv)(points[k])>_F = targets[k]``."""

    points: np.ndarray       # (m, 3)
    matrices: np.ndarray     # (m, 3, 3), symmetric
    targets: np.ndarray      # (m,)
    weight: float = 1.0
    label: str = "strain"

    def __post_init__(self):
        object.__setattr__(self, "points", _as_points(self.points))
        object.__setattr__(self, "matrices", np.asarray(self.matrices, np.float64).reshape(-1, 3, 3))
        object.__setattr__(self, "targets", np.asarray(self.targets, np.float64).reshape(-1))
        m = len(self.points)
        if len(self.matrices) != m or len(self.targets) != m:
            raise ValueError(f"{self.label}: {m} points, {len(self.matrices)} matrices, "
                             f"{len(self.targets)} targets - one of each per row")
        if not np.allclose(self.matrices, np.swapaxes(self.matrices, 1, 2)):
            raise ValueError(f"{self.label}: strain matrices must be symmetric")
        if self.weight <= 0:
            raise ValueError(f"{self.label}: weight must be > 0, got {self.weight}")


Constraint = Union[ValueConstraint, StrainConstraint]


# -- constraint builders --------------------------------------------------------


def displacement(points, vectors, weight: float = 1.0, label: str = "displacement") -> ValueConstraint:
    """``v(p) = vector`` at each point (3 rows per point)."""
    points, vectors = _as_points(points), _as_points(vectors)
    if len(points) != len(vectors):
        raise ValueError(f"{label}: {len(points)} points but {len(vectors)} vectors")
    return ValueConstraint(np.repeat(points, 3, axis=0), np.tile(np.eye(3), (len(points), 1)),
                           vectors.reshape(-1), weight, label)


def anchors(points, weight: float = 1.0, label: str = "anchor") -> ValueConstraint:
    """``v(p) = 0`` at each point: the region does not move."""
    points = _as_points(points)
    return displacement(points, np.zeros_like(points), weight, label)


def sliding(points, normals, weight: float = 1.0, label: str = "slide") -> ValueConstraint:
    """``n . v(p) = 0`` at each point: free to slide along the surface, never to bulge out of it."""
    normals = _as_points(normals)
    normals = normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    return ValueConstraint(points, normals, np.zeros(len(normals)), weight, label)


def strain(points, frames, stretch, weight: float = 1.0, label: str = "strain") -> StrainConstraint:
    """Prescribed stretch rate along each frame axis, no shear (6 rows per point).

    Args:
        points: (m, 3) sites.
        frames: (m, 3, 3) local frames, columns = axes (see ``frames.local_frames``).
        stretch: (3,) or (m, 3) stretch rate along each axis (positive stretches,
            negative compresses, 0 holds that axis).
    """
    points = _as_points(points)
    frames = np.asarray(frames, np.float64).reshape(-1, 3, 3)
    stretch = np.broadcast_to(np.asarray(stretch, np.float64), (len(points), 3))
    matrices = np.einsum("mab,jbc,mdc->mjad", frames, SYM_BASIS, frames)        # R E_j R^T
    targets = np.concatenate([stretch, np.zeros((len(points), 3))], axis=1)     # shears held at 0
    return StrainConstraint(np.repeat(points, 6, axis=0), matrices.reshape(-1, 3, 3),
                            targets.reshape(-1), weight, label)


def rigid(points, weight: float = 1.0, label: str = "rigid") -> StrainConstraint:
    """``sym(Dv)(p) = 0`` at each point: may translate or rotate as a piece, cannot stretch or bend."""
    points = _as_points(points)
    return StrainConstraint(np.repeat(points, 6, axis=0), np.tile(SYM_BASIS, (len(points), 1, 1)),
                            np.zeros(6 * len(points)), weight, label)


def thin(indices: np.ndarray, positions: np.ndarray, spacing: float,
         max_points: Optional[int] = None, mirror: Optional["MirrorMap"] = None) -> np.ndarray:
    """An even sample of ``indices`` no two of which are closer than ``spacing``.

    Constraint points much closer together than the kernel width make the
    Gram matrix nearly singular - the field cannot resolve detail finer than
    sigma, so the extra rows only add ill-conditioning (and wild, huge
    solutions). Use ``spacing`` ~ sigma / 2. Greedy farthest-point sampling,
    started from the first index.

    Args:
        indices: vertex indices to sample.
        positions: (N, 3) vertex positions.
        spacing: minimum distance between kept points.
        max_points: optional hard cap on the number kept.
        mirror: sample only the left half (signed distance >= 0 from
            ``mirror.plane``) and add the mirror images, so a symmetric region
            yields a symmetric sample - and a symmetric mode a symmetric field.

    Returns:
        Sorted unique vertex indices.
    """
    indices = np.unique(np.asarray(indices, dtype=np.int64))
    positions = np.asarray(positions, float)
    if mirror is not None:
        left = indices[mirror.plane.signed_distance(positions[indices]) >= 0]
        return mirror.symmetric(thin(left, positions, spacing, max_points))
    if len(indices) == 0:
        return indices
    points = positions[indices]
    limit = len(indices) if max_points is None else min(int(max_points), len(indices))
    chosen = [0]
    nearest = np.linalg.norm(points - points[0], axis=1)
    while len(chosen) < limit:
        candidate = int(np.argmax(nearest))
        if nearest[candidate] < spacing:
            break
        chosen.append(candidate)
        nearest = np.minimum(nearest, np.linalg.norm(points - points[candidate], axis=1))
    return np.sort(indices[chosen])


# -- the field -------------------------------------------------------------------


def _gauss(sq_dist: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * sq_dist / sigma ** 2)


@dataclass(frozen=True)
class KernelExpansion:
    """A solved field in closed form, one term per distinct constraint point:

        v(x) = sum_c exp(-|x - c|^2 / 2 sigma^2) (a_c + M_c (x - c))

    The value rows at a point sum into ``a_c``, its strain rows into ``M_c``.
    This is everything needed to evaluate the field anywhere - what training
    stores and carries into its own frame, without the constraints or a solver.
    """

    sigma: float
    centers: np.ndarray      # (C, 3)
    vectors: np.ndarray      # (C, 3)    a_c
    matrices: np.ndarray     # (C, 3, 3) M_c

    def __call__(self, points, chunk: int = 1024) -> np.ndarray:
        """(n, 3) field values at ``points``."""
        points = _as_points(points)
        out = np.zeros_like(points)
        for start in range(0, len(points), chunk):
            r = points[start:start + chunk, None, :] - self.centers[None, :, :]       # (n, C, 3)
            k = _gauss((r ** 2).sum(-1), self.sigma)                                  # (n, C)
            out[start:start + chunk] = k @ self.vectors + np.einsum("nc,cab,ncb->na", k, self.matrices, r)
        return out

    def transported(self, scale: float, rotation: np.ndarray, translation: np.ndarray) -> "KernelExpansion":
        """The same field in the frame ``x' = scale * rotation @ x + translation``.

        A displacement field moves with the frame - ``v'(x') = scale * R v(x)`` -
        and stays a Gaussian expansion: centres move, sigma scales with the
        frame, ``a -> s R a`` and ``M -> R M R^T``.
        """
        rotation = np.asarray(rotation, float)
        return KernelExpansion(
            sigma=self.sigma * scale,
            centers=scale * self.centers @ rotation.T + np.asarray(translation, float),
            vectors=scale * self.vectors @ rotation.T,
            matrices=np.einsum("ab,cbd,ed->cae", rotation, self.matrices, rotation),
        )

    def scaled(self, gain: float) -> "KernelExpansion":
        """``gain`` times this field."""
        return KernelExpansion(self.sigma, self.centers, gain * self.vectors, gain * self.matrices)


class KernelField:
    """The minimal-norm Gaussian-RKHS field meeting ``constraints``; callable on any points."""

    def __init__(self, sigma: float, constraints: Sequence[Constraint], nu: float = 1e-4):
        """Assemble and solve.

        Args:
            sigma: kernel width, in mesh units - the length over which the field
                varies; constraints closer than ~sigma interact strongly.
            constraints: value and strain constraints, in any order.
            nu: softness, relative to the mean diagonal of G. Each constraint's
                own ``weight`` divides it, so weight 10 holds a set 10x harder.
                Besides reconciling conflicting rows it bounds the solution
                when constraint points crowd closer than sigma (see :func:`thin`).

        Raises:
            ValueError: sigma <= 0 or no constraint rows.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.sigma = float(sigma)
        values = [c for c in constraints if isinstance(c, ValueConstraint)]
        strains = [c for c in constraints if isinstance(c, StrainConstraint)]
        self.constraints: List[Constraint] = values + strains

        cat = lambda arrays, shape: np.concatenate(arrays) if arrays else np.zeros(shape)
        self._yv = cat([c.points for c in values], (0, 3))
        self._dv = cat([c.directions for c in values], (0, 3))
        cv = cat([c.targets for c in values], (0,))
        wv = cat([np.full(len(c.targets), c.weight) for c in values], (0,))
        self._ys = cat([c.points for c in strains], (0, 3))
        self._ss = cat([c.matrices for c in strains], (0, 3, 3)) * self.sigma     # scaled rows
        cs = cat([c.targets for c in strains], (0,)) * self.sigma
        ws = cat([np.full(len(c.targets), c.weight) for c in strains], (0,))
        if len(cv) + len(cs) == 0:
            raise ValueError("a KernelField needs at least one constraint row")

        gram = self._gram()
        c = np.concatenate([cv, cs])
        weights = np.concatenate([wv, ws])
        regulariser = nu * float(np.mean(np.diag(gram))) / weights
        system = gram + np.diag(regulariser)
        try:
            lam = scipy.linalg.solve(system, c, assume_a="pos")
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
            lam = np.linalg.lstsq(system, c, rcond=None)[0]
        self._lam_v, self._lam_s = lam[:len(cv)], lam[len(cv):]
        #: Squared RKHS norm of the field - the "cost" of the deformation it describes.
        self.energy = float(lam @ gram @ lam)
        #: Max |L_k(v) - c_k| over all rows, in each row's own units (strain rows: per sigma).
        self.max_violation = float(np.abs(gram @ lam - c).max())

    @property
    def n_rows(self) -> int:
        return len(self._lam_v) + len(self._lam_s)

    def expansion(self) -> KernelExpansion:
        """This field as a :class:`KernelExpansion` - one term per distinct constraint point."""
        points = np.concatenate([self._yv, self._ys])
        centers, row_center = np.unique(points, axis=0, return_inverse=True)
        row_center = row_center.reshape(-1)
        vectors = np.zeros((len(centers), 3))
        matrices = np.zeros((len(centers), 3, 3))
        n_value = len(self._yv)
        np.add.at(vectors, row_center[:n_value], self._lam_v[:, None] * self._dv)
        np.add.at(matrices, row_center[n_value:],
                  (self._lam_s / self.sigma ** 2)[:, None, None] * self._ss)
        return KernelExpansion(self.sigma, centers, vectors, matrices)

    def _gram(self) -> np.ndarray:
        s2 = self.sigma ** 2
        yv, dv, ys, ss = self._yv, self._dv, self._ys, self._ss

        r_vv = yv[None, :, :] - yv[:, None, :]
        g_vv = _gauss((r_vv ** 2).sum(-1), self.sigma) * (dv @ dv.T)

        r_vs = ys[None, :, :] - yv[:, None, :]                      # y_strain_j - y_value_i
        k_vs = _gauss((r_vs ** 2).sum(-1), self.sigma)
        g_vs = -k_vs / s2 * np.einsum("ia,jab,ijb->ij", dv, ss, r_vs)

        r_ss = ys[None, :, :] - ys[:, None, :]
        k_ss = _gauss((r_ss ** 2).sum(-1), self.sigma)
        frob = np.einsum("iab,jab->ij", ss, ss)
        s_i_r = np.einsum("iab,ijb->ija", ss, r_ss)                 # S_i r_ij
        s_j_r = np.einsum("jab,ijb->ija", ss, r_ss)                 # S_j r_ij
        g_ss = k_ss * (frob / s2 - (s_i_r * s_j_r).sum(-1) / s2 ** 2)

        return np.block([[g_vv, g_vs], [g_vs.T, g_ss]])

    def __call__(self, points: np.ndarray, chunk: int = 1024) -> np.ndarray:
        """(n, 3) field values at ``points`` (any points, not only mesh vertices)."""
        points = _as_points(points)
        out = np.zeros_like(points)
        s2 = self.sigma ** 2
        for start in range(0, len(points), chunk):
            x = points[start:start + chunk]
            if len(self._lam_v):
                r = x[:, None, :] - self._yv[None, :, :]
                k = _gauss((r ** 2).sum(-1), self.sigma)
                out[start:start + chunk] += (k * self._lam_v) @ self._dv
            if len(self._lam_s):
                r = x[:, None, :] - self._ys[None, :, :]
                k = _gauss((r ** 2).sum(-1), self.sigma) * (self._lam_s / s2)
                out[start:start + chunk] += np.einsum("nj,jab,njb->na", k, self._ss, r)
        return out
