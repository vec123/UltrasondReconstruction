"""Reader that turns a Deformetrica deterministic atlas into a linear SSM.

The BabyFM model arrives as a .mat that already carries a mean and a basis of
eigenfunctions. The fetus model does not: it is the raw output of a
Deformetrica ``DeterministicAtlas`` estimation, which stores a shape population
as one template mesh plus, per subject, an initial momentum attached to a
shared set of control points. The subject shapes are not in the files at all -
they are what the template *becomes* when it is carried along the LDDMM
geodesic those momenta generate.

So the conversion has two halves:

1. **Shooting.** For every subject, integrate the Hamiltonian system

   .. math::
      \\dot c_i = \\sum_j K(c_i, c_j)\\,\\alpha_j, \\qquad
      \\dot \\alpha_i = -\\sum_j (\\alpha_i \\cdot \\alpha_j)\\,\\nabla_{c_i} K(c_i, c_j)

   from :math:`t=0` to :math:`t=1`, dragging the template vertices along the
   induced velocity field :math:`\\dot x = \\sum_j K(x, c_j)\\,\\alpha_j`. The
   kernel is Deformetrica's Gaussian, :math:`K(x,y)=\\exp(-\\|x-y\\|^2/\\sigma^2)`,
   with :math:`\\sigma` the deformation kernel width (``kwd`` in params.json).
   Because every subject is reached by deforming the *same* template, the
   resulting meshes are vertex-wise corresponding by construction.

2. **PCA.** The shot population is then an ordinary (S, 3N) data matrix, and
   its principal components give exactly the mean / eigenfunctions /
   eigenvalues triple :class:`~src.model.model.StatisticalShapeModel` expects:
   orthonormal modes whose eigenvalues are variances.

This is a linearisation - shape space is a Riemannian manifold and the PCA is
done in ambient vertex coordinates - but it is the same linearisation the .mat
model already embodies (``refShape + meanDeformation`` plus a linear basis), so
the two models behave alike downstream. Shooting first, rather than running PCA
on the momenta directly, is what keeps it honest: the linear step then only has
to cover the spread of the population *around its mean*, not the whole
deformation from the template.

The atlas itself carries no landmark annotation. ``load_atlas_ssm``'s
``landmarks`` argument optionally points at the name -> vertex-index JSON
written by ``scripts/place_landmarks.py`` (see :mod:`src.model.landmarks`),
which fills the landmark dict the same way the .mat's fields do - so once a
landmark file exists, an atlas model can drive every landmark-dependent stage
just like the .mat one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Deformetrica's default 'number_of_time_points' for shooting.
DEFAULT_TIME_POINTS = 11


@dataclass
class DeterministicAtlas:
    """The estimated parameters of a Deformetrica deterministic atlas."""

    template: np.ndarray          # (N, 3) template vertices
    faces: Optional[np.ndarray]   # (F, 3) template triangles, 0-based
    control_points: np.ndarray    # (M, 3)
    momenta: np.ndarray           # (S, M, 3), one initial momentum per subject
    kernel_width: float           # sigma of the deformation kernel
    source: Path

    @property
    def n_subjects(self) -> int:
        return self.momenta.shape[0]

    @property
    def n_vertices(self) -> int:
        return self.template.shape[0]


# -- reading ---------------------------------------------------------------

def is_atlas(path) -> bool:
    """True when ``path`` looks like a Deformetrica atlas directory."""
    path = Path(path)
    return path.is_dir() and _find(path, "*Momenta*.txt") is not None


def _find(root: Path, pattern: str) -> Optional[Path]:
    """First match of ``pattern``, searched shallowest-first.

    The exported directories nest ('shape model/shape model/...') and the file
    names carry the model class as a prefix ('DeterministicAtlas__...'), so the
    files are located by pattern rather than by a fixed relative path.
    """
    matches = sorted(root.rglob(pattern), key=lambda p: (len(p.parts), p.name))
    return matches[0] if matches else None


def _require(root: Path, pattern: str) -> Path:
    match = _find(root, pattern)
    if match is None:
        raise FileNotFoundError(f"no file matching '{pattern}' under {root}")
    return match


def _read_params(root: Path) -> dict:
    """params.json, tolerating the trailing comma Deformetrica leaves in it."""
    path = _find(root, "params.json")
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", text))


def _read_control_points(path: Path) -> np.ndarray:
    points = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{path.name}: expected (M, 3) control points, got {points.shape}")
    return points


def _read_momenta(path: Path) -> np.ndarray:
    """(S, M, 3) momenta from the blank-line separated per-subject blocks.

    The header is ``<n_subjects> <n_control_points> <dimension>``; the rest of
    the file is one block of control-point rows per subject.
    """
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().split()
        values = np.fromstring(handle.read().replace("\n", " "), sep=" ")

    if len(header) != 3:
        raise ValueError(f"{path.name}: expected a '<S> <M> <dim>' header, got {header!r}")
    shape = tuple(int(v) for v in header)

    if values.size != int(np.prod(shape)):
        raise ValueError(
            f"{path.name}: header declares {shape} = {int(np.prod(shape))} values, "
            f"found {values.size}"
        )
    return values.reshape(shape)


def _read_template(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Vertices and triangles of the legacy-format template .vtk."""
    import vtk

    from src.geometry import get_vtp_faces, get_vtp_vertices

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = reader.GetOutput()

    if polydata.GetNumberOfPoints() == 0:
        raise ValueError(f"{path.name}: no points read - is it a legacy .vtk polydata file?")

    vertices = np.asarray(get_vtp_vertices(polydata), dtype=np.float64)
    return vertices, get_vtp_faces(polydata)


def load_atlas(path, kernel_width: Optional[float] = None) -> DeterministicAtlas:
    """Reads the estimated parameters of the atlas rooted at ``path``."""
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"a Deformetrica atlas is a directory, got {root}")

    control_points = _read_control_points(_require(root, "*ControlPoints*.txt"))
    momenta = _read_momenta(_require(root, "*Momenta*.txt"))
    template, faces = _read_template(_require(root, "*Template*.vtk"))

    if momenta.shape[1] != control_points.shape[0]:
        raise ValueError(
            f"momenta are attached to {momenta.shape[1]} control points but "
            f"{control_points.shape[0]} were read"
        )

    if kernel_width is None:
        kernel_width = _read_params(root).get("kwd")
        if kernel_width is None:
            raise ValueError(
                f"no deformation kernel width found: {root} has no params.json carrying "
                f"'kwd', so model.atlas.kernel_width has to be set explicitly"
            )
    kernel_width = float(kernel_width)
    if kernel_width <= 0:
        raise ValueError(f"kernel width must be positive, got {kernel_width}")

    return DeterministicAtlas(
        template=template,
        faces=faces,
        control_points=control_points,
        momenta=momenta,
        kernel_width=kernel_width,
        source=root,
    )


# -- geodesic shooting -----------------------------------------------------

def _gaussian(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """(len(x), len(y)) Gaussian kernel matrix, exp(-|x-y|^2 / sigma^2)."""
    sq = (
        np.einsum("ij,ij->i", x, x)[:, None]
        + np.einsum("ij,ij->i", y, y)[None, :]
        - 2.0 * (x @ y.T)
    )
    return np.exp(-np.maximum(sq, 0.0) / (sigma * sigma))


def _hamiltonian_derivative(c: np.ndarray, a: np.ndarray, sigma: float):
    """dc/dt and d(alpha)/dt of the control-point Hamiltonian system."""
    K = _gaussian(c, c, sigma)
    dc = K @ a

    # d(alpha)_i/dt = -sum_j (a_i . a_j) grad_{c_i} K_ij, and for the Gaussian
    # grad_{c_i} K_ij = -2/sigma^2 (c_i - c_j) K_ij, which leaves a sum that
    # factors into two matrix products.
    weights = K * (a @ a.T)
    da = (2.0 / (sigma * sigma)) * (weights.sum(axis=1)[:, None] * c - weights @ c)
    return dc, da


def shoot(
    control_points: np.ndarray,
    momenta: np.ndarray,
    points: np.ndarray,
    sigma: float,
    time_points: int = DEFAULT_TIME_POINTS,
) -> np.ndarray:
    """Flows ``points`` along the geodesic generated by a single momentum.

    Integrated with the midpoint rule (Deformetrica's 'RK2' scheme) over
    ``time_points - 1`` steps of the unit time interval.
    """
    steps = int(time_points) - 1
    if steps < 1:
        raise ValueError(f"time_points must be at least 2, got {time_points}")

    c = np.array(control_points, dtype=np.float64, copy=True)
    a = np.array(momenta, dtype=np.float64, copy=True)
    x = np.array(points, dtype=np.float64, copy=True)
    dt = 1.0 / steps

    for _ in range(steps):
        dc, da = _hamiltonian_derivative(c, a, sigma)
        dx = _gaussian(x, c, sigma) @ a

        # midpoint: re-evaluate the field half a step in, then take the full
        # step with that derivative
        c_mid, a_mid, x_mid = c + 0.5 * dt * dc, a + 0.5 * dt * da, x + 0.5 * dt * dx
        dc_mid, da_mid = _hamiltonian_derivative(c_mid, a_mid, sigma)
        dx_mid = _gaussian(x_mid, c_mid, sigma) @ a_mid

        c += dt * dc_mid
        a += dt * da_mid
        x += dt * dx_mid

    return x


def shoot_population(
    atlas: DeterministicAtlas,
    time_points: int = DEFAULT_TIME_POINTS,
    verbose: bool = False,
) -> np.ndarray:
    """(S, N, 3) subject meshes, one geodesic per subject from the template."""
    shapes = np.empty((atlas.n_subjects, atlas.n_vertices, 3), dtype=np.float64)
    for s in range(atlas.n_subjects):
        shapes[s] = shoot(
            atlas.control_points, atlas.momenta[s], atlas.template,
            atlas.kernel_width, time_points,
        )
        if verbose and (s + 1) % 50 == 0:
            print(f"  shot {s + 1}/{atlas.n_subjects} subjects")
    return shapes


# -- PCA -------------------------------------------------------------------

def population_pca(shapes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, orthonormal modes and their variances for an (S, N, 3) population.

    Centring bounds the rank at ``S - 1``, so numerically empty modes are
    dropped here; any deliberate truncation is :func:`truncate`'s job.
    """
    data = np.asarray(shapes, dtype=np.float64).reshape(len(shapes), -1)
    n_subjects = len(data)
    if n_subjects < 2:
        raise ValueError(f"PCA needs at least 2 subjects, got {n_subjects}")

    mean = data.mean(axis=0)
    _, singular, right = np.linalg.svd(data - mean, full_matrices=False)

    eigenvalues = singular ** 2 / (n_subjects - 1)
    keep = eigenvalues > eigenvalues.max() * 1e-12

    return mean, np.ascontiguousarray(right[keep].T), eigenvalues[keep]


def truncate(
    eigenfunctions: np.ndarray,
    eigenvalues: np.ndarray,
    n_components: Optional[int] = None,
    variance_threshold: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keeps the leading modes, by explained variance and/or by count.

    Both limits apply when both are given, so the stricter one wins. Modes come
    out in descending eigenvalue order, as PCA produced them.
    """
    keep = len(eigenvalues)

    if 0.0 < variance_threshold < 1.0:
        fraction = np.cumsum(eigenvalues) / eigenvalues.sum()
        keep = min(keep, int(np.searchsorted(fraction, variance_threshold) + 1))

    if n_components is not None:
        if n_components < 1:
            raise ValueError(f"n_components must be positive, got {n_components}")
        keep = min(keep, n_components)

    return eigenfunctions[:, :keep], eigenvalues[:keep]


# -- caching ---------------------------------------------------------------

def _cache_path(atlas: DeterministicAtlas, time_points: int) -> Path:
    """Where the shot-and-decomposed model is kept.

    Only the shooting parameters go in the name. Truncation and normalisation
    are applied *after* the cache is read - they are cheap and cost nothing to
    redo - so changing them does not invalidate the expensive part.
    """
    return atlas.source / f"ssm_shot_sigma{atlas.kernel_width:g}_tp{int(time_points)}.npz"


def _read_cache(path: Path, atlas: DeterministicAtlas):
    """The cached full-rank PCA, or None when it is absent or stale."""
    if not path.is_file():
        return None
    try:
        with np.load(path) as cached:
            if (int(cached["n_subjects"]) != atlas.n_subjects
                    or int(cached["n_vertices"]) != atlas.n_vertices):
                return None
            return cached["mean"], cached["eigenfunctions"], cached["eigenvalues"]
    except Exception as exc:                     # noqa: BLE001 - a bad cache is not fatal
        print(f"  ignoring unreadable cache {path.name}: {type(exc).__name__}: {exc}")
        return None


def _write_cache(path: Path, atlas: DeterministicAtlas, mean, eigenfunctions, eigenvalues):
    try:
        np.savez_compressed(
            path, mean=mean, eigenfunctions=eigenfunctions, eigenvalues=eigenvalues,
            n_subjects=atlas.n_subjects, n_vertices=atlas.n_vertices,
            kernel_width=atlas.kernel_width,
        )
    except OSError as exc:
        print(f"  could not write cache {path}: {exc}")


# -- the loader the runners see --------------------------------------------

def load_atlas_ssm(
    path,
    normalize: bool = True,
    kernel_width: Optional[float] = None,
    time_points: int = DEFAULT_TIME_POINTS,
    n_components: Optional[int] = None,
    variance_threshold: float = 1.0,
    cache: bool = True,
    landmarks: Optional[Path] = None,
    verbose: bool = False,
):
    """Same 4-tuple as :func:`src.model.load.load_mat_ssm`, from an atlas.

    ``landmarks`` optionally names the JSON written by
    ``scripts/place_landmarks.py`` (name -> vertex index into the mean shape).
    Left unset - or pointing at a file that does not exist yet - the landmark
    dictionary comes back empty, carrying the keys the .mat loader uses so
    callers can ask for landmarks uniformly and simply find none.
    """
    from src.model.load import normalize_ssm
    from src.model.landmarks import read_landmark_map

    atlas = load_atlas(path, kernel_width=kernel_width)
    if verbose:
        print(f"atlas: {atlas.n_subjects} subjects, {atlas.n_vertices} vertices, "
              f"{len(atlas.control_points)} control points, sigma {atlas.kernel_width:g}")

    cache_path = _cache_path(atlas, time_points)
    decomposition = _read_cache(cache_path, atlas) if cache else None

    if decomposition is None:
        if verbose:
            print(f"  shooting {atlas.n_subjects} geodesics ({time_points} time points)")
        shapes = shoot_population(atlas, time_points=time_points, verbose=verbose)
        decomposition = population_pca(shapes)
        if cache:
            _write_cache(cache_path, atlas, *decomposition)
    elif verbose:
        print(f"  decomposition read from {cache_path.name}")

    mean, eigenfunctions, eigenvalues = decomposition
    eigenfunctions, eigenvalues = truncate(
        eigenfunctions, eigenvalues,
        n_components=n_components, variance_threshold=variance_threshold,
    )

    if normalize:
        mean, eigenfunctions, eigenvalues = normalize_ssm(mean, eigenfunctions, eigenvalues)

    n_vertices = len(mean) // 3
    names, indices = [], []
    for name, index in (read_landmark_map(landmarks) if landmarks else {}).items():
        if 0 <= index < n_vertices:
            names.append(name)
            indices.append(index)
        elif verbose:
            print(f"  landmark '{name}': vertex {index} is out of range for "
                  f"{n_vertices} vertices, skipped")
    if verbose and names:
        print(f"  landmarks: {len(names)} read from {Path(landmarks).name}")

    landmark_data = {"verts ": np.asarray(indices, dtype=np.int64), "names ": names}
    return mean, eigenfunctions, eigenvalues, landmark_data


# -- the model kept on the manifold ----------------------------------------

class MomentumShapeModel:
    """The atlas as an LDDMM shape model: PCA in the tangent space, then shoot.

    :func:`load_atlas_ssm` linearises twice over - it shoots the population and
    then does PCA on the resulting vertex coordinates, so a sample is a sum of
    ambient displacements. This class moves the statistics one level down, onto
    the initial momenta, and keeps the deformation itself nonlinear:

    .. math:: p(\\alpha) = \\bar{p} + \\sum_k \\alpha_k \\psi_k, \\qquad
              x(\\alpha) = \\mathrm{shoot}(\\bar{x}_{\\text{template}}, p(\\alpha))

    That is the standard LDDMM statistical model (tangent PCA). PCA still
    appears - it has to, the momenta are what varies between subjects - but it
    is done where the space really is linear, and every sample is reached by a
    geodesic, so it is a diffeomorphism of the template: it cannot fold the
    surface through itself the way an ambient linear combination can.

    The cost is that a sample is no longer a matrix product: each one needs an
    integration, ~0.3 s at 6 time points against microseconds for the linear
    model, which is why the viewer keeps both.
    """

    def __init__(self, atlas: DeterministicAtlas, time_points: int = DEFAULT_TIME_POINTS):
        self.atlas = atlas
        self.time_points = int(time_points)

        momenta = atlas.momenta.reshape(atlas.n_subjects, -1)
        self.mean_momentum = momenta.mean(axis=0)
        _, singular, right = np.linalg.svd(momenta - self.mean_momentum, full_matrices=False)

        eigenvalues = singular ** 2 / (atlas.n_subjects - 1)
        keep = eigenvalues > eigenvalues.max() * 1e-12
        self.eigenvalues = eigenvalues[keep]
        self.modes = np.ascontiguousarray(right[keep].T)      # (3M, K)

    @property
    def n_modes(self) -> int:
        return len(self.eigenvalues)

    @property
    def n_vertices(self) -> int:
        return self.atlas.n_vertices

    @property
    def variance_share(self) -> np.ndarray:
        return self.eigenvalues / self.eigenvalues.sum()

    def momentum(self, alpha: np.ndarray) -> np.ndarray:
        """(M, 3) initial momentum for the coefficients ``alpha``."""
        return (self.mean_momentum + self.modes @ alpha).reshape(-1, 3)

    def reconstruct(self, alpha: np.ndarray, time_points: Optional[int] = None) -> np.ndarray:
        """(N, 3) vertices reached by shooting the template along ``alpha``."""
        return shoot(
            self.atlas.control_points, self.momentum(alpha), self.atlas.template,
            self.atlas.kernel_width, time_points or self.time_points,
        )

    def mean_shape(self, time_points: Optional[int] = None) -> np.ndarray:
        """The shape at alpha = 0 - the mean *momentum* shot, not the template."""
        return self.reconstruct(np.zeros(self.n_modes), time_points)


def load_momentum_model(
    path,
    kernel_width: Optional[float] = None,
    time_points: int = DEFAULT_TIME_POINTS,
    verbose: bool = False,
) -> MomentumShapeModel:
    """Reads an atlas and decomposes its momenta in the tangent space."""
    atlas = load_atlas(path, kernel_width=kernel_width)
    model = MomentumShapeModel(atlas, time_points=time_points)
    if verbose:
        print(f"atlas: {atlas.n_subjects} subjects, {atlas.n_vertices} vertices, "
              f"{len(atlas.control_points)} control points, sigma {atlas.kernel_width:g}")
        print(f"  momentum PCA: {model.n_modes} modes, "
              f"mode 0 carries {model.variance_share[0]:.1%}")
    return model


def load_atlas_triangulation(path):
    """(F, 3) template triangles, or None when the template carries no polys.

    The kernel width is irrelevant to the topology, so it is not required here.
    """
    return load_atlas(path, kernel_width=1.0).faces
