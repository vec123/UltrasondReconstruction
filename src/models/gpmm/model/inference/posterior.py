"""Gaussian-process posterior of a low-rank shape model, given partial observations.

The model is ``x = mu + Phi alpha`` with ``alpha ~ N(0, Lambda)``, plus an
isotropic residual of variance ``noise`` that the truncation left out.
Observing R vertices ``I`` as ``y = x_I + eps``, ``eps ~ N(0, Sigma)`` with a
diagonal ``Sigma`` (see :class:`~src.models.gpmm.model.inference.observations.Observations`),
gives a Gaussian posterior over ``alpha`` in closed form. It is written in the
whitened coefficients ``beta = Lambda^{-1/2} alpha``, so a zero eigenvalue
costs nothing and the only system solved is K x K:

    B      = Phi_I Lambda^{1/2}                      (3R, K)
    W      = (Sigma + noise I)^{-1}                  (3R,) diagonal
    A      = I + B^T W B                             (K, K), SPD
    beta   = A^{-1} B^T W (y - mu_I)
    S      = Lambda^{1/2} A^{-1} Lambda^{1/2}        posterior Cov(alpha)

That is the low-rank form of the GPMM posterior (Luethi et al. 2018, Eq.
``posterior_inference_kernel`` in ``theory/shape_deformation_models.tex``).
No row of ``Phi`` is assumed orthonormal, which matters: the GPMM archives
are orthonormal in the *mass* inner product, where ``Phi_I^T Phi_I`` is
nothing in particular.

**The posterior is a model of the same kind** - mean ``mu + Phi alpha_mean``,
covariance ``Phi S Phi^T`` - so it is returned as one. With ``S = U D U^T``,
``Phi' = Phi U`` and ``Lambda' = D``; ``U`` is orthogonal, so ``Phi'`` is
orthonormal in whichever inner product ``Phi`` was. Everything that takes a
:class:`~src.models.gpmm.model.model.StatisticalShapeModel` (``ssm_fit``, sampling, the
std fields) therefore takes the posterior as it is, and :func:`save_posterior`
writes it in the same ``.npz`` layout as any built model.

Two consequences worth knowing:

* The posterior stays in the prior's span. A pure PCA model cannot follow an
  observation outside its modes; that is the reason to condition a model with
  a parameterised term (``baby_pca_sum_matern32``) rather than ``baby_pca``.
* ``noise`` is carried over unchanged. Observing a vertex would also shrink the
  residual near it; keeping the prior value is the conservative choice.

**Pose.** The model has no pose freedom: a model built from aligned shapes
explains a rotated or rescaled observation with its shape modes, which fits the
observed part at the price of extreme coefficients and an implausible shape
everywhere else. :func:`posterior_with_pose` removes the pose first - a
weighted similarity of the model shape onto the observations - conditions on
the pose-free observations, and alternates the two. The result keeps that
``pose``, so :meth:`PosteriorResult.posed` puts any shape of the posterior back
where the observations are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import scipy.linalg as sla

from src.models.gpmm.kernels.data import Pca
from src.models.gpmm.model.inference.continuous_inference import build_inference
from src.models.gpmm.model.model import StatisticalShapeModel
from src.models.gpmm.model.model_memory_io import ModelCacheSpec, save_model
from src.models.gpmm.model.inference.observations import Observations
from src.shared.transforms.similarity import SimilarityTransform


@dataclass
class PosteriorResult:
    """A conditioned model, plus what produced it and how well it explains it."""

    prior: StatisticalShapeModel
    observations: Observations
    model: StatisticalShapeModel      # the posterior, as a model in its own right
    alpha_mean: np.ndarray            # (K,) posterior mean, in the PRIOR's coefficients
    coef_cov: np.ndarray              # (K, K) posterior Cov(alpha), same basis
    log_marginal_likelihood: float    # log p(y) under the prior - compare noise settings with it
    #: SimilarityTransform from the model frame to the frame the observations
    #: were given in, or None for the identity. ``observations`` are the
    #: pose-free ones, in the model frame, like every shape of ``model``.
    pose: Optional[SimilarityTransform] = None

    @property
    def mean(self) -> np.ndarray:
        """(3N,) posterior mean shape."""
        return self.model.mean

    def posed(self, shape) -> np.ndarray:
        """(3N,) ``shape`` (model frame) moved by ``pose`` to the observations' frame."""
        shape = np.asarray(shape, dtype=np.float64)
        if self.pose is None:
            return shape.reshape(-1)
        return self.pose.apply_points(shape.reshape(-1, 3)).reshape(-1)

    def mahalanobis_sq(self) -> float:
        """``sum alpha_k^2 / lambda_k`` of the posterior mean under the prior.

        About ``n_components`` for a typical shape of the prior; far above it,
        the observations push the model outside its plausible range.
        """
        lam = np.asarray(self.prior.eigenvalues, dtype=np.float64)
        keep = lam > 0
        return float(np.sum(self.alpha_mean[keep] ** 2 / lam[keep]))

    def vertex_std(self, include_noise: bool = False) -> np.ndarray:
        """(N,) posterior std per vertex, as ``sqrt(trace)`` of its 3x3 covariance."""
        return self.model.vertex_std(include_noise=include_noise)

    def prior_std(self, include_noise: bool = False) -> np.ndarray:
        """(N,) the same quantity before conditioning."""
        return self.prior.vertex_std(include_noise=include_noise)

    def informative_mask(self, ratio: float = 0.5) -> np.ndarray:
        """(N,) True where an observation's weight ``1 / sigma^2`` is at least
        ``ratio`` times the largest one.

        The split that means the same under every noise model: under ``cutoff``
        it is the observed set, under ``logistic`` - which observes everything -
        the vertices below about its threshold, where the weight has not yet
        halved. Relative to the best observation rather than to the prior's std:
        a few thousand observations each looser than the prior still constrain
        the fit together.
        """
        mask = np.zeros(self.model.n_vertices, dtype=bool)
        if self.observations.n:
            weight = 1.0 / self.observations.variances
            mask[self.observations.indices] = weight >= ratio * weight.max()
        return mask

    def residuals(self) -> np.ndarray:
        """(R,) distance from the posterior mean to each observed target."""
        mean = self.mean.reshape(-1, 3)[self.observations.indices]
        return np.linalg.norm(mean - self.observations.targets, axis=1)

    def sample(self, n: int = 1, rng=None, include_noise: bool = False) -> np.ndarray:
        """(n, 3N) shapes drawn from the posterior."""
        return self.model.sample(n, rng=rng, include_noise=include_noise)

    def summary(self, length_scale: float = 1.0) -> Dict[str, Any]:
        """Plain-float digest for a JSON report.

        Lengths are in the model frame's units times ``length_scale`` - pass
        ``1 / transform.scale`` to report them in the frame the observations
        came from.
        """
        informative = self.informative_mask()
        std = self.vertex_std()
        prior_std = self.prior_std()
        residuals = self.residuals()

        def stats(values):
            if not values.size:
                return {}
            values = values * float(length_scale)
            return {"mean": float(values.mean()), "median": float(np.median(values)),
                    "max": float(values.max())}

        pose = {}
        if self.pose is not None:
            cos = np.clip((np.trace(self.pose.rotation) - 1.0) / 2.0, -1.0, 1.0)
            pose = {"rotation_deg": float(np.degrees(np.arccos(cos))), "scale": float(self.pose.scale)}

        return {
            "n_vertices": int(self.model.n_vertices),
            "n_observed": int(self.observations.n),
            "n_informative": int(informative.sum()),
            "n_components": int(self.model.n_components),
            "mahalanobis_sq": self.mahalanobis_sq(),
            "pose": pose,
            "log_marginal_likelihood": float(self.log_marginal_likelihood),
            "observation_std": stats(self.observations.stds),
            "residual_informative": stats(residuals[informative[self.observations.indices]]),
            "posterior_std_informative": stats(std[informative]),
            "posterior_std_uninformative": stats(std[~informative]),
            "prior_std_informative": stats(prior_std[informative]),
            "prior_std_uninformative": stats(prior_std[~informative]),
        }


def condition_whitened(B, w, r):
    """``(beta, A^{-1}, log p(r))`` for ``r = B beta + eps``, ``beta ~ N(0, I)``,
    ``eps ~ N(0, diag(1 / w))`` - the closed form of :func:`posterior`.

    Any Gaussian prior ``N(m, L L^T)`` on coefficients comes down to this with
    ``B = Phi_I L`` and ``r`` the residual from ``mu_I + Phi_I m``; it needs no
    model built for the prior, which is what an EM over many cases calls.
    """
    BtW = B.T * w[None, :]
    A = np.eye(B.shape[1]) + BtW @ B
    factor = sla.cho_factor(A, lower=True, check_finite=False)
    BtWr = BtW @ r
    beta = sla.cho_solve(factor, BtWr, check_finite=False)
    A_inv = sla.cho_solve(factor, np.eye(B.shape[1]), check_finite=False)

    # log N(r; 0, B B^T + W^{-1}) via Woodbury and the determinant lemma.
    quad = float(r @ (w * r) - BtWr @ beta)
    logdet = 2.0 * float(np.log(np.diag(factor[0])).sum()) - float(np.log(w).sum())
    lml = -0.5 * (quad + logdet + r.size * np.log(2.0 * np.pi))
    return beta, A_inv, lml


def posterior(model: StatisticalShapeModel, observations: Observations,
              include_model_noise: bool = True) -> PosteriorResult:
    """Conditions ``model`` on ``observations``; both in the model's own frame.

    ``include_model_noise`` adds the model's unrepresented ``noise`` to every
    observation's variance, which is what keeps a vertex the truncated basis
    cannot reach from pulling the whole fit towards it.
    """
    if observations.indices.size and observations.indices.max() >= model.n_vertices:
        raise ValueError(
            f"observation names vertex {observations.indices.max()}, but the model has "
            f"only {model.n_vertices} vertices"
        )

    lam = np.clip(np.asarray(model.eigenvalues, dtype=np.float64), 0.0, None)
    sqrt_lam = np.sqrt(lam)

    B = model.phi_rows(observations.indices) * sqrt_lam[None, :]
    w = observations.precision(model.noise if include_model_noise else 0.0)
    r = observations.targets.reshape(-1) - model.x_rows(model.mean, observations.indices)
    beta, A_inv, lml = condition_whitened(B, w, r)

    alpha_mean = sqrt_lam * beta
    coef_cov = sqrt_lam[:, None] * A_inv * sqrt_lam[None, :]
    coef_cov = 0.5 * (coef_cov + coef_cov.T)

    return PosteriorResult(
        prior=model,
        observations=observations,
        model=_posterior_model(model, alpha_mean, coef_cov),
        alpha_mean=alpha_mean,
        coef_cov=coef_cov,
        log_marginal_likelihood=lml,
    )


def fit_pose(shape, observations: Observations, with_scale: bool = True) -> SimilarityTransform:
    """Weighted Procrustes of ``shape``'s observed vertices onto the observations.

    Weighted by precision, so a vertex the noise model barely trusts barely
    steers the pose either.
    """
    source = np.asarray(shape, dtype=np.float64).reshape(-1, 3)[observations.indices]
    return SimilarityTransform.from_correspondences(
        source, observations.targets, with_scale=with_scale,
        weights=1.0 / observations.variances, stage="posterior_pose")


def posterior_with_pose(model: StatisticalShapeModel, observations: Observations,
                        with_scale: bool = True, iterations: int = 3,
                        include_model_noise: bool = True) -> PosteriorResult:
    """Pose and shape together: alternate a pose fit and the posterior.

    The first pose aligns the model mean, i.e. plain pre-alignment; each further
    iteration refits the pose to the latest posterior mean and conditions the
    prior again on the observations without it. ``observations`` are in the
    model frame up to that pose; the result's ``pose`` maps back to them.
    """
    shape = model.mean
    for _ in range(max(1, int(iterations))):
        pose = fit_pose(shape, observations, with_scale=with_scale)
        result = posterior(model, observations.transformed(pose.inverse()),
                           include_model_noise=include_model_noise)
        shape = result.mean
    result.pose = pose
    return result


def _posterior_model(prior: StatisticalShapeModel, alpha_mean, coef_cov) -> StatisticalShapeModel:
    """The posterior as a model: ``mu + Phi alpha_mean``, ``Phi U``, ``D``.

    It keeps the prior's domain on purpose: the process is still indexed by
    the prior's reference points, and the mass inner product ``Phi'`` is
    orthonormal in is that domain's.
    """
    values, vectors = np.linalg.eigh(coef_cov)
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 0.0, None)
    eigenfunctions = prior.eigenfunctions @ vectors[:, order]

    kernel = Pca.from_eigendecomposition(eigenfunctions, values)
    inference = None
    if prior.domain is not None:
        inference = build_inference("none").prepare(prior.domain, kernel, eigenfunctions)

    return StatisticalShapeModel(
        mean=prior.reconstruct(alpha_mean),
        eigenfunctions=eigenfunctions,
        eigenvalues=values,
        landmarks=prior.landmarks,
        faces=prior.faces,
        domain=prior.domain,
        kernel=kernel,
        inference=inference,
        noise=prior.noise,
        inner_product=prior.inner_product,
    )


def save_posterior(path, result: PosteriorResult,
                   provenance: Optional[Dict[str, Any]] = None) -> None:
    """Writes the posterior model as a regular GPMM ``.npz``.

    Its kernel is the exact ``pca`` of its own decomposition, so
    :func:`~src.models.gpmm.model.model_memory_io.load_model_file` reads it back like
    any built model and ``ssm_fit`` can take it as ``shape_model_mat``.
    ``provenance`` (JSON-able: prior path, noise model, ...) is stored as the
    archive's ``approximation`` entry and so also enters its cache key.
    """
    model = result.model
    spec = ModelCacheSpec(
        kernel_spec={"name": "pca", "kwargs": {"basis": model.kernel.basis}},
        approximation={"name": "gp_posterior", "kwargs": dict(provenance or {})},
        inference={"name": "none", "kwargs": {}},
    )
    save_model(path, model, spec)
