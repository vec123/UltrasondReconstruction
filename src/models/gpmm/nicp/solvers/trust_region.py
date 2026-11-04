"""Constrained solver: minimises the data term inside a mahalanobis radius."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from src.models.gpmm.nicp.problem import FitProblem
from src.shared.core.registry import RegistryError

from .base import SolveResult, Solver, _should_refresh, register_solver
from .gradient import GradientSolver


@register_solver("trust_region")
@dataclass(frozen=True)
class TrustRegionSolver(Solver):
    """Minimise the data term *subject to* ``sum(alpha_i^2 / lambda_i) <= radius``.

    The same ellipsoid ``mahalanobis_clamp`` enforces, but reached through the
    constraint's Lagrange multiplier instead of by projection. Writing the
    whitened coefficients ``z_i = alpha_i / sqrt(lambda_i)``, the constrained
    problem

        min_z  f(z)   s.t.  ||z||^2 <= Delta^2

    has KKT conditions ``grad f(z) + sigma z = 0``, ``sigma >= 0`` and
    ``sigma (||z||^2 - Delta^2) = 0``. So sigma *is* a mahalanobis prior weight -
    precisely the one whose unconstrained minimiser lands on the boundary - and
    re-solving it every iteration is what a fixed configured weight cannot do
    once ICP re-matching has moved the design matrix underneath it.

    Two regimes, chosen by what the objective admits. The multiplier is found the
    same way in both, so a fit means the same thing either way:

    exact (``matched_l2``)
        The classic trust-region subproblem (More-Sorensen). With
        ``A = s * Phi~^T W Phi~`` and ``b = s * Phi~^T W u`` in whitened
        coordinates, ``z(sigma) = (A + sigma I)^-1 b``; one eigendecomposition of
        A (K x K, K at most a few hundred, so free) turns the secular equation
        into ``sum c_i^2 / (d_i + sigma)^2 = Delta^2``, strictly decreasing in
        sigma and so uniquely solvable by bisection. Each iteration solves the
        whole linearised problem at the current correspondences rather than
        taking a step towards it.

    general (anything else, e.g. ``p2plane_chamfer``)
        No closed form, so the multiplier is found around an inner optimiser: for
        a trial sigma, descend ``f(alpha) + sigma * sum(alpha_i^2/lambda_i)`` from
        a warm start and read off the radius it settles at. The same bisection
        applies - at the cost of one inner solve per bracket step, which is why
        ``inner_iterations`` is small and the bisection is capped.

    **How much of this survives a non-quadratic objective.** Three claims, which
    fail at different points, and it is worth being precise about which:

    1. *A solution exists.* For any continuous f, since the feasible set is a
       closed ball. Never at issue.
    2. *The radius decreases as the multiplier grows,* so bisection is meaningful.
       This needs no convexity and no quadratic model - only that each
       ``z(sigma)`` is a GLOBAL minimiser of the penalised objective, whereupon
       adding the two optimality inequalities for sigma_1 < sigma_2 gives
       ``(sigma_2 - sigma_1)(||z_1||^2 - ||z_2||^2) >= 0`` directly. The exact
       regime returns global minimisers and so inherits this outright. The
       general regime returns whatever Adam reaches from a warm start, and it is
       *there* - in the local solve, not in the shape of f - that monotonicity is
       forfeited.
    3. *The multiplier certifies the constrained optimum.* Quadratic only. The
       trust-region subproblem is the textbook non-convex problem with no duality
       gap: KKT plus ``A + sigma I >= 0`` is necessary AND sufficient for the
       global solution. For a general f, KKT is merely necessary at a local
       minimum, so what comes back is a stationary point that respects the bound,
       not a certified best fit within it.

    The objectives here are quadratic more often than the regime split suggests.
    ``matched_l2`` at fixed correspondences is exactly quadratic in alpha. The
    Chamfer terms are *piecewise* quadratic: the nearest-neighbour assignment is
    piecewise constant in alpha, and within one assignment cell both the
    point-to-point and the point-to-plane residual are quadratic. So the general
    regime is not fighting curvature, it is fighting the cell boundaries - which
    is also why ``_curvature`` can take a genuine Gauss-Newton Hessian from
    autograd there rather than an approximation to one. A robust kernel or p=1
    leaves that quadratic structure for good.

    The fit's own ``prior`` must be ``none``: this solver supplies the prior, and
    a second one on top would move the boundary it is solving for.
    """

    #: chi-square level defining the radius, as in mahalanobis_clamp.
    p: float = 0.99
    #: Which magnitude the radius constrains - the whole posterior, or its mean.
    #:
    #: "expected"  the EXPECTED radius under the posterior,
    #:             E[||z||^2 | u] = ||z_hat||^2 + tr(Sigma_z) <= Delta^2, the
    #:             quantity derived in theory/shape_deformation_models.tex. This
    #:             is the honest reading of "the fit is inside the 99% level":
    #:             a draw from the posterior is inside it, not merely the single
    #:             shape at its centre. The point estimate consequently lands
    #:             strictly INSIDE the ellipsoid, by exactly tr(Sigma_z), because
    #:             the posterior spread is charged against the same budget.
    #: "mean"      the posterior mean alone, ||z_hat||^2 <= Delta^2. The
    #:             constraint binds, so the fit sits exactly ON the boundary:
    #:             the most data-fitting shape whose CENTRE is still admissible,
    #:             with half its posterior mass outside the level.
    #:
    #: Matching the objective s||u - Phi~ z||_W^2 + sigma||z||^2 against that
    #: section's S(z) gives sigma_n^2 = sigma/s and Sigma_z = sigma (A + sigma I)^-1,
    #: so tr(Sigma_z) = sigma * sum 1/(d_i + sigma), where A is *half* the
    #: Hessian of the data term in whitened coordinates. The exact regime has A
    #: in closed form; the general regime takes it from autograd (see
    #: ``_curvature``), so both settings are available under either regime.
    constrain: str = "expected"
    #: explicit Delta^2, overriding ``p`` when set.
    radius_sq: Optional[float] = None
    iterations: int = 10
    #: bisection steps on the multiplier (exact regime).
    sigma_iterations: int = 60
    # -- general regime only ------------------------------------------------
    optimizer: str = "adam"
    lr: float = 1.0e-1
    inner_iterations: int = 25
    #: bisection steps in the general regime - each one costs an inner solve.
    outer_iterations: int = 12
    #: general regime, constrain="expected": recompute the curvature at every
    #: trial multiplier instead of once per ICP iteration at the warm start.
    #: Exact, and costs one Hessian per bracket step rather than one per round.
    hessian_refresh: bool = False
    #: basis directions per batched double-backward when building that Hessian.
    #: Lower it if the vmapped graph does not fit in memory.
    hessian_chunk: int = 32
    tolerance: float = 0.0
    patience: int = 3
    log_every: int = 1

    def __post_init__(self):
        if not 0.0 < self.p < 1.0:
            raise ValueError(f"p must be in (0, 1), got {self.p}")
        if self.radius_sq is not None and self.radius_sq <= 0:
            raise ValueError(f"radius_sq must be positive, got {self.radius_sq}")
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if self.optimizer not in GradientSolver.OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {list(GradientSolver.OPTIMIZERS)}, "
                f"got {self.optimizer!r}"
            )

    def validate(self, problem: FitProblem) -> None:
        if problem.prior.name != "none":
            raise RegistryError(
                f"solver 'trust_region' solves for the prior weight itself - it is the "
                f"constraint's multiplier - so the fit's own prior must be 'none', got "
                f"'{problem.prior.name}'. Drop the prior, or use a different solver."
            )
        # E[||z||^2|u] = ||z_hat||^2 + tr(Sigma_z) has a FLOOR of K, approached as
        # sigma -> infinity: with the data switched off the posterior is the prior,
        # whose expected squared radius is K. So this constraint is feasible only
        # above K - unlike the point-estimate one, which can always be met by
        # shrinking towards the mean. chi2_crit(p, K) > K holds for every p above
        # roughly a half, so a sane level is never the problem; an explicit
        # radius_sq can be, and silently bisecting to sigma = 1e300 would hide it.
        if self.constrain == "expected":
            target, k = self.radius(problem.space), problem.space.n_components
            if target <= k:
                raise RegistryError(
                    f"solver 'trust_region' cannot constrain the expected radius to "
                    f"{target:.6g} for a model with K={k} modes: E[||z||^2|u] >= tr(Sigma_z) "
                    f"-> K as the prior takes over, so nothing below {k} is reachable. "
                    f"Raise 'p' (chi2_99 for K={k} is comfortably above it), raise "
                    f"'radius_sq', or switch to constrain: mean."
                )

    def radius(self, space) -> float:
        if self.radius_sq is not None:
            return float(self.radius_sq)
        from scipy.stats import chi2
        return float(chi2.ppf(self.p, space.n_components))

    @staticmethod
    def _trace(d: torch.Tensor, sigma: float) -> float:
        """``tr(Sigma_z) = sum_i sigma / (d_i + sigma)``, valid at sigma = 0.

        Written with sigma in the numerator rather than as ``sigma * sum 1/(d+sigma)``
        because A is routinely singular - fewer matched coordinates than modes,
        or simply directions the data cannot see - and the two forms disagree
        there. A direction with d_i = 0 is one the likelihood says nothing about,
        so its posterior variance is the prior's, 1, which is what the limit
        sigma/(0 + sigma) gives; the factored form gives 0 * inf = nan and would
        silently poison the comparison that drives the bisection.
        """
        if sigma <= 0.0:
            return float((d <= 0).sum())       # informed directions contribute 0
        return float((sigma / (d + sigma)).sum())

    @staticmethod
    def _bisect(radius_of_sigma, target, steps):
        """Smallest sigma >= 0 whose solution satisfies the radius.

        The map is decreasing, so an unconstrained solution already inside the
        ball returns sigma = 0 and the constraint is simply inactive - which is
        the KKT complementarity condition, not a special case.
        """
        if radius_of_sigma(0.0) <= target:
            return 0.0
        lo, hi = 0.0, 1.0
        while radius_of_sigma(hi) > target:
            hi *= 10.0
            if hi > 1.0e300:
                raise RegistryError(
                    f"solver 'trust_region' found no multiplier meeting radius^2={target:.6g}: "
                    f"the radius was still above it at sigma=1e300, where the prior wholly "
                    f"dominates the data. The bound is unreachable, not merely tight - check "
                    f"'radius_sq'/'p' against the model's mode count."
                )
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            if radius_of_sigma(mid) > target:
                lo = mid
            else:
                hi = mid
        return hi

    # -- exact regime -------------------------------------------------------

    def _solve_matched(self, problem, ctx, target_sq):
        space = problem.space
        corr = ctx.require_correspondences(problem.loss.name)
        source = torch.as_tensor(corr.source_idx, dtype=torch.long, device=space.device)
        target = torch.as_tensor(corr.target_idx, dtype=torch.long, device=space.device)

        root_lam = space.eigenvalues.sqrt()
        Phi_I = space.phi_rows(corr.source_idx)                      # (3P, K)
        mean_I = space.mean.view(-1, 3)[source].reshape(-1)          # (3P,)
        u = problem.target_points[target].reshape(-1) - mean_I       # deformation from the mean

        weight = torch.as_tensor(corr.weight, dtype=space.dtype, device=space.device)
        scale = 1.0 / weight.sum() if problem.loss.reduction == "mean" else 1.0
        w_coord = weight.repeat_interleave(3)

        Phi_t = Phi_I * root_lam.unsqueeze(0)                        # Phi~ = Phi_I Lam^1/2
        A = (scale * (Phi_t.T @ (w_coord.unsqueeze(1) * Phi_t))).double()
        b = (scale * (Phi_t.T @ (w_coord * u))).double()

        d, Q = torch.linalg.eigh(A)
        d = d.clamp_min(0.0)
        c = Q.T @ b

        def radius_of(sigma):
            # nan only where a null direction meets a zero coefficient, i.e. an
            # unidentified mode: infinitely far outside, so bracket upwards.
            z_sq = float(torch.nan_to_num(((c / (d + sigma)) ** 2), nan=float("inf")).sum())
            if self.constrain == "mean":
                return z_sq
            return z_sq + self._trace(d, sigma)

        sigma = self._bisect(radius_of, target_sq, self.sigma_iterations)
        z = Q @ (c / (d + sigma))
        z = torch.nan_to_num(z, nan=0.0)
        return z.to(space.dtype) * root_lam, sigma, self._trace(d, sigma)

    # -- general regime -----------------------------------------------------

    def _descend(self, problem, ctx, alpha0, sigma):
        space = problem.space
        alpha = alpha0.clone().requires_grad_(True)
        if self.optimizer == "adam":
            opt = torch.optim.Adam([alpha], lr=self.lr)
        elif self.optimizer == "sgd":
            opt = torch.optim.SGD([alpha], lr=self.lr)
        else:
            opt = torch.optim.LBFGS([alpha], lr=self.lr, max_iter=1, history_size=20)

        def closure():
            opt.zero_grad(set_to_none=True)
            ctx.alpha = alpha
            total, _ = problem.evaluate(alpha)
            total = total + sigma * space.mahalanobis_sq(alpha)
            total.backward()
            return total

        for _ in range(self.inner_iterations):
            opt.step(closure)
        return alpha.detach()

    def _curvature(self, problem, ctx, alpha) -> torch.Tensor:
        """Eigenvalues of ``A``, half the data term's Hessian in whitened coordinates.

        The exact regime assembles ``A = s Phi~^T W Phi~`` directly because it
        knows the residual. Nothing here does, so the same matrix is taken from
        autograd instead: ``H_alpha = grad^2 f(alpha)``, then whitened by
        ``H_z = Lam^1/2 H_alpha Lam^1/2`` since ``alpha = Lam^1/2 z``, then
        halved to match the convention in which ``(A + sigma I) z = b``.

        For a squared-distance data term this *is* the Gauss-Newton Hessian, not
        an approximation of it: the forward model ``x(alpha) = xbar + Phi alpha``
        is linear, so the second-order term that Gauss-Newton drops is the one
        weighted by the model's curvature, which is zero. The nearest-neighbour
        assignment contributes nothing either - it is piecewise constant in
        alpha, so within an assignment cell f is exactly quadratic and this is
        its Hessian. What the eigenvalues therefore describe is the cell the fit
        currently sits in; the clamp at zero covers the boundaries between cells
        and any robust kernel, where the true Hessian can turn indefinite and a
        covariance would stop meaning anything.

        Only the eigenvalues are needed, never the eigenvectors: the trace of
        ``Sigma_z = sigma (A + sigma I)^-1`` is a function of the spectrum alone.
        """
        space = problem.space
        root_lam = space.eigenvalues.sqrt()
        a = alpha.detach().clone().requires_grad_(True)
        ctx.alpha = a
        total, _ = problem.evaluate(a)                     # prior is 'none' - data term only
        grad = torch.autograd.grad(total, a, create_graph=True)[0]

        eye = torch.eye(a.numel(), dtype=a.dtype, device=a.device)
        rows = []
        for chunk in eye.split(max(1, self.hessian_chunk)):
            try:
                rows.append(torch.autograd.grad(
                    grad, a, grad_outputs=chunk, retain_graph=True, is_grads_batched=True
                )[0])
            except RuntimeError:                           # vmap refused some op in the loss
                rows.append(torch.stack([
                    torch.autograd.grad(grad, a, grad_outputs=row, retain_graph=True)[0]
                    for row in chunk
                ]))
        h_alpha = torch.cat(rows).detach()
        h_z = root_lam.unsqueeze(1) * h_alpha * root_lam.unsqueeze(0)
        h_z = 0.5 * (h_z + h_z.T)                          # symmetrise away the round-off
        return torch.linalg.eigvalsh((0.5 * h_z).double()).clamp_min(0.0)

    def _solve_general(self, problem, ctx, alpha, target_sq):
        space = problem.space
        cache, curv = {}, {}
        expected = self.constrain == "expected"
        # One curvature for the whole bracket unless asked otherwise. A is the
        # curvature of the data term, which barely moves over the small range of
        # multipliers being bracketed, and it enters only the correction term
        # tr(Sigma_z) rather than the radius itself - so freezing it at the warm
        # start buys back the (outer_iterations - 1) Hessians it would otherwise
        # cost. hessian_refresh: true for the exact version.
        frozen = self._curvature(problem, ctx, alpha) if expected and not self.hessian_refresh else None

        def radius_of(sigma):
            got = self._descend(problem, ctx, alpha, sigma)
            cache[sigma] = got
            z_sq = float(space.mahalanobis_sq(got))
            if not expected:
                return z_sq
            curv[sigma] = frozen if frozen is not None else self._curvature(problem, ctx, got)
            return z_sq + self._trace(curv[sigma], sigma)

        sigma = self._bisect(radius_of, target_sq, self.outer_iterations)
        if sigma not in cache:
            cache[sigma] = self._descend(problem, ctx, alpha, sigma)
        trace = self._trace(curv[sigma], sigma) if expected and sigma in curv else None
        return cache[sigma], sigma, trace

    # -- the loop -----------------------------------------------------------

    def solve(self, problem: FitProblem, log=None, on_iteration=None) -> SolveResult:
        self.validate(problem)
        space = problem.space
        target_sq = self.radius(space)
        exact = problem.loss.name == "matched_l2"
        ctx = problem.prepare(self.iterations)

        alpha = problem.alpha0.clone()
        history: List[Dict[str, float]] = []
        last_terms: Dict[str, float] = {}
        previous = None
        stalled = 0
        converged = False
        message = "iteration budget exhausted"

        if log:
            log(f"    [trust_region] {'exact' if exact else 'general'} regime, "
                f"constraining the {'expected radius' if self.constrain == 'expected' else 'posterior mean'} "
                f"to radius^2={target_sq:.6g} (chi2 p={self.p}, K={space.n_components})")

        for iteration in range(self.iterations):
            ctx.iteration = iteration
            if on_iteration:
                on_iteration(iteration, alpha, dict(last_terms))
            if _should_refresh(problem, iteration) and problem.correspondence.produces_matches:
                ctx.correspondences = problem.estimate_correspondences(alpha)

            if exact:
                alpha, sigma, trace = self._solve_matched(problem, ctx, target_sq)
            else:
                alpha, sigma, trace = self._solve_general(problem, ctx, alpha, target_sq)

            alpha = problem.projection(alpha, space)

            with torch.no_grad():
                ctx.alpha = alpha
                total, terms = problem.evaluate(alpha)
                radius_sq = float(space.mahalanobis_sq(alpha))
            last_terms = dict(terms)
            terms.update({"iteration": iteration, "sigma": sigma,
                          "mahalanobis_sq": radius_sq, "radius_sq": target_sq})
            if trace is not None:
                terms.update({"trace_sigma_z": trace, "expected_sq": radius_sq + trace})
            history.append(terms)

            if log and (iteration % max(1, self.log_every) == 0
                        or iteration == self.iterations - 1):
                # The constrained quantity is reported against the budget; the
                # other one trails it, which is the whole argument for "expected"
                # in one line - under constrain:mean the expected radius is
                # visibly the larger number, and it is not the one being held.
                if self.constrain == "expected" and trace is not None:
                    budget = (f"E|z|^2={radius_sq + trace:.6g}/{target_sq:.6g} "
                              f"(|z_hat|^2={radius_sq:.6g} + tr={trace:.6g})")
                else:
                    aside = f" (E|z|^2={radius_sq + trace:.6g})" if trace is not None else ""
                    budget = f"|z_hat|^2={radius_sq:.6g}/{target_sq:.6g}{aside}"
                log(f"    [{iteration + 1}/{self.iterations}] objective={float(total):.4e} "
                    f"| sigma={sigma:.4e} {budget}")

            current = float(total)
            if previous is not None and self.tolerance > 0:
                rel = abs(previous - current) / max(abs(previous), 1e-30)
                stalled = stalled + 1 if rel < self.tolerance else 0
                if stalled >= self.patience:
                    converged = True
                    message = f"relative improvement below {self.tolerance} for {stalled} rounds"
                    break
            previous = current

        return SolveResult(alpha=alpha, history=history, iterations=len(history),
                           converged=converged, message=message)
