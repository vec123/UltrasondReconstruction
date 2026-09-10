"""The assembled fit: model, target, and the four components that define it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.SSM.nicp.cloud import PointCloud
from src.SSM.nicp.correspondence import CorrespondenceStrategy, Correspondences
from src.SSM.nicp.losses import Loss, LossContext
from src.SSM.nicp.priors import Prior
from src.SSM.nicp.projections import Projection
from src.SSM.nicp.shape import ShapeSpace


class FitProblem:
    """Everything a solver needs, with the target uploaded once.

    ``refresh_every`` turns a fit into a proper ICP loop: 0 estimates the
    correspondences once from the initial shape (what the original pipeline
    does), n > 0 re-estimates them every n iterations from the shape as it
    stands. It has no effect on a correspondence-free loss, which re-associates
    the surfaces on every evaluation anyway.
    """

    def __init__(
        self,
        space: ShapeSpace,
        target: PointCloud,
        correspondence: CorrespondenceStrategy,
        loss: Loss,
        prior: Prior,
        projection: Projection,
        alpha0=None,
        refresh_every: int = 0,
        seed: Optional[int] = None,
    ):
        self.space = space
        self.target = target
        self.correspondence = correspondence
        self.loss = loss
        self.prior = prior
        self.projection = projection
        self.refresh_every = int(refresh_every)
        self.alpha0 = space.as_alpha(alpha0)

        if self.refresh_every < 0:
            raise ValueError(f"refresh_every must be >= 0, got {self.refresh_every}")
        if loss.requires_correspondences and not correspondence.produces_matches:
            raise ValueError(
                f"loss '{loss.name}' needs correspondences but the correspondence method is "
                f"'{correspondence.name}', which produces none. Pair '{loss.name}' with "
                f"'nearest'/'constrained', or use a correspondence-free loss."
            )

        self.target_points = torch.as_tensor(
            target.points, dtype=space.dtype, device=space.device
        )
        generator = None
        if seed is not None:
            generator = torch.Generator(device=space.device)
            generator.manual_seed(int(seed))

        self.context = LossContext(
            target=self.target_points,
            target_cloud=target,
            space=space,
            alpha=self.alpha0,
            generator=generator,
        )

    def vertices(self, alpha: torch.Tensor) -> torch.Tensor:
        return self.space.reconstruct(alpha)

    def estimate_correspondences(self, alpha: torch.Tensor) -> Optional[Correspondences]:
        verts = self.space.to_numpy(alpha).reshape(-1, 3)
        return self.correspondence.estimate(verts, self.target)

    def wants_normals(self) -> bool:
        """Target normals are only worth their cost when a term actually reads them."""
        terms = getattr(self.loss, "losses", [self.loss])
        return any(getattr(item, "needs_target_normals", False) for item in terms)

    def prepare(self, n_iterations: int) -> LossContext:
        ctx = self.context
        ctx.n_iterations = max(1, int(n_iterations))
        ctx.iteration = 0
        if self.wants_normals() and ctx.target_normals is None:
            ctx.target_normals = torch.as_tensor(
                self.target.normals, dtype=self.space.dtype, device=self.space.device
            )
        return ctx

    def evaluate(self, alpha: torch.Tensor):
        """Data term + prior at ``alpha``; differentiable when alpha requires grad."""
        pred = self.vertices(alpha)
        value = self.loss(pred, self.context)
        penalty = self.prior(alpha, self.space)
        total = value.total + penalty

        terms = dict(value.terms)
        terms["prior"] = float(penalty.detach())
        terms["objective"] = float(total.detach())
        return total, terms

    def describe(self) -> str:
        return (
            f"correspondence={self.correspondence.describe()}\n"
            f"  loss         ={self.loss.describe()}\n"
            f"  prior        ={self.prior.describe()}\n"
            f"  projection   ={self.projection.describe()}"
        )


def build_problem(spec, space: ShapeSpace, target: PointCloud,
                  alpha0=None, seed: Optional[int] = None) -> FitProblem:
    """Resolves a :class:`~src.config.SSM.schema.NicpFitSpec` into a FitProblem.

    The single place the five registries are turned into an assembled fit, so
    the standalone runner and the pipeline stage build the same problem from
    the same spec and cannot drift apart.
    """
    from src.SSM.nicp.correspondence import build_correspondence
    from src.SSM.nicp.losses import build_loss
    from src.SSM.nicp.priors import build_prior
    from src.SSM.nicp.projections import build_projection

    return FitProblem(
        space=space,
        target=target,
        correspondence=build_correspondence(spec.correspondence.name, spec.correspondence.kwargs),
        loss=build_loss(spec.loss.name, spec.loss.kwargs),
        prior=build_prior(spec.prior.name, spec.prior.kwargs),
        projection=build_projection(spec.projection.name, spec.projection.kwargs),
        alpha0=alpha0,
        refresh_every=spec.refresh_every,
        seed=seed,
    )
