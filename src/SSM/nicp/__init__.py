"""NICP building blocks.

Each part of a fit is picked by name from a registry and configured with its own
keyword arguments, so the four axes vary independently:

    correspondence  none | nearest | constrained
    loss            matched_l2 | chamfer | point_to_plane | cpd | composite
    prior           none | isotropic | mahalanobis
    solver          normal_equations | gradient
    projection      none | chi_clamp | mahalanobis_clamp

A classic NICP step is (constrained, matched_l2, isotropic, normal_equations);
a correspondence-free fit is (none, chamfer, mahalanobis, gradient). Nothing in
the code special-cases either combination.
"""

from src.SSM.nicp.cloud import PointCloud
from src.SSM.nicp.correspondence import (
    ConstrainedMatching,
    CorrespondenceError,
    Correspondences,
    CorrespondenceStrategy,
    NearestNeighbourMatching,
    NoCorrespondence,
    SymmetricMatching,
    available_correspondences,
    build_correspondence,
    register_correspondence,
)
from src.SSM.nicp.losses import (
    ChamferLoss,
    CompositeLoss,
    CpdLoss,
    Loss,
    LossContext,
    LossValue,
    MatchedL2,
    P2PlaneChamferLoss,
    PointToPlaneLoss,
    available_losses,
    build_loss,
    register_loss,
)
from src.SSM.nicp.metrics import coefficient_metrics, displacement_metrics, surface_metrics
from src.SSM.nicp.priors import Prior, available_priors, build_prior, register_prior
from src.SSM.nicp.problem import FitProblem
from src.SSM.nicp.projections import (
    Projection,
    available_projections,
    build_projection,
    register_projection,
)
from src.SSM.nicp.registry import Registry, RegistryError
from src.SSM.nicp.shape import ShapeSpace
from src.SSM.nicp.weight_fields import ChamferSide
from src.SSM.nicp.sinkhorn import SinkhornMatching, unbalanced_sinkhorn
from src.SSM.nicp.solvers import (
    GradientSolver,
    NormalEquationsSolver,
    SolveResult,
    Solver,
    available_solvers,
    build_solver,
    register_solver,
)

__all__ = [
    "PointCloud",
    "ShapeSpace",
    "ChamferSide",
    "FitProblem",
    "Registry",
    "RegistryError",
    # correspondence
    "ConstrainedMatching",
    "CorrespondenceError",
    "Correspondences",
    "CorrespondenceStrategy",
    "NearestNeighbourMatching",
    "NoCorrespondence",
    "SymmetricMatching",
    "SinkhornMatching",
    "unbalanced_sinkhorn",
    "available_correspondences",
    "build_correspondence",
    "register_correspondence",
    # losses
    "ChamferLoss",
    "CompositeLoss",
    "CpdLoss",
    "Loss",
    "LossContext",
    "LossValue",
    "MatchedL2",
    "P2PlaneChamferLoss",
    "PointToPlaneLoss",
    "available_losses",
    "build_loss",
    "register_loss",
    # priors / projections
    "Prior",
    "available_priors",
    "build_prior",
    "register_prior",
    "Projection",
    "available_projections",
    "build_projection",
    "register_projection",
    # solvers
    "GradientSolver",
    "NormalEquationsSolver",
    "SolveResult",
    "Solver",
    "available_solvers",
    "build_solver",
    "register_solver",
    # metrics
    "coefficient_metrics",
    "displacement_metrics",
    "surface_metrics",
]
