"""Data terms: how far the model surface is from the target surface.

A loss is a callable ``(predicted vertices, context) -> LossValue``. The
context carries the target, the current correspondences *if any*, the
coefficients and where the fit has got to, so a loss can decide for itself how
the two surfaces are associated. That single ``Optional`` is what separates the
two families:

    matched_l2   consumes a fixed pairing produced by a correspondence method
    chamfer/cpd  ignore the pairing and associate the surfaces themselves,
                 fresh at every iteration and in both directions

The config selects one by name and passes its keyword arguments:

    loss:
      name: chamfer
      kwargs:
        source_to_target: {weight: 1.0, base_sigma: 1.0}
        target_to_source: {weight: 5.0, base_sigma: 1.0, field_name: consistency_score}

Nearest-neighbour indices inside the correspondence-free losses are taken
without gradient and the distance is recomputed from the live tensors, which is
the standard treatment: the index is piecewise constant in the coefficients, so
it carries no gradient of its own.
"""

from __future__ import annotations

from .base import (  # noqa: F401
    LOSSES,
    Loss,
    LossContext,
    LossValue,
    _check_ratio,
    _reduce,
    _subsample,
    available_losses,
    build_loss,
    register_loss,
)
from .chamfer import ChamferLoss  # noqa: F401
from .composite import CompositeLoss  # noqa: F401
from .cpd import CpdLoss  # noqa: F401
from .matched import MatchedL2  # noqa: F401
from .plane import P2PlaneChamferLoss, PointToPlaneLoss, _mesh_edges  # noqa: F401
