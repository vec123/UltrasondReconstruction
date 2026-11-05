"""One interface for the shape models of this repo: a code in, the points of the template out.

A ShapeModel decodes codes z [T, D] into its template's points, in the model's
own frame and units (decode). Its CodePrior says which codes are likely,
through the parameters theta [T, P] that a fit optimises:

    z = offset + theta @ basis.T,    energy(theta) = 1/2 (theta - mean)^T precision (theta - mean)

The kinds of this repo:

    gpmm   src/models/gpmm/shape_model.py    z = theta = the coefficients in sigma units, N(0, I)
    lddmm  src/models/lddmm/shape_model.py   z = the flow's code, the prior of the run's training
                                             codes (gauss, pca or pca_residual)

A YAML names a model by its kind and the arguments of that kind's load():

    model: {kind: gpmm, path: ${PROJECT_ROOT}/assets/shape_models/baby/baby_pca.npz}
    model: {kind: lddmm, run: ${OUT}/resnet_lddmm_cohort_..., code_prior: pca}

MODEL_KINDS maps each kind to its "module:Class" string. The composition root
(scripts/pipeline/registry.py) fills it from the registrations of the layers,
so this module imports no model, and loading a model imports only its kind's
code. torch is imported only where a prior computes.
"""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from src.shared.core.registry import LazyRegistry

CATEGORY = "model"
MODEL_KINDS = LazyRegistry()


@dataclass
class Template:
    """The points that a model's codes move, in the model frame."""

    points: np.ndarray  # [N, 3]
    faces: Optional[np.ndarray] = None  # [F, 3]
    landmarks: Dict[str, int] = field(default_factory=dict)  # name -> vertex index; empty if the model has none


@dataclass
class CodePrior:
    """z = offset + theta @ basis.T; energy(theta) = 1/2 (theta - mean)^T precision (theta - mean). torch tensors."""

    offset: Any  # [D]
    basis: Any  # [D, P]
    mean: Any  # [P], where a fit starts
    precision: Any  # [P, P]
    label: str = ""

    @property
    def size(self):
        """P, the number of parameters."""
        return self.mean.shape[0]

    def code(self, theta):
        """[T, P] -> the codes [T, D]."""
        return self.offset + theta @ self.basis.T

    def energy(self, theta):
        """[T, P] -> [T]."""
        d = theta - self.mean
        return 0.5 * ((d @ self.precision) * d).sum(-1)

    def sample(self, n, generator=None):
        """[n, P] parameters drawn from the prior."""
        import torch

        covariance_factor = torch.linalg.cholesky(torch.linalg.inv(self.precision))
        noise = torch.randn(n, self.size, generator=generator, dtype=self.mean.dtype, device=self.mean.device)
        return self.mean + noise @ covariance_factor.T

    @classmethod
    def standard(cls, size, device="cpu", label="standard"):
        """theta = z, N(0, I)."""
        import torch

        eye = torch.eye(size, device=device)
        zeros = torch.zeros(size, device=device)
        return cls(zeros, eye, zeros.clone(), eye.clone(), label)


class ShapeModel(ABC):
    """A template and a decoder of codes; see the module docstring. A kind sets template and prior."""

    kind = ""
    template: Template
    prior: CodePrior

    @abstractmethod
    def decode(self, z, idx=None):
        """codes z [T, D] -> [T, M, 3] torch: the template vertices idx ([T, M] long; None for all of
        them) moved by z, in the model frame. Differentiable in z."""

    @classmethod
    @abstractmethod
    def load(cls, **kwargs) -> "ShapeModel":
        """The model that a YAML ``model:`` mapping names (its keys except ``kind``)."""


def load_model(spec, registry=MODEL_KINDS) -> ShapeModel:
    """The model of a YAML ``model:`` mapping: ``{kind: <name>, <arguments of that kind's load()>}``."""
    spec = dict(spec)
    kind = spec.pop("kind", None)
    if kind is None:
        raise ValueError(f"A model needs 'kind:', one of {registry.available(CATEGORY)}.")
    cls = registry.resolve(CATEGORY, kind)
    accepted = list(inspect.signature(cls.load).parameters)
    unknown = sorted(set(spec) - set(accepted))
    if unknown:
        raise ValueError(f"Unknown key(s) {unknown} for model kind '{kind}'; accepted: {accepted}.")
    return cls.load(**spec)
