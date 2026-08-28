"""Torch view of the statistical shape model.

The forward model is linear, ``x(alpha) = mean + Phi @ alpha``, so autodiff over
alpha costs one matrix product per direction and the same object serves the
gradient solvers and the closed-form ones. The heavy arrays are uploaded once
and shared by every fit in a run.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.model.model import StatisticalShapeModel


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return {"float32": torch.float32, "float64": torch.float64}[str(dtype)]
    except KeyError:
        raise ValueError(f"dtype must be float32 or float64, got {dtype!r}")


class ShapeSpace:
    """``mean``, ``Phi`` and the eigenvalues as tensors, plus the forward model."""

    def __init__(self, mean, eigenfunctions, eigenvalues, faces=None,
                 device="auto", dtype=torch.float64):
        self.device = resolve_device(device) if isinstance(device, str) else device
        self.dtype = resolve_dtype(dtype)
        #: (F, 3) triangulation shared by every instance, or None. With it, an
        #: instance is a surface and its normals are oriented; without it, every
        #: instance is a point cloud and normals come from a local SVD whose
        #: sign is arbitrary.
        self.faces = None if faces is None else np.asarray(faces, dtype=np.int64)

        self.mean = torch.as_tensor(np.asarray(mean).reshape(-1), dtype=self.dtype, device=self.device)
        self.Phi = torch.as_tensor(np.asarray(eigenfunctions), dtype=self.dtype, device=self.device)
        self.eigenvalues = torch.as_tensor(
            np.asarray(eigenvalues).reshape(-1), dtype=self.dtype, device=self.device
        )

        if self.Phi.shape[0] != self.mean.shape[0]:
            raise ValueError(f"Phi has {self.Phi.shape[0]} rows, mean has {self.mean.shape[0]}")
        if self.eigenvalues.shape[0] != self.Phi.shape[1]:
            raise ValueError(
                f"{self.eigenvalues.shape[0]} eigenvalues for {self.Phi.shape[1]} eigenfunctions"
            )

    @classmethod
    def from_model(cls, model: StatisticalShapeModel, faces=None,
                   device="auto", dtype=torch.float64) -> "ShapeSpace":
        return cls(model.mean, model.eigenfunctions, model.eigenvalues,
                   faces=faces, device=device, dtype=dtype)

    @property
    def n_vertices(self) -> int:
        return self.mean.shape[0] // 3

    @property
    def n_components(self) -> int:
        return self.Phi.shape[1]

    def zeros(self) -> torch.Tensor:
        return torch.zeros(self.n_components, dtype=self.dtype, device=self.device)

    def as_alpha(self, alpha=None) -> torch.Tensor:
        """Coefficients on this space's device and dtype, from a tensor or an array."""
        if alpha is None:
            return self.zeros()
        if isinstance(alpha, torch.Tensor):
            return alpha.detach().to(device=self.device, dtype=self.dtype).reshape(-1)
        return torch.as_tensor(np.asarray(alpha).reshape(-1), dtype=self.dtype, device=self.device)

    def reconstruct(self, alpha: torch.Tensor) -> torch.Tensor:
        """(N, 3) vertices for the given coefficients; differentiable in ``alpha``."""
        return (self.mean + self.Phi @ alpha).view(-1, 3)

    def phi_rows(self, vertex_idx) -> torch.Tensor:
        """(3P, K) block of Phi for P vertices - the design matrix of a matched fit."""
        idx = torch.as_tensor(np.asarray(vertex_idx), dtype=torch.long, device=self.device)
        return self.Phi.view(self.n_vertices, 3, self.n_components)[idx].reshape(-1, self.n_components)

    def to_numpy(self, alpha: torch.Tensor) -> np.ndarray:
        """Flat (3N,) reconstruction, matching ShapeInstance.reconstruct()."""
        with torch.no_grad():
            return self.reconstruct(alpha).reshape(-1).double().cpu().numpy()

    def mahalanobis_sq(self, alpha: torch.Tensor) -> torch.Tensor:
        """sum (alpha_i^2 / lambda_i) - how implausible the instance is under the SSM."""
        return (alpha ** 2 / self.eigenvalues).sum()

    def __repr__(self) -> str:
        return (
            f"ShapeSpace(n_vertices={self.n_vertices}, n_components={self.n_components}, "
            f"device={self.device}, dtype={str(self.dtype).replace('torch.', '')})"
        )
