"""A GPMM as a ShapeModel (src/shared/shape_model.py): the code is the coefficients in sigma units.

    decode(s) = mean + Phi sqrt(lambda) s,    prior N(0, I)

in the model's own frame (the frame of its mean), as ssm_decoder of
scripts/studies/code_fit_partial_tests.py did without its mm factor.
"""

import numpy as np
import torch

from src.models.gpmm.model.model_memory_io import load_model_file
from src.shared.shape_model import CodePrior, ShapeModel, Template


class GpmmShapeModel(ShapeModel):
    kind = "gpmm"

    def __init__(self, model, device="cpu"):
        """``model``: a StatisticalShapeModel."""
        self.model = model
        self.mean = torch.as_tensor(model.mean.reshape(-1, 3), dtype=torch.float32, device=device)
        self.basis = torch.as_tensor((model.eigenfunctions * np.sqrt(np.clip(model.eigenvalues, 0, None)))
                                     .reshape(-1, 3, model.n_components), dtype=torch.float32, device=device)
        landmarks = {}
        if model.landmarks and "verts " in model.landmarks:
            landmarks = dict(zip(model.landmark_names(), model.landmark_indices().tolist()))
        self.template = Template(model.mean.reshape(-1, 3), model.faces, landmarks)
        self.prior = CodePrior.standard(model.n_components, device, label="gpmm")

    @classmethod
    def load(cls, path, device="cpu"):
        """The model in ``path`` (.npz, or a legacy source that load_model_file reads)."""
        return cls(load_model_file(path), device)

    def decode(self, z, idx=None):
        if idx is None:
            return self.mean + torch.einsum("nck,tk->tnc", self.basis, z)
        return self.mean[idx] + torch.einsum("tmck,tk->tmc", self.basis[idx], z)
