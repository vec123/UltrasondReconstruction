"""A trained cohort ResNet-LDDMM run as a ShapeModel (src/shared/shape_model.py).

    decode(z) = phi_z(template) * scale + center,    the run's world frame

as the flow decoders of the inspection scripts did (without their mm factor).
The code is the flow's code z; code_prior says how a fit parametrises it, from
the run's training codes (as code_parametrization of
scripts/inspection/code_fit_partial_tests.py did):

    gauss         theta = z, prior N(mean, covariance) of the training codes; needs
                  more training codes than code entries, else the covariance is singular.
    pca           z = mean + U sqrt(lambda) theta, over the principal components of the
                  training codes that hold pca_var of their variance; prior 1/2 |theta|^2.
    pca_residual  theta = (s, r), z = mean + U sqrt(lambda) s + sqrt(v) r, v the training
                  codes' variance per entry; prior 1/2 |s|^2 + 1/2 sum_e w_e r_e^2. w_e
                  comes from each grid node's distance to the template surface in grid
                  cells: 0 up to residual_near, 1 from residual_far on, linear between,
                  mapped onto residual_w (low, high). A low w lets the data move that
                  node's region beyond the statistical part. Only for a direct-grid run.

A pair run (no code) is not a ShapeModel yet.
"""

import dataclasses

import numpy as np
import torch

from src.models.lddmm.resnet_flow.apply.trained_run import load_run
from src.models.lddmm.resnet_flow.runner import _normalization_domain
from src.shared.shape_model import CodePrior, ShapeModel, Template

CODE_PRIORS = ("gauss", "pca", "pca_residual")


def build_code_prior(kind, train_cfg, train_codes, template_norm, domain, device,
               pca_var=0.99, residual_w=(1.0, 100.0), residual_near=0.5, residual_far=1.0):
    """The CodePrior ``kind`` (see the module docstring) of a run's training codes [M, D]."""
    z = train_codes.double().numpy()
    mean = z.mean(0)
    d = z.shape[1]
    if kind == "gauss":
        prec = np.linalg.inv(np.cov(z, rowvar=False))
        return CodePrior(torch.zeros(d, device=device), torch.eye(d, device=device),
                         torch.as_tensor(mean, dtype=torch.float32, device=device),
                         torch.as_tensor(prec, dtype=torch.float32, device=device), "gauss")
    _, sv, vt = np.linalg.svd(z - mean, full_matrices=False)
    var = sv ** 2 / (len(z) - 1)
    k = int(np.searchsorted(np.cumsum(var) / var.sum(), pca_var) + 1)
    basis = torch.as_tensor((vt[:k].T * np.sqrt(var[:k])), dtype=torch.float32, device=device)   # [D, K]
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    if kind == "pca":
        return CodePrior(mean_t, basis, torch.zeros(k, device=device), torch.eye(k, device=device), f"pca{k}")
    if kind != "pca_residual":
        raise ValueError(f"unknown code_prior {kind!r}; known: {list(CODE_PRIORS)}")
    from src.models.lddmm.resnet_flow.conditioning.grid_regularization import grid_penalty_weights
    from src.models.lddmm.resnet_flow.config import GridRegCfg
    if not train_cfg.code.grid_direct:
        raise ValueError("code_prior pca_residual needs a direct-grid run (code.grid_direct)")
    t = template_norm.points[0].to(device)
    spatial = dataclasses.replace(train_cfg.code, grid_reg=GridRegCfg.from_dict(
        {"kind": "distance", "low": 0.0, "high": 1.0, "near": residual_near, "far": residual_far}))
    reg = grid_penalty_weights(spatial, [t], t, None, domain)[0].to(device).float()   # [D], 0..1
    low, high = residual_w
    w = low + (high - low) * reg
    v = torch.as_tensor(z.var(0), dtype=torch.float32, device=device).clamp_min(1e-12)
    prec = torch.zeros(k + d, k + d, device=device)
    prec[:k, :k] = torch.eye(k, device=device)
    prec[k:, k:] = torch.diag(w)
    return CodePrior(mean_t, torch.cat([basis, torch.diag(v.sqrt())], dim=1),
                     torch.zeros(k + d, device=device), prec, f"pca{k}_residual")


class FlowShapeModel(ShapeModel):
    kind = "lddmm"

    def __init__(self, flow, template_norm, transform, prior, train_codes, device="cpu"):
        self.flow = flow
        self.template_norm = template_norm
        self.points = template_norm.points[0].to(device)  # [N, 3] normalised
        self.center, self.scale = transform.center.to(device), transform.scale.to(device)
        self.prior = prior
        self.train_codes = train_codes  # [M, D], the codes the run learned for its cohort
        faces = template_norm.faces
        self.template = Template((self.points * self.scale + self.center).double().cpu().numpy(),
                                 None if faces is None else faces.cpu().numpy())

    @classmethod
    def load(cls, run, step=None, device="cpu", code_prior="gauss", pca_var=0.99,
             residual_w=(1.0, 100.0), residual_near=0.5, residual_far=1.0):
        """The cohort run folder ``run``, from final.pt (step None) or step_<step>.pt."""
        train_cfg, flow, train_codes, _, template_norm, transform = load_run(run, step, device)
        prior = build_code_prior(code_prior, train_cfg, train_codes, template_norm,
                                 _normalization_domain(train_cfg), device, pca_var,
                                 tuple(residual_w), residual_near, residual_far)
        return cls(flow, template_norm, transform, prior, train_codes, device)

    def decode(self, z, idx=None):
        template = self.points.expand(len(z), -1, -1) if idx is None else self.points[idx]
        return self.flow(template, z).points[-1] * self.scale + self.center
