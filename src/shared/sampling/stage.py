"""Stage: shapes from a shape model's codes, as files a cohort run reads.

    stage: sample
    model: {kind: gpmm, path: assets/shape_models/baby/baby_pca.npz}
    codes: {source: prior, n: 1000, sampling: sobol, seed: 0}
    output_dir: ...

Any ShapeModel (src/shared/shape_model.py) works. Each code is

    theta = prior mean + L s,   L L^T = the prior covariance,   z = prior.code(theta)

with s standard normal per parameter, from one of three sources:

    prior   n draws of s (draws.standard_draws: random or sobol, clip), on the
            first `parameters` parameters; the others stay at the prior mean
    modes   s = step on one parameter, for each of the first `parameters`
            parameters and each of `steps` (in standard deviations)
    file    the `sigmas` of a coefficients.npz written before

For a GPMM, s = theta = z: the mode coefficients in standard deviations.

Output layout (`output_dir`), the same as in the cohort folders written before this stage:

  members/sample_00000.vtp ...   the shapes; file order = code index
  template.vtp                   the shape of the prior mean (a GPMM: its mean)
  coefficients.npz               sigmas = s [n, K], theta [n, P], z [n, D], vertex_idx,
                                 source, sampling, seed, model (the spec, JSON), prior

`subset: N` keeps the same N random vertices on every shape and on the
template, so correspondence holds. The meshes then have no faces.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.shared.sampling.draws import standard_draws
from src.shared.shape_model import load_model
from src.shared.vtk.create import create_polydata
from src.shared.vtk.io import save_vtp

COEFFICIENTS_FILE = "coefficients.npz"
SOURCES = ("prior", "modes", "file")


@dataclass
class CodesConfig:
    source: str = "prior"  # prior | modes | file (see the module docstring)
    n: int = 1000  # prior: number of shapes
    sampling: str = "random"  # prior: random | sobol
    seed: int = 0  # prior: seed of the draws; all sources: seed of the subset
    clip: Optional[float] = None  # prior: clip each draw to +-clip
    parameters: Optional[int] = None  # prior, modes: the first K parameters only; None = all
    steps: List[float] = field(default_factory=lambda: [-2.0, -1.0, 0.0, 1.0, 2.0])  # modes: in SD
    file: str = ""  # file: a coefficients.npz; its `sigmas` are used


@dataclass
class Config:
    model: Dict[str, Any]  # a ShapeModel spec: {kind: ..., <its load() arguments>}
    output_dir: str
    codes: CodesConfig = field(default_factory=CodesConfig)
    subset: int = 0  # 0: the full mesh; N: the same N random vertices on every shape, no faces
    batch: int = 64  # shapes decoded at once
    device: str = "cpu"


def standard_codes(codes: CodesConfig, size):
    """s [n, K] of `codes`, K <= size (the prior's number of parameters)."""
    k = size if codes.parameters is None else int(codes.parameters)
    if not 0 < k <= size:
        raise ValueError(f"codes.parameters must be in 1..{size}, got {k}")
    if codes.source == "prior":
        return standard_draws(codes.n, k, codes.sampling, codes.seed, codes.clip)
    if codes.source == "modes":
        return np.concatenate([np.outer(codes.steps, np.eye(k)[i]) for i in range(k)])
    if codes.source == "file":
        sigmas = np.load(codes.file)["sigmas"]
        if sigmas.shape[1] > size:
            raise ValueError(f"{codes.file} has {sigmas.shape[1]} parameters; the model has {size}")
        return sigmas
    raise ValueError(f"codes.source must be one of {SOURCES}, got {codes.source!r}")


def run(cfg: Config):
    out = Path(cfg.output_dir)
    members_dir = out / "members"
    if members_dir.exists() and any(members_dir.glob("*.vtp")):
        raise FileExistsError(f"{members_dir} already holds .vtp files; choose an empty output_dir")

    model = load_model(cfg.model) if "device" in cfg.model else load_model({**cfg.model, "device": cfg.device})
    prior = model.prior
    s = standard_codes(cfg.codes, prior.size)
    k = s.shape[1]
    with torch.no_grad():
        factor = torch.linalg.cholesky(torch.linalg.inv(prior.precision))[:, :k]
        theta = prior.mean + torch.as_tensor(s, dtype=prior.mean.dtype, device=prior.mean.device) @ factor.T
        z = prior.code(theta)
        template_points = model.decode(prior.code(prior.mean[None]))[0].cpu().numpy()

    vertex_idx = np.arange(len(template_points))
    faces = model.template.faces
    if cfg.subset:
        if not 0 < cfg.subset < len(template_points):
            raise ValueError(f"subset must be in 1..{len(template_points) - 1}, got {cfg.subset}")
        rng = np.random.default_rng(cfg.codes.seed + 1)
        vertex_idx = np.sort(rng.choice(len(template_points), cfg.subset, replace=False))
        faces = None  # the triangulation does not survive dropping vertices

    save_vtp(create_polydata(template_points[vertex_idx], faces), str(out / "template.vtp"))
    for start in range(0, len(z), cfg.batch):
        with torch.no_grad():
            points = model.decode(z[start:start + cfg.batch]).cpu().numpy()
        for i, shape in enumerate(points, start):
            save_vtp(create_polydata(shape[vertex_idx], faces), str(members_dir / f"sample_{i:05d}.vtp"))
    np.savez(out / COEFFICIENTS_FILE, sigmas=s, theta=theta.cpu().numpy(), z=z.cpu().numpy(),
             vertex_idx=vertex_idx, source=cfg.codes.source, sampling=cfg.codes.sampling, seed=cfg.codes.seed,
             model=json.dumps(cfg.model), prior=prior.label)

    print(f"wrote {len(z)} shapes ({len(vertex_idx)} vertices, {k} of {prior.size} parameters, "
          f"source {cfg.codes.source}, prior {prior.label}) + template to {out}")
