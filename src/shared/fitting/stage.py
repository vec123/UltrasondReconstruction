"""The code_fit stage: fit the code of a frozen shape model to one target scan (code_fit.py).

    python scripts/run.py configs/code_fit/fused_target.yaml

The model is any ShapeModel kind (``model: {kind: gpmm|lddmm, ...}``), the target a
mesh or point cloud in scan mm. ``frame`` gives the similarity x_scan = s R x_model + t
from the model frame into the scan: ``aligned_template`` is the model's template moved
into the scan frame, vertex for vertex (the similarity is fitted on the two), and
``similarity`` is a {scale, rotation, translation} JSON. The fit runs in the unposed
model frame scaled to mm, as scripts/inspection/code_fit_fused_targets.py fits, so
the data term's sigma_mm and trunc_mm are in mm.

Writes into output_dir:
    fit.vtp        the fitted shape in scan mm, with the template's triangles
    code.json      the parameters theta, the code z, and the prior's label
    summary.json   the distances to the target in mm (target -> fit, fit -> target),
                   the code's squared Mahalanobis distance under the prior
    result.json    the record of the fit (src/shared/results.py), for the evaluation
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy.spatial import cKDTree
from vtk.util.numpy_support import vtk_to_numpy

from src.shared.fitting.code_fit import fit_codes
from src.shared.results import Result, similarity_record, write_results
from src.shared.shape_model import load_model
from src.shared.transforms.similarity import SimilarityTransform
from src.shared.vtk.create import create_polydata
from src.shared.vtk.io import load_vtp, save_vtp

DATA_TERMS = ("chamfer", "varifold")  # l2 needs corresponded observations, which a scan does not have
FIT_FILE = "fit.vtp"


@dataclass
class DataConfig:
    name: str = "chamfer"  # one of DATA_TERMS
    sigma_mm: float = 1.0  # weighs the data against the prior (see code_fit.py)
    trunc_mm: float = 2.0  # chamfer: cap of fit -> target; varifold: width of the mask
    points: int = 2000  # target points drawn per step
    fit_points: int = 6000  # chamfer: template vertices decoded per step
    fit_faces: int = 2000  # varifold: the fixed triangle subset per surface
    varifold_sigmas: List[float] = field(default_factory=lambda: [3.0, 6.0, 12.0])  # varifold: kernel widths (mm)


@dataclass
class FrameConfig:
    """Exactly one of the two: the model frame in the scan."""

    aligned_template: Optional[str] = None
    similarity: Optional[str] = None


@dataclass
class Config:
    target: str
    output_dir: str
    model: Dict[str, Any]  # a ShapeModel spec: {kind: ..., <its load() arguments>}
    frame: FrameConfig = field(default_factory=FrameConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prior_weight: float = 1.0  # 0: the data alone
    trainable: str = "code"  # only "code" so far: the model is frozen
    steps: int = 1000
    lr: float = 2.0e-4  # Adam step size of theta (the code's units; GPMM sigmas need about 0.05)
    lr_decay: bool = False  # cosine decay to 0 over the steps
    device: str = "cuda"
    seed: int = 0


def model_to_scan(frame: FrameConfig, model) -> SimilarityTransform:
    """The similarity x_scan = s R x_model + t of ``frame``."""
    if (frame.aligned_template is None) == (frame.similarity is None):
        raise ValueError("frame: give exactly one of aligned_template and similarity")
    if frame.similarity is not None:
        return SimilarityTransform.from_json(frame.similarity)
    aligned = vtk_to_numpy(load_vtp(frame.aligned_template).GetPoints().GetData()).astype(np.float64)
    template = np.asarray(model.template.points, dtype=np.float64)
    if aligned.shape != template.shape:
        raise ValueError(f"frame.aligned_template has {len(aligned)} points, the model's template {len(template)}; "
                         f"they must be the same mesh, vertex for vertex")
    return SimilarityTransform.from_correspondences(template, aligned, stage="model -> scan")


def run(cfg: Config):
    if cfg.trainable != "code":
        raise ValueError(f"trainable: {cfg.trainable!r} is not supported yet; only 'code' (a frozen model)")
    if cfg.data.name not in DATA_TERMS:
        raise ValueError(f"data.name: {cfg.data.name!r}; known: {list(DATA_TERMS)}")
    device = torch.device(cfg.device)
    model = load_model({"device": cfg.device, **cfg.model})
    to_scan = model_to_scan(cfg.frame, model)
    s, rotation, translation = to_scan.scale, to_scan.rotation, to_scan.translation

    target = load_vtp(cfg.target)
    target_scan = vtk_to_numpy(target.GetPoints().GetData()).astype(np.float64)
    target_mm = torch.as_tensor((target_scan - translation) @ rotation, dtype=torch.float32, device=device)
    prior = model.prior

    def decoder(idx, theta):
        return model.decode(prior.code(theta), idx) * s

    template_faces = None if model.template.faces is None else torch.as_tensor(model.template.faces, dtype=torch.long,
                                                                                device=device)
    varifold = {}
    if cfg.data.name == "varifold":
        polys = vtk_to_numpy(target.GetPolys().GetData()).reshape(-1, 4)[:, 1:] if target.GetNumberOfPolys() else None
        if polys is None or template_faces is None:
            raise ValueError("data.name varifold needs triangles on both the target and the model's template")
        varifold = dict(template_mm=torch.as_tensor(model.template.points * s, dtype=torch.float32, device=device),
                        template_faces=template_faces,
                        obs_triangles=[target_mm[torch.as_tensor(polys, dtype=torch.long, device=device)]],
                        fit_faces=cfg.data.fit_faces, varifold_sigmas=tuple(cfg.data.varifold_sigmas))

    theta = fit_codes(decoder, len(model.template.points), [target_mm], [torch.arange(len(target_mm), device=device)],
                      prior.mean, prior.precision, {"fit": (cfg.data.sigma_mm, cfg.prior_weight)}, cfg.steps, cfg.lr,
                      cfg.data.points, device, data=cfg.data.name, fit_points=cfg.data.fit_points,
                      trunc_mm=cfg.data.trunc_mm, lr_decay=cfg.lr_decay, seed=cfg.seed, **varifold)["fit"]
    with torch.no_grad():
        fit_mm = decoder(None, theta)[0].double().cpu().numpy()
    fit_scan = fit_mm @ rotation.T + translation
    _write(cfg, model, prior, theta, fit_scan, target_scan)


def _write(cfg, model, prior, theta, fit_scan, target_scan):
    os.makedirs(cfg.output_dir, exist_ok=True)
    fit_path = os.path.join(cfg.output_dir, FIT_FILE)
    save_vtp(create_polydata(fit_scan, model.template.faces), fit_path)

    with torch.no_grad():
        mahalanobis_sq = float(2 * prior.energy(theta)[0])
        z = prior.code(theta)[0]
    with open(os.path.join(cfg.output_dir, "code.json"), "w") as f:
        json.dump({"prior": prior.label, "theta": theta[0].tolist(), "z": z.tolist()}, f)

    target_to_fit = cKDTree(fit_scan).query(target_scan)[0]
    fit_to_target = cKDTree(target_scan).query(fit_scan)[0]
    summary = {
        "target_to_fit_mm": {"mean": float(target_to_fit.mean()), "p95": float(np.percentile(target_to_fit, 95))},
        "fit_to_target_mm": {"mean": float(fit_to_target.mean()), "p95": float(np.percentile(fit_to_target, 95))},
        "mahalanobis_sq": mahalanobis_sq,
        "n_parameters": int(prior.size),
    }
    with open(os.path.join(cfg.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    write_results(cfg.output_dir, {"fit": Result(
        recon=fit_path, target=cfg.target, recon_to_scan=similarity_record(SimilarityTransform.identity()),
        model=dict(cfg.model))})
    print(f"[code_fit] target -> fit {summary['target_to_fit_mm']['mean']:.3f} mm (p95 "
          f"{summary['target_to_fit_mm']['p95']:.3f}), Mahalanobis^2 {mahalanobis_sq:.2f} -> {cfg.output_dir}")
