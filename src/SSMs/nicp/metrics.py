"""Fit quality, measured the same way whatever objective produced the fit.

A run driven by Chamfer and a run driven by a matched L2 term report different
loss values that cannot be compared with each other. These metrics are computed
after the fact from the geometry alone, so a sweep over losses, priors and
solvers can still be ranked on one table.

Both directions are reported and both are needed: ``model_to_target`` alone
rewards a model that collapses onto the scanned patch, ``target_to_model``
alone rewards one that covers the scan while drifting elsewhere.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree


def surface_metrics(model_verts: np.ndarray, target_points: np.ndarray,
                    target_tree: Optional[cKDTree] = None) -> Dict[str, float]:
    """Point-to-point distance statistics between the fitted model and the target."""
    model_verts = np.asarray(model_verts, dtype=np.float64).reshape(-1, 3)
    target_points = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)

    tree_t = target_tree if target_tree is not None else cKDTree(target_points)
    forward, _ = tree_t.query(model_verts)            # model -> target
    backward, _ = cKDTree(model_verts).query(target_points)   # target -> model

    return {
        "model_to_target_rms": float(np.sqrt((forward ** 2).mean())),
        "model_to_target_mean": float(forward.mean()),
        "model_to_target_p95": float(np.percentile(forward, 95)),
        "target_to_model_rms": float(np.sqrt((backward ** 2).mean())),
        "target_to_model_mean": float(backward.mean()),
        "target_to_model_p95": float(np.percentile(backward, 95)),
        "chamfer_l2": float((forward ** 2).mean() + (backward ** 2).mean()),
        "hausdorff": float(max(forward.max(), backward.max())),
    }


def coefficient_metrics(alpha: np.ndarray, eigenvalues: np.ndarray) -> Dict[str, float]:
    """How far the fit has moved through the model's own coordinates."""
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)

    mahalanobis_sq = float((alpha ** 2 / eigenvalues).sum())
    return {
        "alpha_norm": float(np.linalg.norm(alpha)),
        "mahalanobis": float(np.sqrt(mahalanobis_sq)),
        "max_sd": float(np.abs(alpha / np.sqrt(eigenvalues)).max()) if alpha.size else 0.0,
    }


def displacement_metrics(fitted: np.ndarray, initial: np.ndarray) -> Dict[str, float]:
    """How far the fit moved the surface from where it started."""
    delta = np.linalg.norm(
        np.asarray(fitted).reshape(-1, 3) - np.asarray(initial).reshape(-1, 3), axis=1
    )
    return {"displacement_mean": float(delta.mean()), "displacement_max": float(delta.max())}
