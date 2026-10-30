"""Loads the shapes of a pair or of a cohort and finds the cohort files."""

import os
import glob
from src.resnet_lddmm.resnet_flow.io import Shape, joint_normalize, load_shape

from .common import _normalization_domain


def load_pair(cfg):
    """cfg.source/cfg.target -> both shapes, normalised into ONE shared frame.

    The frame is a pure function of the two FILES, so calling this again after a
    run reproduces exactly the frame the weights were trained in. That is what
    lets apply put its output back into world coordinates without the checkpoint
    having to carry the transform.

    Args:
        cfg: ExperimentCfg with .source, .target and .centering

    Returns:
        (source_norm, target_norm, transform)
    """
    normalized, transform = joint_normalize(
        [load_shape(cfg.source), load_shape(cfg.target)],
        domain=_normalization_domain(cfg),
    )
    return normalized[0], normalized[1], transform


def cohort_paths(cfg):
    """The cohort member files in cfg.source, in the order their codes are indexed."""
    return _shape_files(cfg.source)


def _shape_files(folder):
    """The shape files (.obj, then .ply, then .vtp, each sorted) in `folder`."""
    paths = sorted(glob.glob(os.path.join(folder, "*.obj"))) + \
            sorted(glob.glob(os.path.join(folder, "*.ply"))) + \
            sorted(glob.glob(os.path.join(folder, "*.vtp")))
    if not paths:
        raise ValueError(f"No shape files found in {folder} (.obj, .ply, or .vtp)")
    return paths


def load_cohort(cfg):
    """cfg.source (a folder of cohort members) + cfg.target (the template) -> one shared frame.

    The cohort counterpart of load_pair: the frame is a pure function of the FILES,
    so apply recomputes exactly the frame the weights were trained in.

    Args:
        cfg: ExperimentCfg in cohort mode

    Returns:
        (paths, cohort_norm, template_norm, transform), member i of cohort_norm
        being paths[i] and holding code i
    """
    paths = cohort_paths(cfg)
    shapes = [load_shape(p) for p in paths] + [load_shape(cfg.target)]
    frame_source = getattr(cfg, "frame_source", None)
    if frame_source is None:
        normalized, transform = joint_normalize(shapes, domain=_normalization_domain(cfg))
    else:
        # The frame of another cohort (a trained run's source), so new targets enter
        # the frame its flow was trained in; then this cohort is moved into it.
        frame_shapes = [load_shape(p) for p in _shape_files(frame_source)] + [load_shape(cfg.target)]
        _, transform = joint_normalize(frame_shapes, domain=_normalization_domain(cfg))
        normalized = [Shape(points=transform.apply(s.points), faces=s.faces, weights=s.weights,
                            normals=s.normals, point_fields=s.point_fields) for s in shapes]
    return paths, normalized[:-1], normalized[-1], transform
