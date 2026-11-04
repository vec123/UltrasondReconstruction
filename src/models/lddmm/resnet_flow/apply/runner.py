"""Composition root for inference: config -> client -> files on disk.

The mirror of runner.build/run, for a model that already exists. It resolves the
things a checkpoint cannot carry (which architecture, which frame, which code),
hands them to a DeformationClient, and writes the result out.

What it deliberately does NOT do:

  * subsample. ``loss.subsample_M`` is a TRAINING economy -- it draws a fresh
    random subset each step so the gradient is cheap. There is no gradient here,
    so the template goes through whole. This is the entire point of applying
    separately rather than reading the training run's exports.
  * augment. ``augmentation`` perturbs the sample to train invariance into the
    model; replaying it at inference would just add noise to the answer.
  * optimize. No loss, no optimizer, no grad.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace as dataclass_replace
from typing import Optional, Tuple

import torch

from src.models.lddmm.resnet_flow.callbacks.checkpoints import (
    CheckpointRef,
    checkpoint_step,
    load_checkpoint,
    resolve_checkpoint,
)
from src.models.lddmm.resnet_flow.apply.client import Deformation, DeformationClient
from src.models.lddmm.resnet_flow.apply.config import (
    ApplyCfg,
    load_apply_config,
    load_training_config,
)
from src.models.lddmm.resnet_flow.config import ExperimentCfg
from src.models.lddmm.resnet_flow.io import (
    FrameTransform,
    Shape,
    export_reference_shapes,
    export_shape,
    export_trajectory,
    load_shape,
)
from src.models.lddmm.resnet_flow import registrations  # Side effect: registers all components
from src.models.lddmm.resnet_flow.runner import (
    build_code_source,
    build_code_source_cohort,
    load_cohort,
    load_pair,
    set_default_dtype,
)
from src.models.lddmm.outputs import ApplyOutputFiles

_FILES = ApplyOutputFiles()
MEAN_RECON = "mean_recon"


def run(apply_cfg: ApplyCfg):
    """Apply a trained flow to a template and write the result.

    Args:
        apply_cfg: ApplyCfg, unresolved (resolved against the training config here)

    Returns:
        (deformation, manifest): the Deformation, and the dict written to apply.json
    """
    train_cfg = load_training_config(apply_cfg)

    # Which .pt FIRST: a pinned checkpoint tags the output directory with its name,
    # so where the results go is not known until the file has been chosen.
    ref = resolve_checkpoint(apply_cfg.checkpoint, apply_cfg.step)
    cfg = apply_cfg.resolved(train_cfg, ref)
    print(f"[apply] {ref.path} ({'pinned' if ref.pinned else 'resolved'}) "
          f"-> {cfg.output_dir}")

    # Before ANY tensor is made: load_shape reads torch.get_default_dtype(), so a
    # run trained in float64 has to set that here or its template loads as float32
    # and the checkpoint will not take.
    set_default_dtype(train_cfg)

    if train_cfg.train.mode == "cohort":
        return run_cohort(cfg, train_cfg, ref)

    # The frame is recomputed from the two TRAINING files, so it is bit-for-bit the
    # one the weights were fitted in -- see runner.load_pair. The normalised source
    # it returns is the training template; the template actually deformed is loaded
    # separately below, because a config may point at a different (e.g. denser) file.
    _, target_norm, transform = load_pair(train_cfg)

    checkpoint = load_checkpoint(ref, map_location=cfg.device)
    code, pose = resolve_conditioning(train_cfg, checkpoint, target_norm)

    client = DeformationClient.from_checkpoint(
        train_cfg, checkpoint, transform,
        code=code, pose=pose, device=cfg.device, chunk_size=cfg.chunk_size,
    )

    template = load_shape(cfg.template)
    print(f"[apply] {cfg.template}: {template.points.shape[1]} points "
          f"(training subsampled to {train_cfg.loss.resolve_subsample_count(template.points.shape[1]) or 'all'})")

    deformation = client.deform(template)

    written = export(deformation, cfg, template_norm=_normalized(template, transform),
                     sample_norm=target_norm)
    manifest = write_manifest(cfg, train_cfg, ref, checkpoint, deformation, written)

    print(f"[apply] wrote {len(written)} file(s) to {cfg.output_dir}")
    return deformation, manifest


def run_cohort(cfg: ApplyCfg, train_cfg: ExperimentCfg, ref: CheckpointRef):
    """Apply a cohort run: deform its template once per selected member, with that member's code.

    Each member's result goes to <output_dir>/<member name>, laid out as a pair
    run's output. Only auto-decoder codes (one learned code per member) and
    unconditioned runs are supported; the frame is the cohort's, recomputed from
    the training files by runner.load_cohort.

    Args:
        cfg: RESOLVED ApplyCfg
        train_cfg: the run's ExperimentCfg (train.mode == "cohort")
        ref: the resolved CheckpointRef

    With ``cfg.mean_code`` the template is also deformed with the mean of all member
    codes, and written to <output_dir>/mean_recon (no reference files: it has no sample).

    Returns:
        {member name: (deformation, manifest)}; the mean recon under MEAN_RECON
    """
    kind = train_cfg.code.kind
    if kind not in ("none", "auto_decoder"):
        raise ValueError(f"cohort apply supports code.kind 'none' and 'auto_decoder', not {kind!r}")

    paths, cohort_norm, _, transform = load_cohort(train_cfg)
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    selected = list(cfg.shapes) if cfg.shapes is not None else names
    unknown = sorted(set(selected) - set(names))
    if unknown:
        raise ValueError(f"shapes {unknown} are not members of the cohort in {train_cfg.source}")

    checkpoint = load_checkpoint(ref, map_location=cfg.device)
    codes = None
    if kind == "auto_decoder":
        code_source = build_code_source_cohort(len(paths), kind, train_cfg.code)
        code_source.load_state_dict(checkpoint["codes"])
        codes = code_source.codes.weight.detach()

    template = load_shape(cfg.template)
    template_norm = _normalized(template, transform)
    print(f"[apply] {cfg.template}: {template.points.shape[1]} points, "
          f"{len(selected)} of {len(names)} cohort member(s)")

    results = {}
    for name in selected:
        i = names.index(name)
        member_cfg = dataclass_replace(cfg, output_dir=os.path.join(cfg.output_dir, name))
        client = DeformationClient.from_checkpoint(
            train_cfg, checkpoint, transform,
            code=None if codes is None else codes[i:i + 1],
            pose=(None, None), device=cfg.device, chunk_size=cfg.chunk_size,
        )
        deformation = client.deform(template)
        written = export(deformation, member_cfg, template_norm=template_norm,
                         sample_norm=cohort_norm[i])
        manifest = write_manifest(member_cfg, train_cfg, ref, checkpoint, deformation, written)
        manifest["cohort_member"] = {"name": name, "index": i, "path": paths[i]}
        with open(os.path.join(member_cfg.output_dir, _FILES.manifest), "w") as f:
            json.dump(manifest, f, indent=2)
        results[name] = (deformation, manifest)
        print(f"[apply] {name} (code {i}) -> {member_cfg.output_dir}")

    if cfg.mean_code:
        if codes is None:
            raise ValueError("mean_code needs learned codes: code.kind 'auto_decoder'")
        mean_cfg = dataclass_replace(
            cfg, output_dir=os.path.join(cfg.output_dir, MEAN_RECON),
            export=dataclass_replace(cfg.export, references=False),
        )
        client = DeformationClient.from_checkpoint(
            train_cfg, checkpoint, transform, code=codes.mean(dim=0, keepdim=True),
            pose=(None, None), device=cfg.device, chunk_size=cfg.chunk_size,
        )
        deformation = client.deform(template)
        written = export(deformation, mean_cfg, template_norm=template_norm, sample_norm=None)
        manifest = write_manifest(mean_cfg, train_cfg, ref, checkpoint, deformation, written)
        manifest["mean_of_codes"] = len(codes)
        with open(os.path.join(mean_cfg.output_dir, _FILES.manifest), "w") as f:
            json.dump(manifest, f, indent=2)
        results[MEAN_RECON] = (deformation, manifest)
        print(f"[apply] {MEAN_RECON} (mean of {len(codes)} codes) -> {mean_cfg.output_dir}")
    return results


def resolve_conditioning(train_cfg: ExperimentCfg, checkpoint: dict, sample: Shape):
    """The code and pose the flow has to be evaluated with, recovered from a checkpoint.

    An unconditioned run needs neither. A run with an ENCODER needs both computed,
    and they are computed the way training computed them: by running the encoder on
    the sample -- un-augmented, since augmentation is a training-time perturbation.

    Args:
        train_cfg: the run's ExperimentCfg
        checkpoint: dict from load_checkpoint
        sample: the normalised target Shape, the encoder's input

    Returns:
        (code or None, (rotation, translation))

    Raises:
        ValueError: for a code kind whose value cannot be recovered without
            information the apply config does not carry
    """
    kind = train_cfg.code.kind
    if kind == "none":
        return None, (None, None)

    if kind != "encoder":
        raise ValueError(
            f"cannot apply a run with code.kind={kind!r}: its code is a per-shape "
            f"parameter learned during training, and a template deformed after the "
            f"fact has no shape id to look one up by. Only 'none' and 'encoder' "
            f"(which recomputes the code from the target) can be applied."
        )

    code_source = build_code_source(kind, train_cfg.code)
    code_source.load_state_dict(checkpoint["codes"])
    code_source.eval()

    with torch.no_grad():
        code = code_source(sample)
        # get_pose reads whatever the last forward cached, so it must follow it.
        pose = (code_source.get_pose()
                if getattr(train_cfg, "use_encoder_pose", False) else (None, None))
    return code, pose


def export(deformation: Deformation, cfg: ApplyCfg, template_norm: Shape,
           sample_norm: Optional[Shape]):
    """Write the requested artefacts, once per requested coordinate space.

    Layout, under cfg.output_dir::

        world/deformed.vtp           the answer, denormalised
        world/template_points.vtp    what it started from
        world/sample_points.vtp      what it was fitted to
        world/trajectory/step_*.vtp  the path, if export.trajectory
        normalized/...               the same, in the model's own frame

    The same geometry in both spaces differs only by an affine transform, so a
    discrepancy between the two directories is a transform bug rather than a model
    one -- the same reasoning as TrajectoryExporter, and the same layout, so the
    two are read the same way.

    Args:
        deformation: from DeformationClient.deform
        cfg: RESOLVED ApplyCfg (output_dir is not None)
        template_norm: the deformed shape's input, normalised, for reference export
        sample_norm: the run's target, normalised, for reference export (None without export.references)

    Returns:
        List of written paths
    """
    written = []
    for space in cfg.export.spaces:
        transform = deformation.transform if space == "world" else FrameTransform.identity()
        out_dir = os.path.join(cfg.output_dir, space)
        os.makedirs(out_dir, exist_ok=True)

        if cfg.export.deformed:
            # Faces come from the template: at full resolution they address exactly
            # these vertices, so the output is a MESH rather than the point cloud a
            # subsampled training export is forced to write. export_shape drops them
            # if that ever stops holding.
            #
            # The template's point arrays come with them, for the same reason and
            # under the same guard: inference deforms the WHOLE template in order,
            # so phi(T) has the template's points moved and not resampled, and every
            # per-point array it carried -- heatmaps, landmark distances, mode
            # scores -- still describes the point it is stored against.
            written.append(export_shape(
                deformation.points, deformation.faces, transform,
                os.path.join(out_dir, _FILES.deformed), label="Deformed",
                point_fields=template_norm.point_fields,
            ))

        if cfg.export.references:
            export_reference_shapes(template_norm, sample_norm, transform, out_dir)
            written += [os.path.join(out_dir, name)
                        for name in (_FILES.template_points, _FILES.sample_points)]

        if cfg.export.trajectory:
            written += export_trajectory(deformation.trajectory, deformation.faces,
                                         transform, os.path.join(out_dir, "trajectory"))

    return written


def write_manifest(cfg: ApplyCfg, train_cfg: ExperimentCfg, ref: CheckpointRef,
                   checkpoint: dict, deformation: Deformation, written) -> dict:
    """Record what produced these files, next to the files.

    The counterpart of runner._dump_config: an output directory that cannot say
    which checkpoint it came from is not reproducible, and "which step was this?"
    is exactly the question asked of an inference output months later. The
    directory NAME answers it for a pinned run; this answers it for both.

    Args:
        cfg: RESOLVED ApplyCfg
        train_cfg: the run's ExperimentCfg
        ref: the resolved CheckpointRef
        checkpoint: dict from load_checkpoint
        deformation: the result
        written: paths written by export()

    Returns:
        The manifest dict (also written to apply.json)
    """
    manifest = {
        "train_config": cfg.train_config,
        "checkpoint": checkpoint["path"],
        # From inside the file, which is the step that was actually trained to --
        # the filename only claims it.
        "checkpoint_step": checkpoint_step(checkpoint),
        # False means this directory is "the current end of the run" and will be
        # overwritten by the next apply; True means it names a fixed checkpoint.
        "checkpoint_pinned": ref.pinned,
        "template": cfg.template,
        "target": train_cfg.target,
        "centering": getattr(train_cfg, "centering", "box"),
        "dtype": getattr(train_cfg, "dtype", "float32"),
        "device": cfg.device,
        "field": {"kind": train_cfg.field.kind, "num_steps": train_cfg.field.num_steps},
        "code": {"kind": train_cfg.code.kind},
        "pose_application": getattr(train_cfg, "pose_application", "post"),
        "num_points": deformation.num_points,
        "num_steps": deformation.num_steps,
        # Named to be unmistakable: the training run drew this many points per step,
        # this run used all of them.
        "train_subsample_M": train_cfg.loss.subsample_M,
        "subsampled": False,
        "max_displacement": float(deformation.displacement.norm(dim=-1).max()),
        "files": sorted(written),
    }

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, _FILES.manifest), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def run_from_path(config_path: str):
    """Load an apply config from disk and run it.

    Args:
        config_path: path to the apply .yaml

    Returns:
        (deformation, manifest)
    """
    return run(load_apply_config(config_path))


def _normalized(shape: Shape, transform: FrameTransform) -> Shape:
    """The same shape with its points carried into the run's normalised frame."""
    return Shape(points=transform.apply(shape.points), faces=shape.faces,
                 weights=shape.weights, normals=shape.normals,
                 point_fields=shape.point_fields)
