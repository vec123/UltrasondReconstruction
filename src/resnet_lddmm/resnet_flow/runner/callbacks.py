"""The default callbacks and the ones a given config adds."""

import os
import torch
from src.resnet_lddmm.resnet_flow.callbacks.checkpoints import CheckpointWriter
from src.resnet_lddmm.resnet_flow.config import FOURIER_FIELD_KINDS
from src.resnet_lddmm.resnet_flow.fields.composite.modes import ModalField, network_of
from src.resnet_lddmm.resnet_flow.fields.composite.prior import PriorResidualField
from src.resnet_lddmm.resnet_flow.callbacks.deformation import DiagnosticsCallback, TrajectoryExporter
from src.resnet_lddmm.resnet_flow.callbacks.deformation.modes import (
    DecompositionExporter,
    ModeAmplitudeLogger,
)
from src.resnet_lddmm.resnet_flow.callbacks.encoder import EncoderGraphLogger, PoseLogger, PoseShapeExporter
from src.resnet_lddmm.resnet_flow.callbacks.gradient_logs import (
    GradientLogger,
    NetworkStructureInspector,
    RequiresGradMonitor,
)
from src.resnet_lddmm.resnet_flow.callbacks.loss import LossLogger, VerboseCallback
from src.resnet_lddmm.resnet_flow.callbacks.schedules import CodePriorEM, CodeRegWarmup, FourierAnnealing

from .data import load_pair


def default_callbacks(cfg) -> list:
    """The callbacks every run gets unless the caller passes its own list.

    Logging, gradient stats, trajectory export, diagnostics and checkpoints;
    ``complete_callbacks`` adds what depends on the built model and the config.
    """
    log_every = getattr(cfg.train, 'log_every', 50)
    save_every = getattr(cfg.train, 'save_every', 100)
    export_shapes = getattr(cfg.train, 'export_shapes', 0)
    export_strategy = getattr(cfg.train, 'export_strategy', 'sequential')
    freeze_flow_at_identity = getattr(cfg, 'freeze_flow_at_identity', False)
    # Seed rng for export strategy if seed is set
    export_rng = None
    if cfg.train.seed is not None:
        export_rng = torch.Generator(device="cpu")
        export_rng.manual_seed(cfg.train.seed)
    return [
        NetworkStructureInspector(),
        VerboseCallback(log_every=log_every, pose_only_mode=freeze_flow_at_identity),
        GradientLogger(every_n_steps=log_every),
        RequiresGradMonitor(every_n_steps=log_every),
        TrajectoryExporter(every_n_steps=save_every, export_shapes=export_shapes,
                         export_strategy=export_strategy, rng=export_rng),
        DiagnosticsCallback(every_n_steps=save_every),
        # The weights themselves, to output_dir/checkpoints. Everything else in
        # this list writes something DERIVED from the model -- geometry, metrics,
        # gradient stats -- none of which can reconstruct it, so without this a
        # finished run leaves nothing to evaluate or resume from. Saves
        # {step, flow, codes, optimizer} (the encoder rides along inside `codes`,
        # which is an nn.Module holding it), plus a final.pt on train end so the
        # last weights survive a run that doesn't stop on the cadence.
        CheckpointWriter(every_n_steps=save_every),
    ]


def _code_prior_em(cfg, stepper):
    """loss.code_prior_em's callback: EM on the code entries the frozen field.prior reads.

    Member names come from cfg.source in load_cohort's order, which is the code order.
    """
    priors = [m for m in stepper.flow.field.modules() if isinstance(m, PriorResidualField)]
    if cfg.train.mode != "cohort" or not priors or priors[0].prior_n_z == 0:
        raise ValueError("loss.code_prior_em needs train.mode: cohort and a field.prior that reads a code")
    names = [os.path.splitext(f)[0] for f in sorted(f for f in os.listdir(cfg.source) if f.lower().endswith(".vtp"))]
    return CodePriorEM(stepper, names, priors[0].prior_n_z, cfg.loss.code_reg_weight, cfg.output_dir,
                       **cfg.loss.code_prior_em)


def complete_callbacks(cfg, callbacks, stepper, transform) -> list:
    """``callbacks`` plus everything this run's model and config call for, in running order.

    The encoder / pose loggers, the Fourier schedules and the designed-mode
    amplitudes are added as the config enables them; LossLogger comes after
    every callback that writes metrics, and every exporter that wants the frame
    transform gets it. The mode decomposition (field.modes.decompose), which
    acts only at train end, is last. Returns a new list; ``callbacks`` is not
    modified.

    Raises:
        ValueError: field.modes.decompose with a per-shape code (code.kind != none).
    """
    callbacks = list(callbacks)
    save_every = getattr(cfg.train, 'save_every', 100)
    export_shapes = getattr(cfg.train, 'export_shapes', 0)

    # Add EncoderGraphLogger if using encoder
    if cfg.code.kind == "encoder":
        graph_log_cadence = getattr(cfg.train, 'save_every', 100)
        callbacks.append(EncoderGraphLogger(every_n_steps=graph_log_cadence))

    # Add PoseLogger if using encoder pose
    if getattr(cfg, 'use_encoder_pose', False):
        pose_log_cadence = getattr(cfg.train, 'log_every', 50)
        augmentation_kind = cfg.augmentation.kind if hasattr(cfg, 'augmentation') else 'se3'
        callbacks.append(PoseLogger(every_n_steps=pose_log_cadence, augmentation_kind=augmentation_kind))
        # The pose as GEOMETRY rather than as a matrix: phi(T) before and after the
        # predicted pose, plus the sample it should land on. Shares save_every with the
        # trajectory export, since both write .vtp and are read the same way.
        callbacks.append(PoseShapeExporter(every_n_steps=save_every,
                                           export_shapes=export_shapes or 2))

    # Coarse-to-fine Fourier bands. After CheckpointWriter, so a checkpoint keeps the
    # alpha its weights were trained with; before LossLogger, so fourier/alpha is logged.
    # The code penalty's warm-up: before Fourier/loggers, like the other schedules.
    if getattr(cfg.loss, "code_reg_warmup", None):
        callbacks.append(CodeRegWarmup(stepper.composer, cfg.loss.code_reg_weight, **cfg.loss.code_reg_warmup))
    if cfg.field.kind in FOURIER_FIELD_KINDS and cfg.field.fourier_anneal is not None:
        callbacks.append(FourierAnnealing(network_of(stepper.flow.field).encoding, **cfg.field.fourier_anneal))
    if cfg.field.kind == "time_varying_local":
        field = network_of(stepper.flow.field)
        if cfg.field.fourier_anneal is not None:
            callbacks.append(FourierAnnealing(field.coarse.encoding, **cfg.field.fourier_anneal))
        if cfg.field.local["fourier_anneal"] is not None:
            callbacks.append(FourierAnnealing(field.fine.encoding, **cfg.field.local["fourier_anneal"],
                                              metric="fourier/alpha_local"))
    # Before LossLogger, so modes/<name>_mm lands in the same record.
    if isinstance(stepper.flow.field, ModalField):
        callbacks.append(ModeAmplitudeLogger(stepper.flow.field))

    # The loss breakdown as JSON. LAST in the list on purpose: metrics is one dict
    # shared by every callback, so running last is what lets DiagnosticsCallback's
    # diag/* and the gradient stats land in the same record instead of a step late.
    # (re-read the cadence rather than reusing log_every: that name only exists on
    # the branch that builds the default list, not when callbacks were passed in)
    callbacks.append(LossLogger(every_n_steps=getattr(cfg.train, 'log_every', 50)))

    # The trained flow split into network and modes, as .vtp files. At train end
    # only, and AFTER CheckpointWriter and LossLogger: if the export fails, the
    # weights and the logs are already written.
    decompose = cfg.field.modes["decompose"] if cfg.field.modes is not None else "none"
    if decompose != "none":
        if cfg.code.kind != "none":
            raise ValueError(f"field.modes.decompose with code.kind={cfg.code.kind!r}: the decomposition "
                             f"runs the network without a code; set decompose: none")
        source, target, _ = load_pair(cfg)       # with landmarks (the loader's copies have none)
        callbacks.append(DecompositionExporter(source, target, transform,
                                               trajectories=decompose == "trajectories"))

    # The code prior's EM, last: its Laplace passes overwrite the stepper's last
    # trajectory and sample, which the callbacks before it read.
    if getattr(cfg.loss, "code_prior_em", None):
        callbacks.append(_code_prior_em(cfg, stepper))

    # Hand the frame transform to every exporter that wants one. This runs AFTER all
    # appends above: PoseShapeExporter is added conditionally further up, and assigning
    # before that left it with transform=None, silently exporting the normalised frame
    # under world/ as well.
    for cb in callbacks:
        if isinstance(cb, (TrajectoryExporter, PoseShapeExporter)):
            cb.transform = transform
    return callbacks
