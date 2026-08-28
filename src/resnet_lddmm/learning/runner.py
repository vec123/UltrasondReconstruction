"""Composition Root.

The ONE place the whole object graph is assembled: config -> components ->
data  -> loss composition  -> callbacks  -> orchestrator -> run.

does NOT import: the config package. It reads attributes off
whatever config object it is handed, so the layering rule still holds and the
runner stays testable with a stub config. The CLI is what turns YAML into that
object (``config/resolve.py``).
"""

import os
import random

import numpy as np
import torch

from src.paths import get_project_root
from src.learning.factories import build_encoder, build_decoder
from src.learning.registry import Registry
from src.learning.data.loading import load_dataset, split_dataset
from src.learning.loader.loaders import OneBatchLoader, ResamplingGraphLoader
from src.learning.losses.composer import LossComposer, LossTerm
from src.learning.trainers.E3_end2end import TrainingStepper, TrainingOrchestrator
from src.learning.callbacks.metrics import MetricsRecorder, MetricsPlotter
from src.learning.callbacks.checkpointing import CheckpointWriter
from src.learning.callbacks.visualization import GeometryVisualizer
from src.learning.callbacks.validation import ValidationRunner


def resolve_path(path):
    """Make a config-relative path absolute against the repo root, so the same
    config works from any working directory and on any machine."""
    if os.path.isabs(path):
        return path
    return os.path.join(get_project_root(), path)


def seed_everything(seed):
    """Mirrors the seeding the training scripts did, and returns the loader rng.

    Returns ``None`` for the generator when ``seed`` is None, which leaves the
    loaders on the global RNG (unseeded, non-reproducible) -- so a config without
    a seed is visibly not reproducible rather than quietly pseudo-reproducible.
    """
    if seed is None:
        return None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)            # seeds CPU + the default CUDA generator
    torch.cuda.manual_seed_all(seed)   # all GPUs (redundant with above but explicit)
    key = torch.Generator(device="cpu")
    key.manual_seed(seed)
    return key


def build_composer(cfg):
    """LossConfig -> LossComposer. The config layer's LossTermConfig is converted
    into the losses package's own LossTerm here, so the composer never imports config."""
    return LossComposer([
        LossTerm(term.name, term.weight, term.kwargs)
        for term in cfg.training.losses.terms
    ])


def wants_two_view(cfg):
    """Whether the loader must draw two views per step.

    DERIVED from the loss config rather than configured separately: a contrastive
    term is exactly the thing that needs two views, so there is one source of
    truth and no way to configure the pair inconsistently.
    """
    return any(term.name == "contrastive" for term in cfg.training.losses.terms)


def build_loaders(cfg, rng):
    """Config -> (train_loader, val_loader), built from the same GraphBuilder."""
    data_path = resolve_path(cfg.data.data_path)
    loaded = load_dataset(
        data_path=data_path, parts=cfg.data.parts,
        load_fields=cfg.data.load_fields, shuffle=cfg.data.shuffle,
        verbose=False, seed=cfg.data.seed,
    )
    if cfg.data.load_fields:
        vertices, mask, areas, normals = loaded
        splits = split_dataset(vertices, mask, areas, normals,
                               val_fraction=cfg.data.val_fraction, seed=cfg.data.seed)
        (t_verts, t_mask, t_areas, t_normals), (v_verts, v_mask, v_areas, v_normals) = splits
    else:
        vertices, mask = loaded
        (t_verts, t_mask), (v_verts, v_mask) = split_dataset(
            vertices, mask, val_fraction=cfg.data.val_fraction, seed=cfg.data.seed)
        t_areas = t_normals = v_areas = v_normals = None

    builder = Registry.create("graph_builder", cfg.data.graph_builder,
                              spec=cfg.data.graph_spec)
    two_view = wants_two_view(cfg)

    if cfg.data.resample_graph or two_view:
        train_loader = ResamplingGraphLoader(
            t_verts, t_mask, builder, rng=rng, two_view=two_view,
            batch_size=cfg.training.batch_size, areas=t_areas, normals=t_normals)
    else:
        graph, supergraph = builder.build(t_verts, t_mask, rng,
                                          areas=t_areas, normals=t_normals)
        train_loader = OneBatchLoader((graph, supergraph, t_verts, t_mask))

    # Validation graph is built from the VALIDATION geometry, so the graph and its
    # reconstruction target describe the same shapes.
    val_graph, val_supergraph = builder.build(v_verts, v_mask, rng,
                                              areas=v_areas, normals=v_normals)
    val_loader = OneBatchLoader((val_graph, val_supergraph, v_verts, v_mask))
    return train_loader, val_loader


def build_callbacks(cfg, val_loader, extra=()):
    """Config cadences -> the callback list. Each owns its own schedule."""
    recorder = MetricsRecorder(every_n_steps=cfg.training.log_every)
    callbacks = [
        recorder,
        MetricsPlotter(recorder, every_n_steps=0),   # 0 = only on validation / train end
        CheckpointWriter(every_n_steps=cfg.training.save_every),
        GeometryVisualizer(every_n_steps=cfg.training.save_every, val_viz_random=cfg.training.val_viz_random),
    ]
    if val_loader is not None and cfg.training.val_every:
        callbacks.append(ValidationRunner(val_loader, every_n_steps=cfg.training.val_every))
    callbacks.extend(extra)
    return callbacks


def run_experiment(cfg, output_dir=None, extra_callbacks=()):
    """Assemble everything from ``cfg`` and run. Returns the final TrainingContext."""
    rng = seed_everything(cfg.seed)
    output_dir = output_dir or resolve_path(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    encoder = build_encoder(cfg.encoder)
    decoder = build_decoder(cfg.decoder)
    train_loader, val_loader = build_loaders(cfg, rng)

    stepper = TrainingStepper(
        encoder, decoder,
        learning_rate=cfg.training.learning_rate,
        composer=build_composer(cfg),
        device=cfg.training.device,
        verbose=cfg.training.verbose,
    )
    orchestrator = TrainingOrchestrator(
        stepper=stepper, dataloader=train_loader,
        callbacks=build_callbacks(cfg, val_loader, extra_callbacks),
        log_dir=output_dir,
    )
    return orchestrator.run(num_steps=cfg.training.num_steps)