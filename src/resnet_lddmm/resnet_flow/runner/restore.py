"""Restores a run from a checkpoint."""

import numpy as np
import torch
from src.resnet_lddmm.resnet_flow.callbacks.checkpoints import (
    checkpoint_step,
    load_checkpoint,
    resolve_checkpoint,
)


def restore_from(cfg, stepper):
    """cfg.init -> the named checkpoint loaded into ``stepper``.

    The read half of CheckpointWriter, and the training-side counterpart to what
    apply does with a finished run. Lives in the composition root rather than in
    a callback because the step it returns has to reach ``orchestrator.run``,
    which a callback -- fired from inside that call -- cannot do.

    Args:
        cfg: ExperimentCfg; a None or absent ``init`` means train from scratch
        stepper: the built PairRegistration/CohortRegistration

    Returns:
        The step to number this run from: the checkpoint's when
        ``continue_step_count`` is set, else 0.

    Raises:
        FileNotFoundError: if nothing matches cfg.init.checkpoint
        ValueError: if the file is not a checkpoint of this pipeline, or lacks a
            part the config asked for
        RuntimeError: from load_state_dict, if the config no longer describes the
            checkpoint's architecture
    """
    init = getattr(cfg, "init", None)
    if init is None:
        return 0

    ref = resolve_checkpoint(init.checkpoint, step=init.step)
    state = load_checkpoint(ref)

    # Named parts the file does not carry, reported together. Skipping them
    # silently would leave the run initialised from LESS than the config says --
    # the one failure mode a partial restore has to be loud about, since a flow
    # that quietly stayed random still trains and still converges to something.
    missing = [p for p in init.parts if p not in state]
    if missing:
        raise ValueError(
            f"{ref.path} has no {', '.join(missing)} to restore; it holds "
            f"{sorted(k for k in state if k != 'path')}. Adjust init.parts."
        )

    stepper.load_state_dict({p: state[p] for p in init.parts})

    if "mean" in (init.code_init, init.code_prior):
        if "codes" not in state:
            raise ValueError(f"init.code_init / code_prior 'mean' needs codes in {ref.path}")
        mean = state["codes"]["codes.weight"].mean(dim=0)
        table = stepper.code_source.codes.weight
        if table.shape[1] != mean.shape[0]:
            raise ValueError(f"init 'mean': the checkpoint's codes have {mean.shape[0]} "
                             f"entries, this run's {table.shape[1]}")
        count = state["codes"]["codes.weight"].shape[0]
        if init.code_init == "mean":
            with torch.no_grad():
                table.copy_(mean.to(table).expand_as(table))
            print(f"[INIT] {table.shape[0]} code(s) start at the mean of the checkpoint's {count} codes")
        if init.code_prior == "mean":
            stepper.code_source.set_penalty_center(mean)
            print(f"[INIT] the code penalty pulls towards the mean of the checkpoint's {count} codes")
    if init.code_prior_file is not None:
        _set_code_prior_from_file(stepper.code_source, init.code_prior_file, cfg.loss.code_reg_weight,
                                  cfg.train.batch)

    written_at = checkpoint_step(state)
    start_step = (written_at + 1) if (init.continue_step_count and written_at is not None) else 0
    print(f"[INIT] restored {', '.join(init.parts)} from {ref.path}"
          + (f" (step {written_at}); continuing from step {start_step}"
             if start_step else " (step counter starts at 0)"))
    return start_step


def _set_code_prior_from_file(code_source, path, code_reg_weight, batch):
    """init.code_prior_file: the Gaussian N(m, C) a CodePriorEM run learned on the leading code entries.

    The penalty gets center [m | 0] and precision blockdiag(C^-1 / (2 B lambda), I): the
    per-member prior of callbacks.schedules.code_prior_em for this run's own member count
    B and code_reg_weight lambda, so a code fit with fewer members keeps it unchanged.
    """
    prior = np.load(path)
    m, C, n = prior["m"], prior["C"], int(prior["n_em"])
    members = code_source.num_shapes
    if batch < members:
        raise ValueError(f"init.code_prior_file needs every member in every batch: train.batch {batch} "
                         f"< {members} members")
    if code_reg_weight <= 0:
        raise ValueError("init.code_prior_file needs loss.code_reg_weight > 0: the prior is that penalty")
    table = code_source.codes.weight
    precision = np.eye(code_source.n_z)
    precision[:n, :n] = np.linalg.inv(C) / (2.0 * members * code_reg_weight)
    center = np.zeros(code_source.n_z)
    center[:n] = m
    code_source.set_penalty_precision(torch.as_tensor(center, dtype=table.dtype),
                                      torch.as_tensor(precision, dtype=table.dtype))
    print(f"[INIT] code prior N(m, C) on the first {n} code entries from {path} "
          f"(trace C {np.trace(C):.4g}); the other {code_source.n_z - n} keep the plain L2")
