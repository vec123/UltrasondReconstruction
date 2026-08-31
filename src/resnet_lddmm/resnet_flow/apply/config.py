"""Configuration for applying a trained flow to a shape.

An apply config REFERENCES a training config rather than restating it. Field
kind, width, num_steps, centering and dtype all have to match the checkpoint
exactly or the weights will not load -- and a second copy of those values is a
copy that can drift. So the only things named here are the ones the training run
could not know: which weights to load, what to deform, and where to write.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field as dataclass_field, replace as dataclass_replace
from typing import TYPE_CHECKING, Optional, Tuple

from src.core.config import load_yaml, reject_unknown_keys
from src.resnet_lddmm.resnet_flow.config import ExperimentCfg

if TYPE_CHECKING:
    # Only used as a type annotation below (postponed by `from __future__ import
    # annotations`), so this stays out of ApplyCfg's own import - which torch-free
    # callers (e.g. the pipeline registry, which imports config_cls eagerly but
    # defers the run itself) rely on staying cheap. checkpoint.py pulls in torch.
    from src.resnet_lddmm.resnet_flow.checkpoint import CheckpointRef

# The coordinate frames an export can be written in.
#
#   "world"       denormalised -- directly comparable with the input meshes, and
#                 what you want if the result is going anywhere else.
#   "normalized"  the frame the model actually operates in, i.e. the domain the
#                 field was fitted on. Read this one when judging the model.
SPACES = ("world", "normalized")


@dataclass(frozen=True)
class ExportCfg:
    """What the apply run writes out.

    Attributes:
        deformed: the terminal shape phi(T). The point of the run; on by default.
        trajectory: every integration step, as one .vtp per step. Off by default
            because it multiplies the written volume by num_steps + 1 and is only
            useful for inspecting the path rather than the result.
        references: the template and the target alongside the output, so the
            three can be loaded together without hunting for the inputs.
        spaces: which frames to write. See SPACES.
    """

    deformed: bool = True
    trajectory: bool = False
    references: bool = True
    spaces: Tuple[str, ...] = SPACES

    @classmethod
    def from_dict(cls, d: dict) -> "ExportCfg":
        reject_unknown_keys(d, {"deformed", "trajectory", "references", "spaces"},
                            what="apply.export")
        spaces = tuple(d.get("spaces", SPACES))
        unknown = set(spaces) - set(SPACES)
        if unknown:
            raise ValueError(
                f"unknown export space(s) {sorted(unknown)}; expected any of {list(SPACES)}"
            )
        if not spaces:
            raise ValueError("export.spaces is empty; nothing would be written")
        return cls(
            deformed=bool(d.get("deformed", True)),
            trajectory=bool(d.get("trajectory", False)),
            references=bool(d.get("references", True)),
            spaces=spaces,
        )


@dataclass(frozen=True)
class ApplyCfg:
    """One inference run: a checkpoint, a shape to deform, a place to put it.

    Attributes:
        train_config: path to the YAML the checkpoint was trained from. Sole
            source of the architecture, the centering and the dtype.
        checkpoint: a .pt file, or a directory of them -- see
            checkpoint.resolve_checkpoint for how each is resolved.
        step: the exact training step to load out of that directory. None leaves
            it to be resolved (final.pt, else the highest step), which is
            "whatever this run ended at" rather than a fixed checkpoint.
        output_dir: where to write. None means <training output_dir>/apply, which
            keeps a run's inference next to the run that produced it. A PINNED
            checkpoint adds a subdirectory naming it either way -- see resolved().
        template: the shape to deform. None means the training config's source,
            which is the usual case: the SAME template the run was fitted to,
            now at full resolution because nothing subsamples at inference.
        device: torch device string for the flow and the points.
        chunk_size: split the cloud into chunks of this many points and integrate
            them separately, to bound peak memory on a large template. 0 disables.
            Exact for pointwise fields only; the client refuses it otherwise.
        export: what to write, see ExportCfg.
    """

    train_config: str
    checkpoint: str
    step: Optional[int] = None
    output_dir: Optional[str] = None
    template: Optional[str] = None
    device: str = "cpu"
    chunk_size: int = 0
    export: ExportCfg = dataclass_field(default_factory=ExportCfg)

    @classmethod
    def from_dict(cls, d: dict) -> "ApplyCfg":
        """Create an ApplyCfg from a parsed YAML mapping, erroring on unknown keys."""
        reject_unknown_keys(d, {
            "train_config", "checkpoint", "step", "output_dir", "template", "device",
            "chunk_size", "export",
        }, what="apply config")

        for required in ("train_config", "checkpoint"):
            if not d.get(required):
                raise ValueError(f"apply config must name a {required!r}")

        chunk_size = int(d.get("chunk_size", 0))
        if chunk_size < 0:
            raise ValueError(f"chunk_size must be >= 0, got {chunk_size}")

        # `step: null` is the documented way to say "resolve one", so an absent key
        # and an explicit null have to mean the same thing -- hence .get() rather
        # than a truthiness test, which would also swallow the legitimate step 0.
        step = d.get("step")
        if step is not None:
            step = int(step)
            if step < 0:
                raise ValueError(f"step must be >= 0, got {step}")

        return cls(
            train_config=d["train_config"],
            checkpoint=d["checkpoint"],
            step=step,
            output_dir=d.get("output_dir"),
            template=d.get("template"),
            device=d.get("device", "cpu"),
            chunk_size=chunk_size,
            export=ExportCfg.from_dict(d.get("export", {})),
        )

    def resolved(self, train_cfg: ExperimentCfg, checkpoint: CheckpointRef) -> "ApplyCfg":
        """Fill the fields that default to something outside this config.

        The output directory gains a ``step_1300``/``final`` subdirectory whenever
        the checkpoint was PINNED -- named by step, or by a direct .pt path. The
        point is that pinning is how several checkpoints get compared, and a flat
        directory would have each run silently overwrite the last; the folder then
        says which one produced it without opening apply.json. An unpinned run
        means "the current end of this run", of which there is only ever one, so
        it stays flat and keeps overwriting itself as training progresses.

        Returned rather than mutated (the dataclass is frozen), so the object the
        runner works with has no None left to re-derive and the manifest records
        what actually ran.

        Args:
            train_cfg: the ExperimentCfg loaded from self.train_config
            checkpoint: the resolved CheckpointRef

        Returns:
            A copy with step, output_dir and template resolved to concrete values
        """
        output_dir = self.output_dir or os.path.join(train_cfg.output_dir, "apply")
        if checkpoint.pinned:
            output_dir = os.path.join(output_dir, checkpoint.label)

        return dataclass_replace(
            self,
            step=self.step if self.step is not None else checkpoint.step,
            output_dir=output_dir,
            template=self.template or train_cfg.source,
        )


def load_apply_config(path) -> ApplyCfg:
    """Load an apply config from YAML.

    Args:
        path: path to the .yaml file

    Returns:
        ApplyCfg (unresolved -- call .resolved(train_cfg) once the training
        config is loaded)

    Raises:
        FileNotFoundError, ValueError, yaml.YAMLError: as load_yaml
    """
    return ApplyCfg.from_dict(load_yaml(path))


def load_training_config(apply_cfg: ApplyCfg) -> ExperimentCfg:
    """Load the ExperimentCfg the checkpoint was trained from.

    Args:
        apply_cfg: the apply config naming it

    Returns:
        ExperimentCfg
    """
    data = load_yaml(apply_cfg.train_config)
    # runner._dump_config stamps a top-level "mode" ("pair"/"cohort") onto every
    # config.json for human provenance; it duplicates train.mode and was never an
    # ExperimentCfg field, so ExperimentCfg.from_dict's unknown-key check rejects
    # it. Strip it here rather than loosen from_dict, since only this file's own
    # dumps carry it - a hand-written training config never should.
    data.pop("mode", None)
    return ExperimentCfg.from_dict(data)
