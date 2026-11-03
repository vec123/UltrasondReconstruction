"""The experiment config: centering, dtype, augmentation, init, training and the top-level ``ExperimentCfg``."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Optional
from src.shared.core.config import reject_unknown_keys

from .field import FieldCfg
from .loss import LossCfg
from .pose_code import CodeCfg, _parse_code_config


# Where joint_normalize places the shapes -- equivalently, where the ORIGIN sits
# relative to them. Not cosmetic: SO3Augmentation and the encoder pose are both
# applied as ``points @ R``, i.e. about the ORIGIN, so this decides whether a
# rotation is a spin or an orbit.
#
#   "box"    [0,1]^3, centroid at (0.5,0.5,0.5), 0.87 from the origin. A rotation
#            ORBITS the shape further than its own diameter, which gives chamfer a
#            strong pose signal (measured: 64x contrast vs 9x). But a rotation-only
#            pose can then only reach targets whose centroid stays on that sphere,
#            so samples that arrive already rotated in place are NOT reachable.
#   "origin" [-0.5,0.5]^3, centroid at the origin. A rotation is a spin in place, so
#            pre-rotated samples are reachable and the pose head's translation (an
#            absolute centroid) composes correctly. Weaker chamfer signal.
CENTERING_DOMAINS = {"box": (0.0, 1.0), "origin": (-0.5, 0.5)}


# Floating-point precision for the whole run. Named here, resolved to a torch dtype in
# the runner, so this module stays importable without torch.
#
# Nothing in the pipeline hardcodes precision -- every tensor derives its dtype from
# the data (``dtype=q0.dtype``, ``dtype=points.dtype``, ...) -- so this one key
# propagates everywhere by itself.
#
#   "float32"  the default, and what every result before this key was produced with.
#   "float64"  ~2x slower on CPU and considerably worse on GPU. Worth it as a
#              DIAGNOSTIC: the pose vectors are the residual of a first moment that
#              nearly cancels (measured ~1% of the per-node norms), and cancellation
#              is exactly where float32 loses relative precision -- 7 significant
#              digits in, ~5 left in the quantity actually used.
DTYPES = ("float32", "float64")


@dataclass
class AugmentationCfg:
    """Configuration for source shape augmentation (random group transformations)."""
    kind: str = "none"                      # none | so3 | se3
    seed: Optional[int] = None              # Random seed for reproducibility
    translation_scale: float = 0.2          # Only for SE3: bounds on uniform translation


@dataclass
class InitCfg:
    """Start this run from a checkpoint instead of from fresh weights.

    Two orthogonal knobs rather than a resume/warm-start mode, because the two
    intents differ only in how they set them:

      * RESUME -- the same config, continued. Restore everything and keep
        counting: ``init: {checkpoint: outputs/prev/checkpoints}``, the defaults.
      * WARM START -- a DIFFERENT config (other loss weights, other target)
        initialised from a learned deformation. ``parts: [flow]`` leaves the
        optimizer fresh, and ``continue_step_count: false`` starts the new run's
        exports at step 0.

    Restoring is strict: ``load_state_dict`` fails naming the mismatched
    parameters when the config no longer describes the checkpoint, so a changed
    ``field.mlp`` is an error rather than a partially-loaded model.

    Attributes:
        checkpoint: a ``.pt`` file, or a run's ``checkpoints/`` directory -- see
            checkpoint.resolve_checkpoint for how each is resolved.
        step: pin an exact step within a checkpoints DIRECTORY. Meaningless
            alongside a ``.pt`` path, which already names one.
        parts: which pieces of the checkpoint to restore, out of PARTS.
        continue_step_count: carry on from the checkpoint's step rather than
            restarting at 0. On by default so a resumed run's checkpoints and
            exports do not overwrite the ones it resumed from. ``train.steps``
            means "run this many MORE steps" either way.
        code_init: "mean" starts every code of this run at the mean of the
            checkpoint's codes (the centre of the learned code space), for fitting
            new targets to a trained flow; None leaves the fresh initialisation.
        code_prior: "mean" centres the code penalty on the checkpoint's code mean
            instead of zero, so a fit is pulled towards the learned code space
            rather than towards the zero code (a different face on a direct grid).
        code_prior_file: an .npz a CodePriorEM run wrote (code_prior_em/step_*.npz):
            its Gaussian N(m, C) on the leading code entries becomes the code
            penalty, the rest keep the plain L2 (runner._set_code_prior_from_file).
    """

    # The keys CheckpointWriter puts in the file, which are exactly the keys the
    # steppers' state_dict/load_state_dict speak. Validated here so a typo'd part
    # fails at config load rather than silently restoring nothing.
    PARTS = ("flow", "codes", "optimizer")

    checkpoint: str
    step: Optional[int] = None
    parts: tuple = PARTS
    continue_step_count: bool = True
    code_init: Optional[str] = None
    code_prior: Optional[str] = None
    code_prior_file: Optional[str] = None

    CODE_INITS = ("mean",)

    @classmethod
    def from_dict(cls, d: dict) -> "InitCfg":
        reject_unknown_keys(d, {"checkpoint", "step", "parts", "continue_step_count", "code_init",
                                "code_prior", "code_prior_file"}, what="init")
        if d.get("code_prior") is not None and d.get("code_prior_file") is not None:
            raise ValueError("init.code_prior and init.code_prior_file both set the code penalty; give one")
        code_init, code_prior = d.get("code_init"), d.get("code_prior")
        for key, value in (("code_init", code_init), ("code_prior", code_prior)):
            if value is not None and value not in cls.CODE_INITS:
                raise ValueError(f"unknown init.{key} {value!r}; expected one of {list(cls.CODE_INITS)}")
        if "checkpoint" not in d:
            raise ValueError(
                "init needs a 'checkpoint': a .pt file, or a run's checkpoints/ "
                "directory. Drop the whole init block to train from scratch."
            )

        parts = d.get("parts", cls.PARTS)
        if isinstance(parts, str):          # `parts: flow` -> ("flow",), not 4 chars
            parts = [parts]
        parts = tuple(parts)
        unknown = [p for p in parts if p not in cls.PARTS]
        if unknown:
            raise ValueError(
                f"unknown init.parts {unknown}; expected a subset of {list(cls.PARTS)}"
            )
        if not parts:
            raise ValueError(
                "init.parts is empty, so the checkpoint would be read and nothing "
                "restored. Drop the init block instead."
            )

        return cls(checkpoint=d["checkpoint"],
                   step=d.get("step"),
                   parts=parts,
                   continue_step_count=bool(d.get("continue_step_count", True)),
                   code_init=code_init,
                   code_prior=code_prior,
                   code_prior_file=d.get("code_prior_file"))


@dataclass
class TrainCfg:
    mode: str = "pair"                      # pair | cohort
    steps: int = 2000
    batch: int = 8
    lr: float = 1e-4
    seed: Optional[int] = None
    log_every: int = 50
    save_every: int = 500
    val_every: int = 0
    export_shapes: int = 0                  # 0 = all shapes (pair: always 1, cohort: min(batch_size, cohort_size))
    export_strategy: str = "sequential"     # sequential | random
    device: str = "cpu"                     # cpu | cuda | cuda:<n> - where the flow trains
    # Train the codes only: the flow (restored through `init`) gets no gradient and
    # is left out of the optimizer. Cohort mode; this is how a trained flow is fitted
    # to new targets with its own training loss.
    freeze_flow: bool = False
    # With freeze_flow: flow parameters whose name holds one of these strings stay trainable,
    # shared over the cohort (e.g. [norm]: the LayerNorm affines of field.norm: layer).
    # Pair them with loss terms parameter_anchor / field_change_kinetic (losses/fine_tune.py).
    freeze_flow_except: tuple = ()


@dataclass
class ExperimentCfg:
    source: str                             # pair: two paths; cohort: folder + template
    target: str
    output_dir: str
    # Folder of per-annotator landmark CSVs (Slicer fiducial export: x,y,z,label,
    # ...) in the same world frame as source/target. None = no landmark guidance;
    # only read when loss.landmark_weight > 0. See landmarks.py.
    landmarks: Optional[str] = None
    field: FieldCfg = dataclass_field(default_factory=FieldCfg)
    code: CodeCfg = dataclass_field(default_factory=CodeCfg)
    loss: LossCfg = dataclass_field(default_factory=LossCfg)
    augmentation: AugmentationCfg = dataclass_field(default_factory=AugmentationCfg)
    train: TrainCfg = dataclass_field(default_factory=TrainCfg)
    # Where the run's weights come from. None -- the default -- is a fresh model.
    init: Optional[InitCfg] = dataclass_field(default=None)
    centering: str = "box"                  # box | origin — see CENTERING_DOMAINS
    # Cohort mode: a folder of shapes that, with the template, defines the normalised
    # frame instead of `source`. Set it to a trained run's source when fitting new
    # targets to its flow, so they enter the frame the flow was trained in.
    frame_source: Optional[str] = None
    dtype: str = "float32"                  # float32 | float64 — see DTYPES
    # Apply the code source's learned pose to the flow output (unidirectional only).
    # Named `use_pose` in configs; `use_encoder_pose` is the original spelling and
    # stays valid, because the pose does not have to come from an encoder -- the
    # so3/se3 code sources carry one as free parameters. Both names set this field.
    use_encoder_pose: bool = False
    freeze_flow_at_identity: bool = False   # If True, skip flow computation; encoder learns pose only
    # Where the pose enters relative to the flow. "post" (pred = phi(T) @ R) leaves the
    # trajectory independent of R, so kinetic and isometry are EXACTLY constant in the
    # pose and only the data term can train it. "pre" (pred = phi(T @ R)) puts the
    # trajectory downstream of R, which is what makes "the correct pose needs the least
    # non-rigid flow" trainable. See UnidirectionalMappingError.
    pose_application: str = "post"          # post | pre

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentCfg":
        """Create ExperimentCfg from dict, erroring on unknown keys.

        Handles nested config for encoder_config and graph_spec.
        """
        reject_unknown_keys(d, {
            "source", "target", "output_dir", "landmarks", "field", "code", "loss",
            "augmentation", "train", "init", "centering", "dtype", "use_pose",
            "use_encoder_pose", "freeze_flow_at_identity", "pose_application",
            "frame_source",
        })

        # Two spellings of one flag. Conflicting values are a mistake worth naming
        # rather than resolving by precedence -- there is no reading of the config
        # under which the author meant both.
        use_pose = d.get("use_pose")
        use_encoder_pose = d.get("use_encoder_pose")
        if use_pose is not None and use_encoder_pose is not None \
                and bool(use_pose) != bool(use_encoder_pose):
            raise ValueError(
                f"use_pose={use_pose} contradicts use_encoder_pose={use_encoder_pose}; "
                f"they are the same flag. Keep one (use_pose is the current name)."
            )
        use_pose = bool(use_pose if use_pose is not None else (use_encoder_pose or False))

        field_cfg = FieldCfg.from_dict(d.get("field", {})) if "field" in d else FieldCfg()
        loss_cfg = LossCfg(**d.get("loss", {})) if "loss" in d else LossCfg()
        augmentation_cfg = AugmentationCfg(**d.get("augmentation", {})) if "augmentation" in d else AugmentationCfg()
        train_cfg = TrainCfg(**d.get("train", {})) if "train" in d else TrainCfg()
        init_cfg = InitCfg.from_dict(d["init"]) if d.get("init") else None

        centering = d.get("centering", "box")
        if centering not in CENTERING_DOMAINS:
            raise ValueError(
                f"unknown centering {centering!r}; expected one of "
                f"{sorted(CENTERING_DOMAINS)}"
            )

        dtype = d.get("dtype", "float32")
        if dtype not in DTYPES:
            raise ValueError(
                f"unknown dtype {dtype!r}; expected one of {list(DTYPES)}"
            )

        pose_application = d.get("pose_application", "post")
        if pose_application not in ("post", "pre"):
            raise ValueError(
                f"unknown pose_application {pose_application!r}; expected 'post' or 'pre'"
            )

        # Handle code config with special parsing for encoder_config and graph_spec
        code_dict = d.get("code", {})
        code_cfg = _parse_code_config(code_dict)

        return cls(
            source=d["source"],
            target=d["target"],
            output_dir=d["output_dir"],
            landmarks=d.get("landmarks"),
            field=field_cfg,
            code=code_cfg,
            loss=loss_cfg,
            augmentation=augmentation_cfg,
            train=train_cfg,
            init=init_cfg,
            centering=centering,
            frame_source=d.get("frame_source"),
            dtype=dtype,
            use_encoder_pose=use_pose,
            freeze_flow_at_identity=d.get("freeze_flow_at_identity", False),
            pose_application=pose_application,
        )
