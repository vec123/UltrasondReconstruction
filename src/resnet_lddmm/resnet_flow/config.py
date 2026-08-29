from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace as dataclass_replace
from typing import Optional, List, Any, Dict

from src.core.config import reject_unknown_keys

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
class FieldCfg:
    kind: str = "time_varying"              # time_varying | stationary | equivariant_stationary
    num_steps: int = 10                     # K — configured in ONE place
    width: int = 512                        # block width (ignored by stationary/equivariant_stationary)
    activation: str = "relu"
    fa: tuple = (64, 64, 64)                # stationary: FA-NN layer widths
    df: tuple = (256, 256, 256, 256, 256)   # DF-NN layer widths
    fourier_n_e: int = 3
    # Equivariant stationary field (only if kind == "equivariant_stationary")
    hidden_irreps: str = "4x0e + 2x1o"      # intermediate irreps representation (single-layer fallback)
    gate_hidden_dim: int = 64               # hidden dim for gating network in SelfInteraction
    use_tensor_product_self: bool = True    # use SelfInteraction (V⊗V+gating) vs simple Linear
    layers: Optional[List[dict]] = dataclass_field(default=None)  # layer stack config for depth
    # Equivariant contextual field (only if kind == "equivariant_contextual")
    context_irreps: str = "32x0e + 16x1o"   # features after message passing aggregation
    sh_lmax: int = 2                        # spherical harmonics lmax for bipartite convolution
    # Simple equivariant contextual field (only if kind == "equivariant_contextual_simple")
    hidden_dim: int = 128                   # hidden dimension in MLP layers
    n_layers: int = 3                       # number of MLP layers


@dataclass
class PoseCfg:
    """A rigid pose learned alongside (or instead of) the flow — see poses/.

    Orthogonal to CodeCfg.kind: any code source can carry any pose. Rigid-only
    registration is ``code: {kind: none, pose: {kind: se3}}`` with
    ``freeze_flow_at_identity: true``; pose plus deformation is the same pose block
    on ``kind: auto_decoder``.

    Bounds confine the fit to the basin around the identity, which is what makes a
    rigid chamfer fit well posed when the two shapes are known to start roughly
    aligned -- chamfer on its own does not reliably prefer the correct alignment
    over a large flip. Both are enforced by a smooth radial squash that is the
    identity near zero and never produces a dead gradient (poses.free.bound_norm).
    """

    kind: str = "none"                      # none | so3 | se3
    max_angle_deg: Optional[float] = None   # largest rotation in degrees; None = unbounded
    # Largest translation, in NORMALISED units -- joint_normalize fits the pair into
    # a domain of width 1, so 0.1 is a tenth of its bounding box. "se3" only.
    max_translation: Optional[float] = None

    KINDS = ("none", "so3", "se3")

    @classmethod
    def from_dict(cls, d: dict) -> "PoseCfg":
        reject_unknown_keys(d, {"kind", "max_angle_deg", "max_translation"},
                            what="code.pose")
        kind = d.get("kind", "none")
        if kind not in cls.KINDS:
            raise ValueError(
                f"unknown pose kind {kind!r}; expected one of {list(cls.KINDS)}"
            )
        cfg = cls(kind=kind,
                  max_angle_deg=d.get("max_angle_deg"),
                  max_translation=d.get("max_translation"))
        # A translation bound under a rotation-only pose would be accepted and then
        # ignored, which reads as "bounded" in the config and is not.
        if cfg.max_translation is not None and kind != "se3":
            raise ValueError(
                f"max_translation is meaningless for pose kind {kind!r}, which has "
                f"no translation. Use kind: se3, or drop max_translation."
            )
        return cfg


@dataclass
class CodeCfg:
    kind: str = "none"                      # none | auto_decoder | encoder
    n_z: int = 256
    position_aware: bool = False            # use grid interpolation (True) or broadcast (False)?
    conditioning_method: str = "concat"     # concat | film — how to apply features to velocity
    grid: int = 2                           # g — grid resolution (only if position_aware=True)
    grid_channels: int = 32                 # C — grid-head output per corner (only if position_aware=True)

    # Encoder-specific config (only used if kind == "encoder")
    encoder_config: Optional[EncoderConfig] = dataclass_field(default=None)
    graph_spec: Optional[GraphSpec] = dataclass_field(default=None)

    # The rigid pose this code source carries, if any. Orthogonal to `kind`.
    pose: PoseCfg = dataclass_field(default_factory=PoseCfg)


@dataclass
class ChamferSideCfg:
    """One direction of a field-weighted chamfer term (data_name: field_weighted_chamfer).

    Generalises "one shared source_to_target_weight/target_to_source_weight plus
    one shared loss.sigma" into one independent (weight, base_sigma) PER
    DIRECTION, optionally modulated by a per-vertex precision field.

    weight and base_sigma are read by the data term itself (FieldWeightedCDData):
    data weight for this direction = weight / (2 * base_sigma**2), exactly the
    same "data weight = 1/(2*sigma**2)" convention loss.sigma already uses
    elsewhere in this file, just local to one direction instead of shared.

    field_name/field_source are deliberately NOT read by the data term -- turning
    a field name into an actual per-vertex tensor needs the run's normalised
    shapes, which do not exist yet when Registry.create builds the data term (see
    LossCfg.data_kwargs). They are resolved once, later, at build time (see
    field_weights.resolve_point_weight_field and runner._add_field_weighted_chamfer),
    into a tensor that reaches the data term the same way any other per-point
    weight already does: as pred_w/tgt_w, supplied by UnidirectionalMappingError
    on every call.

    Attributes:
        weight: how much this direction counts in the combined data loss (same
            role as source_to_target_weight/target_to_source_weight elsewhere).
        base_sigma: this direction's own noise scale, in the same normalised
            length units as loss.sigma.
        field_name: name of a point field to read. None (the default) means a
            CONSTANT field: every point in this direction is weighted equally,
            which is today's plain chamfer with a local sigma. Otherwise
            interpreted according to field_transform below.
        field_source: path to the .vtp the field lives on. None (the default)
            reads it from the run's own source/target shape -- whichever one
            this direction is indexed by (source_to_target reads the SOURCE's
            fields, target_to_source reads the TARGET's). If given, the file
            must describe the SAME vertices, in the same order, as that shape.
        field_transform: how to turn the named field into a per-vertex
            precision MULTIPLIER (dimensionless -- it scales base_sigma's
            precision up/down, it does not replace it):
              None (default) -- the field IS the multiplier already, used as-is.
              "gaussian_distance" -- the field is a DISTANCE (e.g. to the
                nearest landmark: small near it, large far away); converted via
                weight = exp(-0.5 * (distance / field_scale)**2), turning a
                "close to X" measurement into a "how much X matters here"
                weight, which a raw distance is not -- used directly it would
                emphasise points FAR from X, exactly backwards.
        field_scale: required when field_transform is set. Same units as the
            named field's own values (e.g. a landmark-distance field's
            world-frame length) -- no FrameTransform conversion needed, since
            it only ever appears as a RATIO against the field's own values.
    """
    weight: float = 1.0
    base_sigma: float = 0.1
    field_name: Optional[str] = None
    field_source: Optional[str] = None
    field_transform: Optional[str] = None
    field_scale: Optional[float] = None

    TRANSFORMS = ("gaussian_distance",)

    @classmethod
    def from_dict(cls, d: dict) -> "ChamferSideCfg":
        reject_unknown_keys(d, {"weight", "base_sigma", "field_name", "field_source",
                                "field_transform", "field_scale"},
                            what="chamfer side")
        weight = float(d.get("weight", 1.0))
        base_sigma = float(d.get("base_sigma", 0.1))
        if weight < 0:
            raise ValueError(f"chamfer side weight must be >= 0, got {weight}")
        if base_sigma <= 0:
            raise ValueError(f"chamfer side base_sigma must be > 0, got {base_sigma}")

        field_transform = d.get("field_transform")
        field_scale = d.get("field_scale")
        if field_transform is not None:
            if field_transform not in cls.TRANSFORMS:
                raise ValueError(f"unknown field_transform {field_transform!r}; "
                                 f"expected one of {cls.TRANSFORMS} or null")
            if field_scale is None:
                raise ValueError(
                    f"field_transform={field_transform!r} needs a field_scale "
                    f"(same units as the named field's own values); pass one, "
                    f"or drop field_transform to use the field as-is."
                )
            field_scale = float(field_scale)
            if field_scale <= 0:
                raise ValueError(f"field_scale must be > 0, got {field_scale}")

        return cls(weight=weight, base_sigma=base_sigma,
                   field_name=d.get("field_name"), field_source=d.get("field_source"),
                   field_transform=field_transform, field_scale=field_scale)


@dataclass
class LossCfg:
    data_name: str = "chamfer"              # chamfer | weighted_chamfer | field_weighted_chamfer | pcd | ncd | l2 | sinkhorn | emd | none
    # Constructor kwargs for the data term, passed straight to Registry.create.
    # The chamfer family (chamfer/weighted_chamfer/pcd/ncd) takes per-direction
    # weights -- source_to_target_weight, target_to_source_weight, both 1.0 by
    # default, named for the `source:`/`target:` shapes above. Setting
    # source_to_target_weight to 0.0 makes the term ONE-SIDED, which is what a
    # PARTIAL target needs: "every source point lies on the target surface" is
    # false when most of the source was never scanned, and minimising it
    # collapses the source into the retained patches. See TwoSidedDataTerm.
    #
    # The same chamfer family also takes an optional ROBUST kernel, quadratic
    # below robust_breakpoint and one of {linear, truncated, exponential} above it
    # -- e.g. {robust_kernel: linear, robust_breakpoint: 0.05, robust_slope: 0.0}
    # caps a far correspondence's pull instead of letting it grow without bound.
    # robust_breakpoint is in the same normalised length units as sigma below.
    # See data_terms.ROBUST_KINDS / _robust_penalty for exactly what each kind does.
    #
    # data_name: field_weighted_chamfer takes a DIFFERENT shape: two ChamferSideCfg
    # dicts, {source_to_target: {...}, target_to_source: {...}}, each with its own
    # weight/base_sigma/field_name/field_source instead of one shared
    # source_to_target_weight/target_to_source_weight plus the one shared sigma
    # below (which this data term ignores -- see ChamferSideCfg and
    # FieldWeightedCDData for what each key does, and runner.build for how
    # field_name/field_source get resolved once the shapes are loaded).
    #
    # data_name: none takes NO kwargs -- it is the explicit "no chamfer term at
    # all" opt-out, always zero/no-gradient, for runs where loss.landmark_weight
    # (or another non-chamfer term) is meant to be the only positional
    # supervision. See data_terms.NullData.
    data_kwargs: dict = dataclass_field(default_factory=dict)
    direction: str = "forward"              # forward | bidirectional
    sigma: float = 0.1                      # data weight = 1/(2σ²), named per D4
    kinetic_weight: float = 1.0
    code_reg_weight: float = 0.0
    weight_decay: float = 0.0               # Θ only — see Phase 6
    isometry_weight: float = 0.0            # disabled by default
    equivariant_deformation_weight: float = 0.0   # ||phi(g.T) - g.phi(T)||^2; 0 = not computed
    equivariant_deformation_kwargs: dict = dataclass_field(default_factory=dict)  # field_only, translation
    pose_supervision_weight: float = 0.0          # ||R_hat - R_aug||_F^2 against the drawn element; 0 = not computed
    pose_supervision_kwargs: dict = dataclass_field(default_factory=dict)        # translation
    flow_rotation_penalty_weight: float = 0.0     # ||R_flow - I||_F^2; fixes the pose/deformation gauge
    flow_rotation_penalty_kwargs: dict = dataclass_field(default_factory=dict)
    landmark_weight: float = 0.0            # 0 = disabled, matches isometry_weight
    # Kwargs for landmarks.build_landmark_term_kwargs (sigma_floor, heatmap_prefix,
    # label_weights, hard_precision) -- NOT constructor kwargs for the module
    # itself (unlike the other *_kwargs fields above), because LandmarkGuidanceLoss's
    # constructor takes already resolved tensors, not raw floats. See runner._build_pair.
    # label_weights: {label: weight} multiplies that label's precision on top
    # of its annotator-agreement-derived one -- SOFT, Mahalanobis-style; e.g.
    # {n: 3.0, sn: 0.5} leans harder on the nasion, eases off subnasale.
    # Omitted labels default to 1.0.
    # hard_precision: {label: precision} REPLACES that label's precision
    # outright (std and label_weights both discarded for it) -- the strict
    # option, for when a landmark's fit must be enforced hard rather than
    # softly weighted. A label can be in label_weights or hard_precision, never
    # both. See build_landmark_term_kwargs for exactly what "hard" can and
    # can't mean for a quadratic penalty under gradient descent.
    landmark_kwargs: dict = dataclass_field(default_factory=dict)
    isometry_type: str = "strain"           # or "det" / "orthogonal"
    isometry_samples: int = 64              # points per step for isometry (64 = ~5x speedup)
    # True keeps the isometry term a pure FIELD regularizer (the trajectory it reads
    # is detached). Set False with pose_application="pre" so the term can also reach
    # the pose -- otherwise the trajectory depends on R but the gradient still cannot
    # get there. Costs backprop through the integration chain.
    isometry_detach: bool = True
    subsample_M: float = 2000               # subsample source to N points before flow. int (absolute) or 0<x<1 (fraction); 0 = disabled
    save_full: bool = False                 # when subsampling: also compute and export full trajectory
    # Open-ended extra terms, resolved through the "loss_term" Registry category.
    # Each entry: {kind: <registry name>, weight: float, name: <metric key>, kwargs: {}}
    # Adding a loss is an entry here plus a registry line -- never a stepper edit.
    terms: List[dict] = dataclass_field(default_factory=list)

    def resolve_subsample_count(self, n_source: int) -> int:
        """Resolve subsample_M to absolute point count given source size.

        Args:
            n_source: number of source points

        Returns:
            Absolute number of points to subsample to (or 0 if disabled)
        """
        if self.subsample_M == 0:
            return 0
        if self.subsample_M >= 1:
            return int(self.subsample_M)
        # Treat as fraction
        return max(1, int(self.subsample_M * n_source))

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
    ``field.width`` is an error rather than a partially-loaded model.

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
    """

    # The keys CheckpointWriter puts in the file, which are exactly the keys the
    # steppers' state_dict/load_state_dict speak. Validated here so a typo'd part
    # fails at config load rather than silently restoring nothing.
    PARTS = ("flow", "codes", "optimizer")

    checkpoint: str
    step: Optional[int] = None
    parts: tuple = PARTS
    continue_step_count: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "InitCfg":
        reject_unknown_keys(d, {"checkpoint", "step", "parts", "continue_step_count"},
                            what="init")
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
                   continue_step_count=bool(d.get("continue_step_count", True)))


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

        field_cfg = FieldCfg(**d.get("field", {})) if "field" in d else FieldCfg()
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
            dtype=dtype,
            use_encoder_pose=use_pose,
            freeze_flow_at_identity=d.get("freeze_flow_at_identity", False),
            pose_application=pose_application,
        )


def _parse_code_config(code_dict: dict) -> CodeCfg:
    """Parse code config, handling nested encoder_config and graph_spec.

    Separates encoder/graph config from base code config and creates
    EncoderConfig/GraphSpec objects when kind == "encoder".

    Note: EncoderConfig/GraphSpec are only imported when needed (lazy import),
    so basic config parsing doesn't require e3nn/torch_geometric.
    """
    # Extract nested configs before passing to CodeCfg
    encoder_cfg_dict = code_dict.pop("encoder_config", {})
    graph_cfg_dict = code_dict.pop("graph_spec", {})
    pose_cfg = PoseCfg.from_dict(code_dict.pop("pose", {}))

    # Parse encoder_config if provided and kind is encoder
    encoder_cfg = None
    if code_dict.get("kind") == "encoder":
        from src.spec import EncoderConfig, EncoderLayerConfig
        declares_latent_dim = "latent_dim" in encoder_cfg_dict
        if encoder_cfg_dict:
            # Parse layer configs
            layers_data = encoder_cfg_dict.pop("layers", [])
            layers = [EncoderLayerConfig(**layer) for layer in layers_data]
            encoder_cfg = EncoderConfig(layers=layers, **encoder_cfg_dict)
        else:
            # Use default EncoderConfig
            encoder_cfg = EncoderConfig()

        # n_z SEEDS latent_dim when encoder_config did not name one. Without this,
        # `code: {kind: encoder, n_z: 256}` trains a 5-dim code -- EncoderConfig's
        # default -- discarding the one dimension the config actually stated, and
        # saying so only in a printed warning. Via replace() rather than by editing
        # the dict before construction, because EncoderConfig.layers must stay
        # non-empty and defaults to a list this function would otherwise have to
        # rebuild. When BOTH are given the rule below is unchanged: the encoder wins.
        if not declares_latent_dim and "n_z" in code_dict:
            encoder_cfg = dataclass_replace(encoder_cfg, latent_dim=code_dict["n_z"])

        # Validate n_z matches encoder's latent_dim
        config_n_z = code_dict.get("n_z", 256)
        encoder_latent_dim = encoder_cfg.latent_dim
        if config_n_z != encoder_latent_dim:
            print(f"WARNING: code.n_z={config_n_z} does not match encoder_config.latent_dim={encoder_latent_dim}")
            print(f"         Using encoder_config.latent_dim={encoder_latent_dim} (encoder determines code dimension)")
            code_dict["n_z"] = encoder_latent_dim

    # Parse graph_spec if provided and kind is encoder
    graph_spec = None
    if code_dict.get("kind") == "encoder":
        from src.spec import GraphSpec
        graph_spec = GraphSpec(**graph_cfg_dict) if graph_cfg_dict else GraphSpec()

    return CodeCfg(encoder_config=encoder_cfg, graph_spec=graph_spec, pose=pose_cfg,
                   **code_dict)
