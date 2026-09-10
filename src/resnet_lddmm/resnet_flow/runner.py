"""Runner: composition root for ResNetLDDMM (STEPS T16).

The ONE place where Registry.create is called and all components are assembled:
config -> shapes -> normalization -> stepper -> orchestrator -> run.

Does NOT import config directly; reads attributes dynamically (layering rule).
"""

import os
import json
import glob
import random
import dataclasses

import numpy as np
import torch

from src.resnet_lddmm.resnet_flow.checkpoint import checkpoint_step, load_checkpoint, resolve_checkpoint
from src.resnet_lddmm.resnet_flow.codes import SELF_POSING_KINDS
from src.resnet_lddmm.resnet_flow.config import ExperimentCfg, CENTERING_DOMAINS
from src.resnet_lddmm.resnet_flow.poses import TRANSLATING_KINDS
from src.resnet_lddmm.resnet_flow.io import load_shape, joint_normalize, export_trajectory, export_shape
from src.resnet_lddmm.resnet_flow.landmarks import build_landmark_term_kwargs
from src.resnet_lddmm.resnet_flow.field_weights import resolve_point_weight_field
from src.resnet_lddmm.resnet_flow.registration.pair import PairRegistration
from src.resnet_lddmm.resnet_flow.registration.cohort import CohortRegistration
from src.resnet_lddmm.resnet_flow.flow import NeuralODEFlow, IdentityFlowWrapper
from src.resnet_lddmm.resnet_flow.integrators import ForwardEuler, ModifiedEuler
from src.resnet_lddmm.resnet_flow.losses import UnidirectionalMappingError, BidirectionalMappingError
from src.resnet_lddmm.resnet_flow import registrations  #  Side effect: registers all component Load component registrations
from src.resnet_lddmm.learning.registry import Registry
from src.resnet_lddmm.learning.losses.composer import LossComposer, LossTerm
from src.resnet_lddmm.learning.loader.loaders import OneBatchLoader, CohortBatchLoader
from src.resnet_lddmm.learning.trainers.E3_end2end import TrainingOrchestrator
from src.resnet_lddmm.learning.callbacks.base import Callback
from src.resnet_lddmm.learning.callbacks.checkpointing import CheckpointWriter
from src.resnet_lddmm.resnet_flow.callbacks import TrajectoryExporter, DiagnosticsCallback, EncoderGraphLogger, GradientLogger, NetworkStructureInspector, RequiresGradMonitor, PoseLogger, PoseShapeExporter, LossLogger

# Data terms taking the per-direction ChamferSideCfg config shape
# ({source_to_target: {...}, target_to_source: {...}} in data_kwargs) and owning
# their precision internally. Membership here decides three things, and a new
# member that misses any one of them fails SILENTLY rather than loudly:
#   * _add_field_weighted_chamfer resolves field_name/field_source at all
#     (missing -> pred_w/tgt_w stay None, the configured field never applies),
#   * build() forces the loss.sigma-derived composer weight to 1.0
#     (missing -> loss.sigma compounds with the per-direction base_sigmas),
#   * _build_cohort refuses the term (missing -> fields silently ignored there).
FIELD_WEIGHTED_DATA_NAMES = ("field_weighted_chamfer",
                             "field_weighted_point2plane_chamfer")


def _encoder_layers_to_list_of_dicts(encoder_config):
    """Convert EncoderConfig.layers (dataclass objects) to list of dicts for Registry.create.

    EncoderConfig stores layers as EncoderLayerConfig dataclass instances;
    Registry.create(**kwargs) expects dicts to pass as kwargs to GroupEncoder.__init__.
    """
    return [
        {
            "in_irreps": layer.in_irreps,
            "target_irreps": layer.target_irreps,
            "spatial_sh_lmax": layer.spatial_sh_lmax,
            "interaction_sh_lmax": layer.interaction_sh_lmax,
        }
        for layer in encoder_config.layers
    ]


def _build_encoder_code_source(encoder_config, graph_spec, n_z):
    """Factory to instantiate EncoderCodes with graph builder and encoder.

    Creates a GraphBuilder and GroupEncoder from spec/config, then wraps
    in EncoderCodes for training. This is only called when code.kind == "encoder".

    Args:
        encoder_config: EncoderConfig instance (from spec.py)
        graph_spec: GraphSpec instance (from spec.py)
        n_z: latent dimension (int)

    Returns:
        EncoderCodes instance ready to use in training
    """
    if encoder_config is None:
        raise ValueError("encoder_config required for code.kind='encoder'")
    if graph_spec is None:
        raise ValueError("graph_spec required for code.kind='encoder'")

    # Create GraphBuilder
    graph_builder = Registry.create("graph_builder", "radius", spec=graph_spec)

    # Create GroupEncoder with layer specs from encoder_config
    layers_as_dicts = _encoder_layers_to_list_of_dicts(encoder_config)
    encoder = Registry.create(
        "encoder", "group_encoder",
        layers_cfg=layers_as_dicts,
        latent_dim=n_z,
        readout=encoder_config.readout,
        readout_heads=encoder_config.readout_heads,
        supernode_sh_lmax=encoder_config.supernode_sh_lmax,
        transformer_type=encoder_config.transformer_type,
        transformer_cfg=encoder_config.transformer_cfg,
        supernode_samples=encoder_config.supernode_samples,
        supernode_seed=encoder_config.supernode_seed,
        area_pool=encoder_config.area_pool,
        latent_mode=encoder_config.latent_mode,
        pose_mode=encoder_config.pose_mode,
        verbose=encoder_config.verbose,
    )

    # Create EncoderCodes wrapping both
    code_source = Registry.create(
        "code", "encoder",
        graph_builder=graph_builder,
        encoder=encoder,
        n_z=n_z,
    )

    return code_source


class VerboseCallback(Callback):
    """Log loss progression during training."""

    def __init__(self, log_every=1, pose_only_mode=False):
        super().__init__(every_n_steps=log_every)
        self.pose_only_mode = pose_only_mode

    def on_train_start(self, ctx):
        """Print training mode info at startup."""
        if self.pose_only_mode:
            print("[POSE-ONLY MODE] Flow is frozen at identity; encoder learns pose only.")

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Print loss at cadence."""
        if not self._due(step):
            return
        loss = metrics.get("loss", 0.0)
        # Against the LAST step number, not the count: a resumed run starts at
        # ctx.start_step, and "Step 1301/2000" for 2000 further steps reads as a
        # run that is nearly over when it has barely begun.
        last_step = ctx.start_step + ctx.num_steps
        progress = f"Step {step+1:4d}/{last_step} | loss: {loss:.6f}"

        # Add any other metrics
        for key in sorted(metrics.keys()):
            if key != "loss" and not key.startswith("diag/"):
                progress += f" | {key}: {metrics[key]:.6f}"

        print(progress)

    def _due(self, step):
        """Check if we should log this step."""
        return step % self.every_n_steps == 0 or step == 0


def seed_everything(seed):
    """Mirrors seeding: returns loader rng or None if seed is None."""
    if seed is None:
        return None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    key = torch.Generator(device="cpu")
    key.manual_seed(seed)
    return key


def _normalization_domain(cfg):
    """cfg.centering -> the (min, max) domain joint_normalize places shapes in.

    getattr so a stub config without the field still builds, matching how the rest
    of this module reads optional attributes.
    """
    return CENTERING_DOMAINS[getattr(cfg, "centering", "box")]


def set_default_dtype(cfg):
    """cfg.dtype -> the process-wide default dtype.

    Call BEFORE anything is constructed: nn.Module parameters take the default
    dtype at __init__, so setting this later would leave a float32 model reading
    float64 data. Process-wide, by PyTorch's design -- there is no per-model
    equivalent.
    """
    torch.set_default_dtype(getattr(torch, getattr(cfg, "dtype", "float32")))


def build_conditioning(cfg):
    """cfg.code -> the Conditioning module the field is built around, or None.

    Two orthogonal axes: position_aware (grid interpolation or broadcast?) and
    conditioning_method (concat or FiLM modulation?). An unconditioned run
    (code.kind == "none") has no conditioning at all -- including when that code
    source carries a pose, since a pose does not condition the field.
    """
    if cfg.code.kind == "none":
        return None
    if cfg.code.position_aware:
        return Registry.create(
            "conditioning", "position_aware",
            n_z=cfg.code.n_z,
            g=cfg.code.grid,
            channels=cfg.code.grid_channels
        )
    # Broadcast-based conditioning (concat or FiLM)
    return Registry.create("conditioning", cfg.code.conditioning_method, n_z=cfg.code.n_z)


def build_field(cfg, conditioning):
    """cfg.field -> the VelocityField, dispatching on cfg.field.kind.

    Args:
        cfg: ExperimentCfg
        conditioning: Conditioning module or None, from build_conditioning

    Returns:
        VelocityField instance
    """
    if cfg.field.kind == "stationary":
        return Registry.create(
            "field", "stationary",
            activation=cfg.field.activation,
            conditioning=conditioning
        )
    if cfg.field.kind == "equivariant_stationary":
        return Registry.create(
            "field", "equivariant_stationary",
            hidden_irreps=cfg.field.hidden_irreps,
            gate_hidden_dim=cfg.field.gate_hidden_dim,
            use_tensor_product_self=cfg.field.use_tensor_product_self,
            layers_cfg=cfg.field.layers,
            conditioning=conditioning
        )
    if cfg.field.kind == "equivariant_contextual":
        return Registry.create(
            "field", "equivariant_contextual",
            context_irreps=getattr(cfg.field, 'context_irreps', "32x0e + 16x1o"),
            sh_lmax=getattr(cfg.field, 'sh_lmax', 2),
            layers_cfg=cfg.field.layers,
            conditioning=conditioning
        )
    if cfg.field.kind == "equivariant_contextual_simple":
        return Registry.create(
            "field", "equivariant_contextual_simple",
            hidden_dim=getattr(cfg.field, 'hidden_dim', 128),
            n_layers=getattr(cfg.field, 'n_layers', 3),
            conditioning=conditioning
        )
    if cfg.field.kind == "equivariant_template":
        return Registry.create(
            "field", "equivariant_template",
            layers_cfg=cfg.field.layers,
            conditioning=conditioning
        )
    return Registry.create(
        "field", cfg.field.kind,
        num_blocks=cfg.field.num_steps,
        width=cfg.field.width,
        activation=cfg.field.activation,
        conditioning=conditioning
    )


def build_flow(cfg):
    """cfg -> the flow module, freshly initialised.

    The ONE definition of the deformation architecture, so that the module a
    checkpoint was written from is by construction the module its weights load
    back into: a state_dict mismatch can then only come from a changed config,
    never from two copies of this branch drifting apart. Training calls it from
    build(); apply calls it to reconstruct the same model around a checkpoint.

    Args:
        cfg: ExperimentCfg

    Returns:
        NeuralODEFlow, wrapped in IdentityFlowWrapper when the config freezes it
    """
    flow = NeuralODEFlow(
        field=build_field(cfg, build_conditioning(cfg)),
        direct=ForwardEuler(),
        inverse=ModifiedEuler(),
        num_steps=cfg.field.num_steps
    )
    # Conditionally wrap flow at identity for pose-only training
    if getattr(cfg, 'freeze_flow_at_identity', False):
        flow = IdentityFlowWrapper(flow)
    return flow


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


def _build_loss_terms(loss_cfg):
    """cfg.loss.terms -> (composer entries, {name: term module}).

    Zero-weight entries are dropped rather than computed and multiplied by zero,
    matching how isometry has always been skipped when disabled.

    Args:
        loss_cfg: LossCfg with a .terms list of {kind, weight, name, kwargs} dicts

    Returns:
        (list of LossTerm, dict mapping metric name -> instantiated term)

    Raises:
        ValueError: on a missing 'kind', or a name colliding with another term
    """
    specs = list(loss_cfg.terms)

    # First-class weights, configured like every other term (isometry_weight,
    # kinetic_weight, ...). Strictly > 0 to be built at all: at 0 the term is never
    # instantiated and its extra work never runs, rather than being computed and
    # multiplied by zero. A new gated term is one entry here.
    for prefix, kind in (("equivariant_deformation", "equivariant_deformation_loss"),
                         ("pose_supervision", "pose_supervision_loss"),
                         ("flow_rotation_penalty", "flow_rotation_penalty")):
        weight = float(getattr(loss_cfg, f"{prefix}_weight", 0.0) or 0.0)
        if weight > 0:
            specs.append({
                "kind": kind,
                "weight": weight,
                "kwargs": dict(getattr(loss_cfg, f"{prefix}_kwargs", None) or {}),
            })

    entries, modules = [], {}
    for spec in specs:
        if "kind" not in spec:
            raise ValueError(f"loss term needs a 'kind': {spec}")
        name = spec.get("name", spec["kind"])
        if name in modules:
            raise ValueError(
                f"duplicate loss term name {name!r}; give one of them an explicit "
                f"'name' so their metrics stay distinguishable"
            )
        weight = float(spec.get("weight", 1.0))
        if weight <= 0:
            continue
        # Registry.create raises with the list of valid names on an unknown kind.
        modules[name] = Registry.create("loss_term", spec["kind"], **spec.get("kwargs", {}))
        entries.append(LossTerm(name, weight=weight))
    return entries, modules


def _add_landmark_term(cfg, template_shape, transform, composer, loss_terms):
    """Resolve + attach the landmark-guidance term, if cfg.loss.landmark_weight > 0.

    Lives here rather than in ``_build_loss_terms`` because it needs data that
    only exists AFTER shapes are loaded -- the template's ``point_fields`` and
    the FrameTransform both come out of ``load_pair``/``joint_normalize``, which
    runs later than ``_build_loss_terms`` in ``build()``. See landmarks.py and
    ``losses.terms.LandmarkGuidanceLoss``.

    Args:
        cfg: ExperimentCfg
        template_shape: the NORMALISED template Shape (must still carry
            ``point_fields`` -- pass the Shape itself, not a stripped batch
            wrapper such as ``_build_pair``'s ``SimpleBatch``)
        transform: the FrameTransform training runs in
        composer: LossComposer already built; a "landmark" LossTerm is appended
            to it in place when enabled
        loss_terms: {name: module} dict already built by ``_build_loss_terms``

    Returns:
        ``loss_terms`` unchanged when disabled; otherwise a NEW dict (the
        caller's dict is not mutated) with a "landmark" entry added.

    Raises:
        ValueError: ``loss.landmark_weight > 0`` but ``cfg.landmarks`` is unset.
    """
    if cfg.loss.landmark_weight <= 0:
        return loss_terms

    if not cfg.landmarks:
        raise ValueError(
            "loss.landmark_weight > 0 but no `landmarks:` folder is configured; "
            "set `landmarks: <path to landmark CSVs>` or drop landmark_weight."
        )

    kwargs = build_landmark_term_kwargs(cfg, template_shape, transform, **cfg.loss.landmark_kwargs)
    labels = kwargs.pop("labels")

    loss_terms = dict(loss_terms or {})
    loss_terms["landmark"] = Registry.create("loss_term", "landmark_guidance", **kwargs)
    composer.terms.append(LossTerm("landmark", weight=cfg.loss.landmark_weight))

    # A quick visual sanity check -- did the argmax-resolved template vertices
    # land on the right anatomy? Written once, at build time, next to the other
    # reference shapes so it can be opened alongside template_points.vtp.
    export_shape(kwargs["template_points"][0], None, transform,
                 os.path.join(cfg.output_dir, "landmarks_template.vtp"),
                 label="Landmarks (template)")

    print(f"[landmarks] guiding on {len(labels)} labels: {', '.join(labels)}")
    return loss_terms


def _add_field_weighted_chamfer(cfg, source_norm, target_norm, transform, mapping_error, data_term):
    """Resolve each direction's point-weight field (if any) and attach it to mapping_error.

    A no-op whenever cfg.loss.data_name is not in FIELD_WEIGHTED_DATA_NAMES -- the
    field_name/field_source keys only exist in THAT data term's own config shape
    (ChamferSideCfg), so nothing here applies to any other data_name.

    Lives here rather than in _build_loss_terms because, like landmark
    resolution, it needs the NORMALISED shapes and cannot run until after
    load_pair -- Registry.create builds ``data_term`` itself earlier, with only
    the raw config dict available.

    Args:
        cfg: ExperimentCfg
        source_norm, target_norm: NORMALISED shapes (point_fields AND faces
            intact -- pass the Shape itself, not a stripped batch wrapper)
        transform: the run's FrameTransform, forwarded to the sanity-check
            export so the visualised mesh lands back in world coordinates
        mapping_error: the already-built UnidirectionalMappingError for this
            run; gets its point-weight fields attached IN PLACE
        data_term: the already-built FieldWeightedCDData, read back only for
            its already-validated field_name/field_source/field_transform/
            field_scale attributes -- avoids re-parsing cfg.loss.data_kwargs a
            second time.

    Raises:
        ValueError: cfg.loss.direction == "bidirectional" -- BidirectionalMappingError
            has no field-gathering path yet.
    """
    if cfg.loss.data_name not in FIELD_WEIGHTED_DATA_NAMES:
        return

    if cfg.loss.direction == "bidirectional":
        raise ValueError(
            f"data_name={cfg.loss.data_name!r} is not supported under "
            "loss.direction='bidirectional' yet -- BidirectionalMappingError has "
            "no field-gathering path (and no normal-gathering path either). Use "
            "direction='forward' (the default), or drop field_name/field_source "
            "from both directions."
        )

    # Fail at BUILD time, not at the first training step: the target's normals are
    # a static input, so "the file has none" is knowable before a single gradient
    # step has been paid for.
    if cfg.loss.data_name == "field_weighted_point2plane_chamfer":
        if target_norm.normals is None:
            raise ValueError(
                f"data_name={cfg.loss.data_name!r} needs a 'normal' point array on "
                f"the target ({cfg.target}), but it carries none. Either re-run "
                "scan_fusion (it writes the field itself under compute_normals: "
                "true, on the final post-min_support topology), or bake it onto "
                "the existing file with:\n"
                f"    python scripts/write_surface_normals.py {cfg.target}\n"
                "(the target has faces, so these are exact mesh normals, not "
                "estimated ones). io.load_shape reads the array back automatically "
                "and joint_normalize carries it through unchanged -- its single "
                "uniform scale leaves normal DIRECTIONS untouched."
            )
        # A NaN normal is unrecoverable and does not announce itself until the
        # whole loss goes NaN, some steps in. Both producers guard their own
        # division, so reaching this means the array was written by something
        # else -- catch it here rather than after a wasted run.
        if not torch.isfinite(target_norm.normals).all():
            n_bad = int((~torch.isfinite(target_norm.normals).all(dim=-1)).sum())
            raise ValueError(
                f"the target's 'normal' array has {n_bad} non-finite row(s) "
                f"({cfg.target}). A NaN normal propagates through the entire data "
                "term and silently destroys the run. Re-bake it with "
                "scripts/write_surface_normals.py, which returns degenerate "
                "vertices as the ZERO vector instead (scored as zero plane-distance, "
                "falling back to the point-to-point component)."
            )

    template_field = resolve_point_weight_field(
        data_term.source_to_target_field_name, data_term.source_to_target_field_source,
        fallback_shape=source_norm,
        field_transform=data_term.source_to_target_field_transform,
        field_scale=data_term.source_to_target_field_scale,
    )
    sample_field = resolve_point_weight_field(
        data_term.target_to_source_field_name, data_term.target_to_source_field_source,
        fallback_shape=target_norm,
        field_transform=data_term.target_to_source_field_transform,
        field_scale=data_term.target_to_source_field_scale,
    )
    mapping_error.set_point_weight_fields(template_field=template_field, sample_field=sample_field)

    # A visual sanity check, same spirit as landmarks_template.vtp: the
    # RESOLVED weight (post-transform, if any), written onto the FULL mesh it
    # came from -- with connectivity, so it opens as an actual surface in
    # ParaView rather than a bare point cloud, colourable by "chamfer_weight".
    for label, shape, field, filename in (
        ("source_to_target (template)", source_norm, template_field, "field_weighted_chamfer_source.vtp"),
        ("target_to_source (target)", target_norm, sample_field, "field_weighted_chamfer_target.vtp"),
    ):
        if field is None:
            print(f"[field_weighted_chamfer] {label}: no field configured (constant)")
            continue
        print(f"[field_weighted_chamfer] {label}: field over {field.shape[0]} "
              f"vertices, range [{field.min().item():.4g}, {field.max().item():.4g}]")
        export_shape(shape.points[0], shape.faces, transform,
                     os.path.join(cfg.output_dir, filename),
                     label=f"FieldWeightedChamfer ({label})",
                     point_fields={"chamfer_weight": field.numpy()})


def build(cfg: ExperimentCfg):
    """Assemble all components from config: the ONLY place Registry.create is called.

    Args:
        cfg: ExperimentCfg with source/target paths and component kinds

    Returns:
        (stepper, loader, transform) where stepper is PairRegistration or CohortRegistration
    """
    set_default_dtype(cfg)

    seed_everything(cfg.train.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Validate encoder_pose configuration
    use_encoder_pose = getattr(cfg, 'use_encoder_pose', False)
    freeze_flow_at_identity = getattr(cfg, 'freeze_flow_at_identity', False)

    # Validate freeze_flow_at_identity first (most specific)
    if freeze_flow_at_identity and not use_encoder_pose:
        raise ValueError(
            "freeze_flow_at_identity requires use_encoder_pose=true (spelled "
            "use_pose in current configs). Freezing the flow leaves the pose as the "
            "only learnable thing, so without one the loss has no parameters at all."
        )

    if freeze_flow_at_identity and cfg.loss.direction == "bidirectional":
        raise ValueError(
            "freeze_flow_at_identity requires unidirectional flow. "
            "Set loss.direction='forward' or disable freeze_flow_at_identity."
        )

    # Then validate use_encoder_pose
    if use_encoder_pose and cfg.loss.direction == "bidirectional":
        raise ValueError(
            "use_encoder_pose currently supports unidirectional flow only. "
            "Set loss.direction='forward' or disable use_encoder_pose."
        )

    pose_application = getattr(cfg, 'pose_application', 'post')
    if pose_application == "pre":
        if cfg.loss.direction == "bidirectional":
            raise ValueError(
                "pose_application='pre' is implemented for unidirectional flow only. "
                "Set loss.direction='forward'."
            )
        if not use_encoder_pose:
            raise ValueError(
                "pose_application='pre' requires use_encoder_pose=true; there is no "
                "pose to apply otherwise."
            )
        # The whole point of "pre" is to give the flow-cost terms a path to the pose.
        # Detaching inside the isometry term cuts that path while leaving the loss
        # VALUE responsive to the pose -- the failure would look like a weak signal
        # rather than a severed one, so it is called out here.
        if cfg.loss.isometry_weight > 0 and cfg.loss.isometry_detach:
            print(
                "[POSE-PRE] loss.isometry_detach=true: the isometry term still cannot "
                "reach the pose. Set loss.isometry_detach=false to couple them "
                "(kinetic couples either way)."
            )
        if freeze_flow_at_identity:
            print(
                "[POSE-PRE] freeze_flow_at_identity=true: the flow is the identity, so "
                "its cost terms are constant and 'pre' changes nothing."
            )

    # Conditioning + field + integrators, and the identity wrap when the config
    # freezes the flow. Shared verbatim with apply, which rebuilds this same module
    # to load a checkpoint into.
    flow = build_flow(cfg)

    # Registry.create: data term and isometry loss (shared)
    data_term = Registry.create(
        "data_term", cfg.loss.data_name,
        **cfg.loss.data_kwargs
    )
    iso_loss = Registry.create(
        "iso_loss", "isometry",
        loss_type=cfg.loss.isometry_type,
        sample_points=cfg.loss.isometry_samples,
        detach_trajectory=cfg.loss.isometry_detach
    ) if cfg.loss.isometry_weight > 0 else None

    # Build loss composer (shared). data/kinetic/code_reg/isometry are produced by
    # the mapping error and code source; cfg.loss.terms adds anything else.
    #
    # field_weighted_chamfer manages precision PER DIRECTION internally (see
    # ChamferSideCfg.base_sigma / FieldWeightedCDData) -- applying loss.sigma on
    # top here as well would silently compound two independent sigmas into one
    # undocumented product, so it is forced to a no-op (1.0) for that data term.
    if cfg.loss.data_name in FIELD_WEIGHTED_DATA_NAMES:
        data_weight = 1.0
        print(f"[loss] data_name={cfg.loss.data_name!r} manages its own "
              "per-direction base_sigma; loss.sigma is not applied on top "
              "(data_weight forced to 1.0).")
    else:
        data_weight = 1.0 / (2 * cfg.loss.sigma**2)
    extra_terms, loss_terms = _build_loss_terms(cfg.loss)
    terms = [
        LossTerm("data", weight=data_weight),
        LossTerm("kinetic", weight=cfg.loss.kinetic_weight),
        LossTerm("code_reg", weight=cfg.loss.code_reg_weight),
        LossTerm("isometry", weight=cfg.loss.isometry_weight),
        *extra_terms,
    ]
    composer = LossComposer(terms)

    # Build augmentation (shared across pair/cohort)
    # Only pass seed for so3 and se3; none doesn't accept it
    aug_kwargs = {}
    if cfg.augmentation.kind in ("so3", "se3"):
        aug_kwargs["seed"] = cfg.augmentation.seed
    if cfg.augmentation.kind == "se3":
        aug_kwargs["translation_scale"] = cfg.augmentation.translation_scale

    augmentation = Registry.create(
        "augmentation", cfg.augmentation.kind,
        **aug_kwargs
    )

    # Branch on training mode (mapping_error created after loading shapes)
    if cfg.train.mode == "cohort":
        return _build_cohort(cfg, flow, data_term, composer, iso_loss, augmentation, use_encoder_pose, loss_terms)
    else:
        return _build_pair(cfg, flow, data_term, composer, iso_loss, augmentation, use_encoder_pose, loss_terms)


def _validate_pose_config(cfg, use_pose):
    """Reject pose configurations that would be silent no-ops.

    Both directions are checked, because both fail QUIETLY otherwise:

      * a run that asks for a pose but names no source applies none -- the
        steppers duck-type on ``get_pose``, and a source without one simply
        returns (None, None). With the flow frozen that leaves no parameters at
        all, and the first backward raises about grad_fn instead of about config.
      * a pose source that is never applied trains six parameters that touch
        nothing, and the run looks like it is doing rigid alignment while the
        output is unchanged.

    Args:
        cfg: ExperimentCfg
        use_pose: the resolved use_pose / use_encoder_pose flag

    Raises:
        ValueError: naming the specific inconsistency
    """
    pose_kind = cfg.code.pose.kind
    self_posing = cfg.code.kind in SELF_POSING_KINDS

    if use_pose and pose_kind == "none" and not self_posing:
        raise ValueError(
            f"use_pose=true but code.kind={cfg.code.kind!r} supplies no pose and "
            f"code.pose.kind is 'none'. Add a pose source, e.g.\n"
            f"    code:\n      kind: {cfg.code.kind}\n      pose:\n"
            f"        kind: se3\n        max_angle_deg: 30\n        max_translation: 0.1\n"
            f"(or use code.kind='encoder', which predicts its own)."
        )

    if pose_kind != "none" and not use_pose:
        raise ValueError(
            f"code.pose.kind={pose_kind!r} builds a pose that nothing applies. "
            f"Set use_pose=true, or remove the pose block."
        )

    if pose_kind != "none" and self_posing:
        raise ValueError(
            f"code.kind={cfg.code.kind!r} predicts its own pose; a code.pose block "
            f"would be built and never read. Remove one of them."
        )

    # Pair only for now: one shared rotation is meaningless across a cohort, and
    # the per-shape indexing that would make it meaningful is not wired up.
    if pose_kind != "none" and cfg.train.mode == "cohort":
        raise ValueError(
            f"code.pose.kind={pose_kind!r} is implemented for train.mode='pair' only. "
            f"Cohort poses need one per shape, indexed by shape id."
        )


def build_pose_source(pose_cfg, num_shapes=1):
    """cfg.code.pose -> a PoseSource, or None when the run has no pose.

    Separate from the code source because the two are orthogonal: this is injected
    into whichever code source the config names, so `pose` and `code.kind` vary
    independently instead of multiplying into combined kinds.

    Args:
        pose_cfg: PoseCfg with .kind and its bounds
        num_shapes: rows in the pose table; 1 for pair mode

    Returns:
        PoseSource instance, or None for kind "none"
    """
    if pose_cfg.kind == "none":
        return None

    kwargs = {"num_shapes": num_shapes, "max_angle_deg": pose_cfg.max_angle_deg}
    if pose_cfg.kind in TRANSLATING_KINDS:
        kwargs["max_translation"] = pose_cfg.max_translation
    return Registry.create("pose", pose_cfg.kind, **kwargs)


def build_code_source(kind, code_cfg, pose=None):
    """Build a ShapeCode instance (none, auto_decoder, or encoder).

    Args:
        kind: "none" | "auto_decoder" | "encoder"
        code_cfg: CodeCfg instance with all settings
        pose: optional PoseSource from build_pose_source, injected into the code
            source. The encoder predicts its own and takes none.

    Returns:
        ShapeCode instance ready to use in training
    """
    if kind == "encoder":
        return _build_encoder_code_source(
            code_cfg.encoder_config,
            code_cfg.graph_spec,
            code_cfg.n_z
        )
    else:
        # For "none" and "auto_decoder", use Registry with standard args
        return Registry.create("code", kind, pose=pose)


def build_code_source_cohort(num_shapes, kind, code_cfg):
    """Build a ShapeCode for cohort mode (auto_decoder and encoder need num_shapes).

    Args:
        num_shapes: number of shapes in cohort
        kind: "none" | "auto_decoder" | "encoder"
        code_cfg: CodeCfg instance with all settings

    Returns:
        ShapeCode instance ready to use in training
    """
    if kind == "encoder":
        return _build_encoder_code_source(
            code_cfg.encoder_config,
            code_cfg.graph_spec,
            code_cfg.n_z
        )
    elif kind == "auto_decoder":
        return Registry.create("code", kind, num_shapes=num_shapes, n_z=code_cfg.n_z)
    else:
        # "none"
        return Registry.create("code", kind)


def _build_pair(cfg, flow, data_term, composer, iso_loss, augmentation, use_encoder_pose=False, loss_terms=None):
    """Build PairRegistration stepper."""
    # Load and normalize shapes
    source_norm, target_norm, transform = load_pair(cfg)

    # source_norm is still the full Shape (point_fields intact), which is what
    # landmark resolution needs -- do this before it gets wrapped into the
    # field-stripping SimpleBatch below.
    loss_terms = _add_landmark_term(cfg, source_norm, transform, composer, loss_terms)

    # Create mapping error strategy after knowing source size
    subsample_n = cfg.loss.resolve_subsample_count(source_norm.points.shape[1])
    if cfg.loss.direction == "bidirectional":
        mapping_error = BidirectionalMappingError(subsample_n=subsample_n if subsample_n > 0 else None, save_full=cfg.loss.save_full)
    else:  # default to forward
        mapping_error = UnidirectionalMappingError(subsample_n=subsample_n if subsample_n > 0 else None, save_full=cfg.loss.save_full, pose_application=cfg.pose_application)

    _add_field_weighted_chamfer(cfg, source_norm, target_norm, transform, mapping_error, data_term)

    # Create loader with normalized shapes
    class SimpleBatch:
        def __init__(self, shape):
            self.points = shape.points  # [1, N, 3]
            self.weights = shape.weights
            self.faces = shape.faces  # [F, 3] for trajectory export
            # [1, N, 3] or None. Carried so a point-to-plane data term can read
            # the TARGET's normals; the mapping error gathers them on the same
            # subsample draw as the points (see _take_normals).
            self.normals = shape.normals

    loader = OneBatchLoader((SimpleBatch(source_norm), SimpleBatch(target_norm)))

    # Build code source (handles none, auto_decoder, encoder), carrying the
    # configured pose. One shape in pair mode, so the pose table has one row.
    code_source = build_code_source(cfg.code.kind, cfg.code,
                                    pose=build_pose_source(cfg.code.pose, num_shapes=1))

    # Two param groups, matching cohort mode: weight decay on the flow only.
    # Decaying a POSE would pull it toward the identity rather than regularise it,
    # and pose bounds already constrain those parameters. For a code source with no
    # parameters -- every config before poses existed -- the second group is empty
    # and this is exactly the single-group optimizer it replaces.
    optimizer = torch.optim.Adam([
        {"params": list(flow.parameters()), "weight_decay": cfg.loss.weight_decay},
        {"params": list(code_source.parameters()), "weight_decay": 0.0},
    ], lr=cfg.train.lr)

    # Create stepper
    stepper = PairRegistration(flow, code_source, data_term, mapping_error, composer, optimizer, iso_loss=iso_loss, augmentation=augmentation, use_encoder_pose=use_encoder_pose, loss_terms=loss_terms)

    # Dump config to output dir
    _dump_config(cfg, "pair")

    return stepper, loader, transform


def _build_cohort(cfg, flow, data_term, composer, iso_loss, augmentation, use_encoder_pose=False, loss_terms=None):
    """Build CohortRegistration stepper."""
    if cfg.loss.landmark_weight > 0:
        raise NotImplementedError(
            "loss.landmark_weight > 0 is not supported under train.mode='cohort': "
            "cohort mode has no single template Shape to resolve heatmap_<label> "
            "landmarks against (see landmarks.py). Use mode='pair', or drop "
            "landmark_weight."
        )

    if cfg.loss.data_name in FIELD_WEIGHTED_DATA_NAMES:
        raise NotImplementedError(
            f"data_name={cfg.loss.data_name!r} is not supported under "
            "train.mode='cohort' yet: field resolution runs only from _build_pair "
            "(see runner._add_field_weighted_chamfer), so pred_w/tgt_w would "
            "silently stay None here even with field_name configured. Use "
            "mode='pair', or drop field_name/field_source from both directions."
        )

    # Load all cohort shapes from directory
    cohort_paths = sorted(glob.glob(os.path.join(cfg.source, "*.obj"))) + \
                   sorted(glob.glob(os.path.join(cfg.source, "*.ply"))) + \
                   sorted(glob.glob(os.path.join(cfg.source, "*.vtp")))
    if not cohort_paths:
        raise ValueError(f"No shape files found in {cfg.source} (.obj, .ply, or .vtp)")

    cohort_shapes = [load_shape(p) for p in cohort_paths]
    target_shape = load_shape(cfg.target)

    # Normalize cohort + template together
    normalized, transform = joint_normalize(cohort_shapes + [target_shape],
                                           domain=_normalization_domain(cfg))
    cohort_norm = normalized[:-1]
    target_norm = normalized[-1]

    # Create mapping error strategy after knowing source size (use first shape)
    subsample_n = cfg.loss.resolve_subsample_count(cohort_norm[0].points.shape[1])
    if cfg.loss.direction == "bidirectional":
        mapping_error = BidirectionalMappingError(subsample_n=subsample_n if subsample_n > 0 else None, save_full=cfg.loss.save_full)
    else:  # default to forward
        mapping_error = UnidirectionalMappingError(subsample_n=subsample_n if subsample_n > 0 else None, save_full=cfg.loss.save_full, pose_application=cfg.pose_application)

    # Create loader for cohort
    loader = CohortBatchLoader(cohort_norm, target_norm, batch_size=cfg.train.batch)

    # Build code source (handles none, auto_decoder, encoder with cohort-specific logic)
    code_source = build_code_source_cohort(len(cohort_norm), cfg.code.kind, cfg.code)

    # Create optimizer with two param groups: flow with weight_decay, codes without
    optimizer = torch.optim.Adam([
        {"params": flow.parameters(), "weight_decay": cfg.loss.weight_decay},
        {"params": code_source.parameters(), "weight_decay": 0.0},
    ], lr=cfg.train.lr)

    # Create stepper
    stepper = CohortRegistration(flow, code_source, data_term, mapping_error, composer, optimizer, iso_loss=iso_loss, augmentation=augmentation, use_encoder_pose=use_encoder_pose, loss_terms=loss_terms)

    # Dump config to output dir
    _dump_config(cfg, "cohort")

    return stepper, loader, transform


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

    written_at = checkpoint_step(state)
    start_step = (written_at + 1) if (init.continue_step_count and written_at is not None) else 0
    print(f"[INIT] restored {', '.join(init.parts)} from {ref.path}"
          + (f" (step {written_at}); continuing from step {start_step}"
             if start_step else " (step counter starts at 0)"))
    return start_step


def _dump_config(cfg, mode):
    """Dump the FULL config to output dir for provenance.

    Every config class from ExperimentCfg down (FieldCfg, CodeCfg, EncoderConfig,
    GraphSpec, ...) is a plain dataclass, so dataclasses.asdict recurses through
    the whole tree -- this must capture every field, not a hand-picked subset,
    since a field left out here is a field a resumed/compared run can't recover.
    """
    config_path = os.path.join(cfg.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({"mode": mode, **dataclasses.asdict(cfg)}, f, indent=2)


def run(cfg: ExperimentCfg, callbacks=None):
    """Load config, build stepper, run orchestrator.

    Args:
        cfg: ExperimentCfg instance
        callbacks: optional list of callbacks (default: VerboseCallback + TrajectoryExporter + DiagnosticsCallback)

    Returns:
        TrainingContext from orchestrator.run
    """
    if callbacks is None:
        # Default callbacks: verbose logging, trajectory export, diagnostics
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
        callbacks = [
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

    stepper, loader, transform = build(cfg)

    # Weights, if the config names a checkpoint to start from. AFTER build (the
    # modules and optimizer have to exist to load into) and BEFORE the callback
    # list is finalised, so a bad init.checkpoint fails before any output is
    # written rather than a few steps in.
    start_step = restore_from(cfg, stepper)

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

    # The loss breakdown as JSON. LAST in the list on purpose: metrics is one dict
    # shared by every callback, so running last is what lets DiagnosticsCallback's
    # diag/* and the gradient stats land in the same record instead of a step late.
    # (re-read the cadence rather than reusing log_every: that name only exists on
    # the branch that builds the default list, not when callbacks were passed in)
    callbacks.append(LossLogger(every_n_steps=getattr(cfg.train, 'log_every', 50)))

    # Hand the frame transform to every exporter that wants one. This runs AFTER all
    # appends above: PoseShapeExporter is added conditionally further up, and assigning
    # before that left it with transform=None, silently exporting the normalised frame
    # under world/ as well.
    for cb in callbacks:
        if isinstance(cb, (TrajectoryExporter, PoseShapeExporter)):
            cb.transform = transform

    orchestrator = TrainingOrchestrator(
        stepper=stepper,
        dataloader=loader,
        callbacks=callbacks,
        log_dir=cfg.output_dir,
    )
    # train.steps is how many steps to RUN, so a resumed run takes that many more
    # rather than stopping the moment the counter passes it.
    return orchestrator.run(num_steps=cfg.train.steps, start_step=start_step)
