"""Assembles a pair or a cohort registration: the composition root behind ``build``."""

import os
import torch
from src.resnet_lddmm.resnet_flow.config import ExperimentCfg, reject_renamed_loss_distance_keys
from src.resnet_lddmm.resnet_flow.guidance.landmarks import resolve_template_points
from src.resnet_lddmm.resnet_flow.conditioning.grid_regularization import grid_penalty_weights
from src.resnet_lddmm.resnet_flow.registration.pair import PairRegistration
from src.resnet_lddmm.resnet_flow.registration.cohort import CohortRegistration
from src.resnet_lddmm.resnet_flow.fields.composite.modes import ModalField
from src.resnet_lddmm.resnet_flow.losses import BidirectionalMappingError, UnidirectionalMappingError
from src.resnet_lddmm.learning.registry import Registry
from src.resnet_lddmm.learning.losses.composer import LossComposer, LossTerm
from src.resnet_lddmm.learning.loader.loaders import CohortBatchLoader, OneBatchLoader

from .common import _dump_config, _normalization_domain, seed_everything, set_default_dtype
from .data import load_cohort, load_pair
from .field_weighted_terms import (
    FIELD_WEIGHTED_DATA_NAMES,
    _add_cohort_field_weighted_chamfer,
    _add_field_weighted_chamfer,
)
from .loss_terms import (
    _add_fshape_term,
    _add_kinetic_field_term,
    _add_landmark_term,
    _add_part_kinetic_terms,
    _build_loss_terms,
)
from .model import build_flow
from .placement import _place_local_mask, _place_modes, _place_symmetry
from .prior import _check_prior_frame, _init_prior_codes, _load_prior_field
from .sources import build_code_source, build_code_source_cohort, build_pose_source, init_projections


def trainable_parameters(module):
    """The parameters of ``module`` an optimizer should get: not the frozen ones (a prior field's)."""
    return [p for p in module.parameters() if p.requires_grad]


def _flow_param_groups(cfg, flow):
    """Optimizer groups for the flow: the network (with weight decay), the mode amplitudes apart.

    The amplitudes get their own learning rate (field.modes.lr, in mm per step)
    and no decay: decay would pull them to zero, a second price on top of the
    discounted kinetic one.
    """
    field = flow.field
    if not isinstance(field, ModalField):
        return [{"params": trainable_parameters(flow), "weight_decay": cfg.loss.weight_decay}]
    amplitude = field.amplitude
    network = [p for p in flow.parameters() if p is not amplitude]
    return [{"params": network, "weight_decay": cfg.loss.weight_decay},
            {"params": [amplitude], "weight_decay": 0.0, "lr": cfg.field.modes["lr"]}]


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
    if cfg.field.modes is not None and pose_application == "pre" and use_encoder_pose:
        raise ValueError(
            "field.modes with pose_application='pre': the pose would move the template "
            "before the flow, away from where its designed modes are placed. Use 'post'."
        )
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
    init_projections(flow, getattr(cfg.field, "proj_init_std", 0.0))
    _check_prior_frame(cfg)
    _load_prior_field(cfg, flow)

    # Registry.create: data term and isometry loss (shared). Old robust_* keys
    # fail here with their replacement rather than as an unexpected argument.
    reject_renamed_loss_distance_keys(cfg.loss.data_kwargs, "loss.data_kwargs")
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


def _build_pair(cfg, flow, data_term, composer, iso_loss, augmentation, use_encoder_pose=False, loss_terms=None):
    """Build PairRegistration stepper."""
    if cfg.train.freeze_flow:
        raise NotImplementedError("train.freeze_flow is for train.mode='cohort' (fitting codes "
                                  "to a trained flow); a pair run has no codes to train.")
    # Load and normalize shapes
    source_norm, target_norm, transform = load_pair(cfg)

    # source_norm is still the full Shape (point_fields intact), which is what
    # landmark resolution needs -- do this before it gets wrapped into the
    # field-stripping SimpleBatch below.
    loss_terms = _add_landmark_term(cfg, source_norm, transform, composer, loss_terms)
    loss_terms = _add_kinetic_field_term(cfg, source_norm, transform, composer, loss_terms)
    loss_terms = _add_fshape_term(cfg, source_norm, target_norm, transform, composer, loss_terms)
    loss_terms = _add_part_kinetic_terms(cfg, composer, loss_terms)
    _place_local_mask(cfg, source_norm, transform, flow)
    _place_modes(cfg, source_norm, transform, flow)
    _place_symmetry(cfg, source_norm, transform, flow)

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
    # and this is exactly the single-group optimizer it replaces. Designed-mode
    # amplitudes, when configured, are a third group (_flow_param_groups).
    flow_groups = _flow_param_groups(cfg, flow)
    optimizer = torch.optim.Adam([
        flow_groups[0],
        {"params": list(code_source.parameters()), "weight_decay": 0.0},
        *flow_groups[1:],
    ], lr=cfg.train.lr)

    # Create stepper
    stepper = PairRegistration(flow, code_source, data_term, mapping_error, composer, optimizer, iso_loss=iso_loss, augmentation=augmentation, use_encoder_pose=use_encoder_pose, loss_terms=loss_terms)

    # Dump config to output dir
    _dump_config(cfg, "pair")

    return stepper, loader, transform


def _set_grid_penalty_weights(cfg, cohort_norm, template_norm, code_source):
    """Weight a direct grid's code penalty per member and node (cfg.code.grid_reg).

    Coverage reads each member's own target points, distance the template's points,
    region its heatmap_<label> fields; all in the normalised frame the grid spans.
    """
    weights = grid_penalty_weights(
        cfg.code,
        [member.points[0] for member in cohort_norm],
        template_norm.points[0],
        lambda labels: resolve_template_points(template_norm, labels),
        _normalization_domain(cfg),
    )
    code_source.set_penalty_weights(weights)
    per_node = weights[:, ::cfg.code.grid_channels]
    print(f"[grid_reg] {cfg.code.grid_reg.kind}: node weights {per_node.min().item():.2f}..{per_node.max().item():.2f}, "
          f"mean {per_node.mean().item():.2f}; at low: {(per_node <= cfg.code.grid_reg.low + 1e-6).float().mean().item():.0%}"
          + ("" if cfg.loss.code_reg_weight > 0 else " (inactive: loss.code_reg_weight is 0)"))


def _build_cohort(cfg, flow, data_term, composer, iso_loss, augmentation, use_encoder_pose=False, loss_terms=None):
    """Build CohortRegistration stepper."""
    if cfg.loss.landmark_weight > 0:
        raise NotImplementedError(
            "loss.landmark_weight > 0 is not supported under train.mode='cohort': "
            "cohort mode has no single template Shape to resolve heatmap_<label> "
            "landmarks against (see landmarks.py). Use mode='pair', or drop "
            "landmark_weight."
        )

    if cfg.loss.fshape_weight > 0:
        raise NotImplementedError(
            "loss.fshape_weight > 0 is not supported under train.mode='cohort': the term "
            "is built from one template mesh and one target (see runner._add_fshape_term). "
            "Use mode='pair', or drop fshape_weight."
        )

    if cfg.field.kind == "time_varying_local":
        raise NotImplementedError(
            "field.kind='time_varying_local' is not supported under train.mode='cohort': "
            "its region is placed from a single template's landmarks (see "
            "runner._place_local_mask). Use mode='pair'."
        )

    if cfg.field.modes is not None:
        raise NotImplementedError(
            "field.modes is not supported under train.mode='cohort': the modes are placed "
            "on a single source (runner._place_modes) and their amplitudes are one set per "
            "run, not per shape. Use mode='pair'."
        )

    if cfg.loss.data_name in FIELD_WEIGHTED_DATA_NAMES and cfg.loss.direction == "bidirectional":
        raise NotImplementedError(
            f"data_name={cfg.loss.data_name!r} under train.mode='cohort' supports "
            "direction='forward' only."
        )
    if cfg.loss.data_name == "field_weighted_point2plane_chamfer" and cfg.augmentation.kind != "none":
        raise NotImplementedError(
            "field_weighted_point2plane_chamfer under train.mode='cohort' needs "
            "augmentation.kind='none': the cohort stepper augments the members' points "
            "but not their normals."
        )

    _, cohort_norm, target_norm, transform = load_cohort(cfg)
    _place_symmetry(cfg, target_norm, transform, flow)

    # Create mapping error strategy after knowing source size (use first shape)
    subsample_n = cfg.loss.resolve_subsample_count(cohort_norm[0].points.shape[1])
    if cfg.loss.direction == "bidirectional":
        mapping_error = BidirectionalMappingError(subsample_n=subsample_n if subsample_n > 0 else None, save_full=cfg.loss.save_full)
    else:  # default to forward
        mapping_error = UnidirectionalMappingError(subsample_n=subsample_n if subsample_n > 0 else None, save_full=cfg.loss.save_full, pose_application=cfg.pose_application)

    _add_cohort_field_weighted_chamfer(cfg, cohort_norm, target_norm, mapping_error, data_term)
    # The cohort's flowed shape is its single template (target_norm), so the kinetic
    # weight field is read from it exactly as a pair run reads it from its source;
    # the stepper passes each step's subsample draw as LossContext.template_idx.
    loss_terms = _add_kinetic_field_term(cfg, target_norm, transform, composer, loss_terms)
    loss_terms = _add_part_kinetic_terms(cfg, composer, loss_terms)

    # Create loader for cohort
    loader = CohortBatchLoader(cohort_norm, target_norm, batch_size=cfg.train.batch)

    # Build code source (handles none, auto_decoder, encoder with cohort-specific logic)
    code_source = build_code_source_cohort(len(cohort_norm), cfg.code.kind, cfg.code)
    _init_prior_codes(cfg, code_source)
    if cfg.code.kind == "auto_decoder" and cfg.code.position_aware and cfg.code.grid_direct:
        _set_grid_penalty_weights(cfg, cohort_norm, target_norm, code_source)

    # Create optimizer with two param groups: flow with weight_decay, codes without.
    # A frozen flow (fitting new targets to a trained one) gets no gradient and no group.
    if cfg.train.freeze_flow:
        free = []
        for name, param in flow.named_parameters():
            param.requires_grad = any(pattern in name for pattern in cfg.train.freeze_flow_except)
            if param.requires_grad:
                free.append(param)
        groups = [{"params": code_source.parameters(), "weight_decay": 0.0}]
        if cfg.train.freeze_flow_except and not free:
            raise ValueError(f"train.freeze_flow_except {list(cfg.train.freeze_flow_except)} matches no "
                             f"flow parameter")
        if free:
            groups.append({"params": free, "weight_decay": 0.0})
            print(f"[freeze_flow] the flow is frozen except {len(free)} tensors "
                  f"({sum(p.numel() for p in free)} values) matching {list(cfg.train.freeze_flow_except)}, "
                  f"shared over the cohort; and the codes")
        else:
            print("[freeze_flow] the flow is frozen; only the codes are trained")
    else:
        groups = [
            {"params": trainable_parameters(flow), "weight_decay": cfg.loss.weight_decay},
            {"params": code_source.parameters(), "weight_decay": 0.0},
        ]
    optimizer = torch.optim.Adam(groups, lr=cfg.train.lr)

    # Create stepper
    stepper = CohortRegistration(flow, code_source, data_term, mapping_error, composer, optimizer, iso_loss=iso_loss, augmentation=augmentation, use_encoder_pose=use_encoder_pose, loss_terms=loss_terms)

    # Dump config to output dir
    _dump_config(cfg, "cohort")

    return stepper, loader, transform
