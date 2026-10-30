"""Builds the pose and code sources, including the encoder-driven ones."""

import torch.nn as nn
from src.resnet_lddmm.resnet_flow.codes import SELF_POSING_KINDS
from src.resnet_lddmm.resnet_flow.poses import TRANSLATING_KINDS
from src.resnet_lddmm.learning.registry import Registry


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


def init_projections(flow, std):
    """Re-initialise every VelocityBlock's output layer N(0, std^2); std 0 keeps the zero init."""
    if std <= 0:
        return
    from src.resnet_lddmm.resnet_flow.fields.blocks import VelocityBlock
    for module in flow.modules():
        if isinstance(module, VelocityBlock):
            nn.init.normal_(module.proj.weight, mean=0.0, std=std)


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
        return Registry.create("code", kind, num_shapes=num_shapes, n_z=code_cfg.n_z,
                               init_std=getattr(code_cfg, "init_std", 0.01))
    else:
        # "none"
        return Registry.create("code", kind)
