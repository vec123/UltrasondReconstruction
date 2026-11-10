"""Builds the conditioning, the velocity field, the network and the flow from the config."""

import dataclasses
from src.models.lddmm.resnet_flow.dynamics.flow import IdentityFlowWrapper, NeuralODEFlow
from src.models.lddmm.resnet_flow.fields.composite.atlas import AtlasField, basis_rank
from src.models.lddmm.resnet_flow.fields.composite.modes import ModalField
from src.models.lddmm.resnet_flow.fields.composite.prior import PriorResidualField
from src.models.lddmm.resnet_flow.fields.composite.symmetry import SymmetricField
from src.models.lddmm.resnet_flow.dynamics.integrators import ForwardEuler, ModifiedEuler
from src.models.lddmm.learning.registry import Registry

from .common import _normalization_domain, code_size
from .prior import load_prior_config


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
            channels=cfg.code.grid_channels,
            domain=_normalization_domain(cfg),
            direct=cfg.code.grid_direct,
        )
    # Broadcast-based conditioning (concat or FiLM)
    return Registry.create("conditioning", cfg.code.conditioning_method, n_z=cfg.code.n_z)


def build_field(cfg, conditioning, out_dim=None):
    """cfg.field -> the VelocityField of kind cfg.field.kind.

    Each field class builds itself from the config (VelocityField.from_cfg), so a new
    kind needs a class and a registration line, not an edit here.

    Args:
        cfg: ExperimentCfg
        conditioning: Conditioning module or None, from build_conditioning
        out_dim: None for a velocity field; 3 * rank for an AtlasField's basis
            (only the config.ATLAS_FIELD_KINDS take it)

    Returns:
        VelocityField instance
    """
    field_class = Registry.resolve("field", cfg.field.kind)
    if out_dim is None:
        return field_class.from_cfg(cfg.field, conditioning)
    return field_class.from_cfg(cfg.field, conditioning, out_dim=out_dim)


def build_network(cfg, conditioning):
    """cfg.field -> the trained network: build_field, anchored at the zero code when
    field.atlas asks, made symmetric when field.symmetry asks.

    The mirror plane is placed later (_place_symmetry) or restored from the checkpoint.
    The atlas field is inside the symmetry, so a symmetric atlas field is still zero
    at the zero code.
    """
    if cfg.field.atlas is not None:
        field = _build_atlas_field(cfg, conditioning)
    else:
        field = build_field(cfg, conditioning)
    if cfg.field.symmetry is not None:
        field = SymmetricField(field, cfg.field.symmetry["mode"])
    return field


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
    if cfg.field.prior is not None:
        # Weights loaded later (_load_prior_field) or restored from the checkpoint.
        prior_cfg = load_prior_config(cfg)
        prior_n_z = code_size(prior_cfg)
        field = PriorResidualField(build_network(prior_cfg, build_conditioning(prior_cfg)),
                                   build_network(cfg, _residual_conditioning(cfg, prior_n_z)),
                                   prior_n_z)
    else:
        field = build_network(cfg, build_conditioning(cfg))
    if cfg.field.modes is not None:
        # Placed later (_place_modes) or restored from the checkpoint's buffers.
        field = ModalField(field, cfg.field.modes["names"], cfg.field.num_steps)
    flow = NeuralODEFlow(
        field=field,
        direct=ForwardEuler(grad_checkpointing=cfg.field.grad_checkpointing),
        inverse=ModifiedEuler(grad_checkpointing=cfg.field.grad_checkpointing),
        num_steps=cfg.field.num_steps
    )
    # Conditionally wrap flow at identity for pose-only training
    if getattr(cfg, 'freeze_flow_at_identity', False):
        flow = IdentityFlowWrapper(flow)
    return flow


def _build_atlas_field(cfg, conditioning):
    """cfg.field.atlas -> an AtlasField around a basis network of 3 * rank outputs.

    linear: the basis network gets no conditioning, so the velocity is linear in the
    code. nonlinear: it gets the run's conditioning, so it reads the code too.

    Raises:
        ValueError: the run has no code, or a linear atlas field is given a
            position-aware code (which only a conditioning can read).
    """
    n_z = code_size(cfg)
    if n_z == 0:
        raise ValueError("field.atlas needs a code (code.kind auto_decoder or encoder): "
                         "the zero code it anchors does not exist without one")
    linear = cfg.field.atlas["mode"] == "linear"
    if linear and cfg.code.position_aware:
        raise ValueError("field.atlas.mode linear reads the code only through A z, so "
                         "code.position_aware has no effect: set it to false, or use mode nonlinear")
    rank = None if cfg.field.atlas["rank"] == "full" else cfg.field.atlas["rank"]
    network = build_field(cfg, None if linear else conditioning, out_dim=3 * basis_rank(n_z, rank))
    return AtlasField(network, n_z, rank)


def _residual_conditioning(cfg, prior_n_z):
    """The conditioning of a PriorResidualField's network: the code entries after the prior's."""
    n_z = code_size(cfg) - prior_n_z
    if n_z == 0:
        return None
    return build_conditioning(dataclasses.replace(cfg, code=dataclasses.replace(cfg.code, n_z=n_z)))
