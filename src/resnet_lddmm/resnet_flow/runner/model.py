"""Builds the conditioning, the velocity field, the network and the flow from the config."""

import dataclasses
from src.resnet_lddmm.resnet_flow.dynamics.flow import IdentityFlowWrapper, NeuralODEFlow
from src.resnet_lddmm.resnet_flow.fields.composite.modes import ModalField
from src.resnet_lddmm.resnet_flow.fields.composite.prior import PriorResidualField
from src.resnet_lddmm.resnet_flow.fields.composite.symmetry import SymmetricField
from src.resnet_lddmm.resnet_flow.dynamics.integrators import ForwardEuler, ModifiedEuler
from src.resnet_lddmm.learning.registry import Registry

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
            fa=tuple(cfg.field.fa),
            df=tuple(cfg.field.df),
            fourier_n_e=cfg.field.fourier_n_e,
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
    if cfg.field.kind == "time_varying_fourier":
        return Registry.create(
            "field", "time_varying_fourier",
            num_blocks=cfg.field.num_steps,
            mlp=cfg.field.mlp,
            activation=cfg.field.activation,
            conditioning=conditioning,
            n_e=cfg.field.fourier_n_e,
            annealed=cfg.field.fourier_anneal is not None,
            norm=getattr(cfg.field, "norm", None),
        )
    if cfg.field.kind == "stationary_fourier":
        return Registry.create(
            "field", "stationary_fourier",
            mlp=cfg.field.mlp,
            activation=cfg.field.activation,
            conditioning=conditioning,
            n_e=cfg.field.fourier_n_e,
            annealed=cfg.field.fourier_anneal is not None,
            norm=getattr(cfg.field, "norm", None),
        )
    if cfg.field.kind == "time_varying_local":
        local = cfg.field.local
        return Registry.create(
            "field", "time_varying_local",
            num_blocks=cfg.field.num_steps,
            mlp=cfg.field.mlp,
            activation=cfg.field.activation,
            conditioning=conditioning,
            coarse_n_e=cfg.field.fourier_n_e,
            coarse_annealed=cfg.field.fourier_anneal is not None,
            fine_mlp=local["mlp"],
            fine_n_e=local["fourier_n_e"],
            fine_annealed=local["fourier_anneal"] is not None,
        )
    return Registry.create(
        "field", cfg.field.kind,
        num_blocks=cfg.field.num_steps,
        mlp=cfg.field.mlp,
        activation=cfg.field.activation,
        conditioning=conditioning
    )


def build_network(cfg, conditioning):
    """cfg.field -> the trained network: build_field, made symmetric when field.symmetry asks.

    The mirror plane is placed later (_place_symmetry) or restored from the checkpoint.
    """
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


def _residual_conditioning(cfg, prior_n_z):
    """The conditioning of a PriorResidualField's network: the code entries after the prior's."""
    n_z = code_size(cfg) - prior_n_z
    if n_z == 0:
        return None
    return build_conditioning(dataclasses.replace(cfg, code=dataclasses.replace(cfg.code, n_z=n_z)))
