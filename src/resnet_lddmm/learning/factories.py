"""Config -> constructed object factories.

The ONLY place config fields are read and translated into the
plain arguments a component's own constructor already understands. 
This is what enforces the dependency rule (only downwards (i.e L_n imports L_m: m<n)):

    L2  assembly     config schemas, registry, factories   <- this file
    L1  components   models/, modules/, losses/, data/     <- takes plain arguments
    L0  primitives   geometry/, graphs/, vtk/

Components never import config;
this module (and the Registry it drives) 
is the one place that does. 
The litmus test for a component: "could I use this class in
a different project without copying the config package?"
"""

from src.learning.registry import Registry


def build_encoder(cfg):
    """EncoderConfig -> encoder module. The ONLY place encoder config fields are read.

    Matched against GroupEncoder's real constructor (group_encoder.py:16-25) --
    every field is accounted for, including ``area_pool`` (the deleted T0 factory
    dropped it), ``spatial_sh_lmax`` / ``interaction_sh_lmax`` per layer (T7's
    multi-layer update; the latter used to be hardcoded inside GroupEncoder and
    had no effect no matter what config said), and ``latent_mode`` (T9's
    LatentHead strategy: "gaussian" | "deterministic").
    """
    layers_cfg = [
        {
            "in_irreps": layer.in_irreps,
            "target_irreps": layer.target_irreps,
            "spatial_sh_lmax": layer.spatial_sh_lmax,
            "interaction_sh_lmax": layer.interaction_sh_lmax,
        }
        for layer in cfg.layers
    ]
    return Registry.create(
        "encoder", cfg.encoder_type,
        layers_cfg=layers_cfg,
        latent_dim=cfg.latent_dim,
        output_irreps=cfg.output_irreps,
        readout=cfg.readout,
        readout_heads=cfg.readout_heads,
        supernode_sh_lmax=cfg.supernode_sh_lmax,
        transformer_type=cfg.transformer_type,
        transformer_cfg=cfg.transformer_cfg,
        area_pool=cfg.area_pool,
        latent_mode=cfg.latent_mode,
        pose_mode=cfg.pose_mode,
        verbose=cfg.verbose,
    )


def build_decoder(cfg):
    """DecoderConfig -> decoder module.
    Matched against FoldingDecoder / SphereFoldingDecoder's shared constructor
    shape (folding_decoder.py).
    """
    return Registry.create(
        "decoder", cfg.decoder_type,
        num_samples=cfg.num_samples,
        latent_dim=cfg.latent_dim,
        n_freqs=cfg.n_freqs,
        verbose=cfg.verbose,
    )


def build_from_config(cfg):
    """ExperimentConfig -> (encoder, decoder). 
    (Re-) Validates the pairing before returning.
    """
    encoder = build_encoder(cfg.encoder)
    decoder = build_decoder(cfg.decoder)
    if encoder.n_tokens != decoder.expects_tokens:
        raise ValueError(
            f"encoder.encoder_type={cfg.encoder.encoder_type!r} emits "
            f"{encoder.n_tokens} token(s) per shape but decoder.decoder_type="
            f"{cfg.decoder.decoder_type!r} expects {decoder.expects_tokens}."
        )
    return encoder, decoder