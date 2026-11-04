"""Lazy component Registry.

A thin instance of ``src.shared.core.registry.LazyRegistry`` - see there for what it
actually does (maps ``(category, name)`` to a ``"module.path:ClassName"``
STRING, resolved only on the first ``Registry.create()`` call for that entry,
so nothing is imported until something asks for it by name - the reason
e3nn/torch_geometric stay out of the common import path).
"""

from src.shared.core.registry import LazyRegistry

Registry = LazyRegistry()


# --------------------------------------------------------------------------- #
# Registrations: one line per component that exists TODAY. 
# Each target string is checked against the real file it points to 
# --------------------------------------------------------------------------- #
Registry.register("encoder", "group_encoder",
                   "src.models.lddmm.learning.models.group_encoder:GroupEncoder")

Registry.register("decoder", "folding",
                   "src.models.lddmm.learning.models.folding_decoder:FoldingDecoder")

Registry.register("decoder", "sphere_folding",
                   "src.models.lddmm.learning.models.folding_decoder:SphereFoldingDecoder")

Registry.register("transformer", "se3",
                   "src.models.lddmm.learning.modules.equivariant.transformer:SE3Transformer")
Registry.register("transformer", "equiformer",
                   "src.models.lddmm.learning.modules.equivariant.equiformer:EquiformerTransformer")

Registry.register("graph_builder", "radius",
                   "src.models.lddmm.learning.data.builders:RadiusGraphBuilder")
Registry.register("graph_builder", "knn",
                   "src.models.lddmm.learning.data.builders:KNNGraphBuilder")

Registry.register("latent_head", "gaussian",
                   "src.models.lddmm.learning.models.latent_heads:GaussianLatentHead")
Registry.register("latent_head", "deterministic",
                   "src.models.lddmm.learning.models.latent_heads:DeterministicLatentHead")

Registry.register("readout", "attention",
                   "src.models.lddmm.learning.modules.transformers.perceiver_encoder:PerceiverReducer")
# readout="mean" has no class of its own -- the LatentHead base computes it inline
# as a weighted global_add_pool (latent_heads.py:_reduce), not through a component.
# Nothing to register until that path is extracted into a Strategy class of its own.

