"""Lazy component Registry.

Maps ``(category, name)`` 
-> a ``"module.path:ClassName"`` STRING, resolved
(imported) only on the first ``Registry.create()`` call for that entry.

Storing strings instead of live imports means: 
nothing is imported until something asks for it by name.
"""

from importlib import import_module


class Registry:
    """Maps (category, name) -> "module:QualName",
      resolved on first use."""

    _entries: dict = {}

    @classmethod
    def register(cls, category: str, name: str, target: str) -> None:
        """`target` is a STRING like "src.learning.models.folding_decoder:FoldingDecoder".
        Storing a string (not the class) is what keeps the import lazy."""
        cls._entries[(category, name)] = target

    @classmethod
    def resolve(cls, category: str, name: str) -> type:
        """Import and return the CLASS a registration names. No instantiation.

        Both failure paths are re-raised naming the registration. The bare
        importlib error reports only that some module or attribute is missing and
        never which (category, name) pointed at it -- which is the single fact
        needed to repair the entry, and the reason stringly-typed targets rot
        quietly in the first place.
        """
        key = (category, name)
        if key not in cls._entries:
            raise ValueError(
                f"no {category!r} registered under name {name!r}; "
                f"available: {cls.available(category)}"
            )

        target = cls._entries[key]
        module_path, qualname = target.split(":")

        try:
            module = import_module(module_path)
        except ModuleNotFoundError as error:
            # Only when the REGISTERED module is the missing one. A module that
            # imports an absent third-party dependency (e3nn, torch_geometric)
            # raises the same class, and calling that a stale registration would
            # send the reader to the wrong file.
            if error.name != module_path:
                raise
            raise ImportError(
                f"registration {category}/{name} -> {target!r}: module "
                f"{module_path!r} does not exist; it was moved or renamed without "
                f"updating the registration"
            ) from error

        try:
            return getattr(module, qualname)
        except AttributeError as error:
            raise ImportError(
                f"registration {category}/{name} -> {target!r}: {module_path!r} has "
                f"no {qualname!r}; the class was renamed without updating the "
                f"registration"
            ) from error

    @classmethod
    def create(cls, category: str, name: str, **kwargs):
        """Resolve, import, instantiate. Unknown name -> ValueError listing valid names."""
        return cls.resolve(category, name)(**kwargs)

    @classmethod
    def available(cls, category: str) -> list:
        """Names in a category -- powers --help and error messages."""
        return sorted(name for (cat, name) in cls._entries if cat == category)

    @classmethod
    def entries(cls) -> dict:
        """{(category, name): target string} for EVERY registered component.

        A copy, so a caller iterating the registry cannot mutate it by accident.
        The read-only view that rot guards want: `available` answers per category
        and would need the category list to be known up front, which is the one
        thing a "check everything registered" test cannot assume.
        """
        return dict(cls._entries)


# --------------------------------------------------------------------------- #
# Registrations: one line per component that exists TODAY. 
# Each target string is checked against the real file it points to 
# --------------------------------------------------------------------------- #
Registry.register("encoder", "group_encoder",
                   "src.learning.models.group_encoder:GroupEncoder")

Registry.register("decoder", "folding",
                   "src.learning.models.folding_decoder:FoldingDecoder")

Registry.register("decoder", "sphere_folding",
                   "src.learning.models.folding_decoder:SphereFoldingDecoder")

Registry.register("transformer", "se3",
                   "src.learning.modules.equivariant.transformer:SE3Transformer")
Registry.register("transformer", "equiformer",
                   "src.learning.modules.equivariant.equiformer:EquiformerTransformer")

Registry.register("graph_builder", "radius",
                   "src.learning.data.builders:RadiusGraphBuilder")
Registry.register("graph_builder", "knn",
                   "src.learning.data.builders:KNNGraphBuilder")

Registry.register("latent_head", "gaussian",
                   "src.learning.models.latent_heads:GaussianLatentHead")
Registry.register("latent_head", "deterministic",
                   "src.learning.models.latent_heads:DeterministicLatentHead")

Registry.register("readout", "attention",
                   "src.learning.modules.transformers.perceiver_encoder:PerceiverReducer")
# readout="mean" has no class of its own -- the LatentHead base computes it inline
# as a weighted global_add_pool (latent_heads.py:_reduce), not through a component.
# Nothing to register until that path is extracted into a Strategy class of its own.

