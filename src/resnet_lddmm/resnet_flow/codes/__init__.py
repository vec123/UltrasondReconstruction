"""Shape code modules for per-shape learned embeddings."""

from src.resnet_lddmm.resnet_flow.codes.base import ShapeCode
from src.resnet_lddmm.resnet_flow.codes.none import NoCode
from src.resnet_lddmm.resnet_flow.codes.auto_decoder import AutoDecoderCodes

# Code kinds that PREDICT their own pose instead of holding an injected one, and so
# satisfy `use_pose` without a `code.pose` block. Every other kind needs one --
# naming none would apply no pose at all, which surfaces as a loss with no grad_fn
# rather than as a configuration error, so the runner checks this up front.
SELF_POSING_KINDS = frozenset({"encoder"})

__all__ = [
    "ShapeCode",
    "NoCode",
    "AutoDecoderCodes",
    "SELF_POSING_KINDS",
]
