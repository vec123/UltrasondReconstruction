"""Factory for creating Conditioning instances from config."""

from typing import Optional, Dict, Any
from src.resnet_lddmm.conditioning.base import Conditioning, NoConditioning
from src.resnet_lddmm.conditioning.film import ConcatConditioning, FiLMConditioning
from src.resnet_lddmm.conditioning.position_aware import PositionAware


def create_conditioning(config: Optional[Dict[str, Any]]) -> Conditioning:
    """Create a Conditioning instance from configuration.

    Args:
        config: dict with 'method' key and optional method-specific kwargs
                If None or empty, returns NoConditioning()

    Returns:
        Conditioning instance
    """
    if config is None or not config:
        return NoConditioning()

    method = config.get("method", "none").lower()

    if method == "none":
        return NoConditioning()
    elif method == "concat":
        n_z = config.get("n_z", 256)
        return ConcatConditioning(n_z=n_z)
    elif method == "film":
        n_z = config.get("n_z", 256)
        output_dim = config.get("output_dim", 3)
        return FiLMConditioning(n_z=n_z, output_dim=output_dim)
    elif method == "position_aware":
        n_z = config.get("n_z", 256)
        g = config.get("g", 2)
        channels = config.get("channels", 32)
        return PositionAware(n_z=n_z, g=g, channels=channels)
    else:
        raise ValueError(f"Unknown conditioning method: {method}")
