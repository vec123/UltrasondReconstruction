"""Shared, domain-agnostic infrastructure used by every pipeline
(preprocessing, SSM fitting, resnet_lddmm training).

Nothing in here knows about meshes, landmarks, or any specific model - it is
just the plumbing every pipeline was independently reinventing: loading a
YAML file into a typed dataclass, resolving the project root, and looking up
a pluggable component by name. See config.py, paths.py and registry.py for
the pieces themselves.
"""

from src.core.config import from_dict, interpolate, load_config, load_yaml, reject_unknown_keys
from src.core.paths import get_project_root
from src.core.registry import LazyRegistry, RegistryError, TypedRegistry

__all__ = [
    "from_dict",
    "interpolate",
    "load_config",
    "load_yaml",
    "reject_unknown_keys",
    "get_project_root",
    "LazyRegistry",
    "RegistryError",
    "TypedRegistry",
]
