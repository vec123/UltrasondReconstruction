"""Shared YAML <-> dataclass config loading.

Every pipeline's config is a plain ``dataclasses.dataclass``. A YAML file
binds onto it one of three ways:

  * ``from_dict`` (binder.py) - a generic recursive binder for the common
    case: every field is a straight 1:1 mapping, nested dicts bind onto
    nested dataclasses, lists of dicts onto lists of dataclasses.
  * the dataclass's own ``from_dict`` classmethod - for configs whose fields
    need cross-field validation, conditional parsing, or a nested type that
    is only importable lazily (an optional heavy dependency). The generic
    binder calls this instead of recursing when a field's type declares it,
    so the two styles nest inside each other freely.
  * ``ConfigClient`` (client.py) - a strict binder that also checks and
    converts the value types and makes ``Path`` fields absolute.

Either way, unknown keys fail loudly instead of being silently ignored - a
typo'd YAML key should never just run with whatever default nobody chose.

    loading.py    load_yaml, reject_unknown_keys
    variables.py  ${NAME} substitution, cross-stage references, path_variables
    patterns.py   find_paths: the existing paths that fit a pattern, and their variables
    binder.py     from_dict, load_config
    client.py     ConfigClient, ConfigError
"""

from src.shared.core.config.binder import from_dict, load_config
from src.shared.core.config.client import ConfigClient, ConfigError
from src.shared.core.config.loading import load_yaml, reject_unknown_keys
from src.shared.core.config.variables import (
    expand_known,
    interpolate,
    interpolate_paths,
    path_variables,
    substitute_variables,
)
from src.shared.core.config.patterns import find_paths, unknown_variables

__all__ = [
    "ConfigClient",
    "ConfigError",
    "expand_known",
    "find_paths",
    "from_dict",
    "interpolate",
    "interpolate_paths",
    "load_config",
    "load_yaml",
    "path_variables",
    "reject_unknown_keys",
    "substitute_variables",
    "unknown_variables",
]
