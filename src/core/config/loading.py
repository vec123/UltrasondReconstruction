"""Reading a YAML config file, and the unknown-key check every binder uses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path) -> Dict[str, Any]:
    """Reads a YAML file that must contain a mapping (or nothing) at the top level.

    Args:
        path: path to a .yaml file

    Returns:
        The parsed dict, or {} for an empty file.

    Raises:
        FileNotFoundError: if the file does not exist
        ValueError: if the document is neither empty nor a mapping
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(data).__name__}")
    return data


def reject_unknown_keys(data: Dict[str, Any], allowed, what: str = "config") -> None:
    """Fails on any key in ``data`` that is not in ``allowed``.

    Args:
        data: the parsed mapping
        allowed: iterable of accepted key names
        what: name used in the error message

    Raises:
        ValueError: naming the offending keys
    """
    unknown = set(data) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {what} key(s): {', '.join(sorted(unknown))}")
