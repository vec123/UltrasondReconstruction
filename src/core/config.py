"""Shared YAML <-> dataclass config loading.

Every pipeline's config is a plain ``dataclasses.dataclass``. A YAML file
binds onto it one of two ways:

  * ``from_dict`` below - a generic recursive binder for the common case:
    every field is a straight 1:1 mapping, nested dicts bind onto nested
    dataclasses, lists of dicts onto lists of dataclasses.
  * the dataclass's own ``from_dict`` classmethod - for configs whose fields
    need cross-field validation, conditional parsing, or a nested type that
    is only importable lazily (an optional heavy dependency). The generic
    binder calls this instead of recursing when a field's type declares it,
    so the two styles nest inside each other freely.

Either way, unknown keys fail loudly via ``reject_unknown_keys`` instead of
being silently ignored - a typo'd YAML key should never just run with
whatever default nobody chose.
"""

from __future__ import annotations

import dataclasses
import typing
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, get_args, get_origin

import yaml

T = TypeVar("T")


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


def interpolate(value, variables: Dict[str, str]):
    """Recursively substitutes ``${NAME}`` in every string leaf of ``value``.

    Args:
        value: a (possibly nested) dict/list/str/other config value
        variables: replacement text for each ``${NAME}``

    Raises:
        ValueError: if a ``${...}`` placeholder has no matching variable
    """
    if isinstance(value, str):
        for name, replacement in variables.items():
            value = value.replace(f"${{{name}}}", replacement)
        if "${" in value:
            raise ValueError(
                f"Unresolved '${{...}}' placeholder in '{value}'. "
                f"Known variables: {sorted(variables)}."
            )
        return value
    if isinstance(value, dict):
        return {k: interpolate(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate(v, variables) for v in value]
    return value


def from_dict(cls: Type[T], data: Optional[Dict[str, Any]]) -> T:
    """Builds a (possibly nested) dataclass instance from a dict.

    Missing keys fall back to the dataclass's own default. A field whose type
    declares a ``from_dict`` classmethod is built through that instead of
    recursing generically, so a nested config needing custom validation still
    composes with one built by this function.

    Field types are resolved via ``typing.get_type_hints``, so every field's
    type must be resolvable at call time - a dataclass that intentionally
    imports one of its field types lazily (to keep an optional heavy
    dependency out of the common path) should write its own ``from_dict``
    instead of going through this one.

    Raises:
        ValueError: for an unknown key (see ``reject_unknown_keys``)
    """
    data = data or {}
    hints = typing.get_type_hints(cls)
    field_names = {f.name for f in dataclasses.fields(cls)}
    reject_unknown_keys(data, field_names, what=cls.__name__)

    kwargs = {}
    for name in field_names:
        if name not in data:
            continue
        value = data[name]
        ftype = hints.get(name)
        if isinstance(value, dict) and hasattr(ftype, "from_dict"):
            value = ftype.from_dict(value)
        elif dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            value = from_dict(ftype, value)
        elif get_origin(ftype) is list and isinstance(value, list):
            (elem_type,) = get_args(ftype) or (None,)
            if isinstance(elem_type, type) and hasattr(elem_type, "from_dict"):
                value = [elem_type.from_dict(v) if isinstance(v, dict) else v for v in value]
            elif dataclasses.is_dataclass(elem_type):
                value = [from_dict(elem_type, v) if isinstance(v, dict) else v for v in value]
        kwargs[name] = value

    return cls(**kwargs)


def load_config(path, cls: Type[T], variables: Optional[Dict[str, str]] = None) -> T:
    """Loads a YAML file into ``cls``.

    Args:
        path: path to the YAML file
        cls: the dataclass to bind into
        variables: if given, every string field is ``${NAME}``-interpolated
            against these before binding, e.g.
            ``{"PROJECT_ROOT": str(get_project_root())}``. Omit to skip
            interpolation entirely.
    """
    data = load_yaml(path)
    if variables is not None:
        data = interpolate(data, variables)
    return from_dict(cls, data)
