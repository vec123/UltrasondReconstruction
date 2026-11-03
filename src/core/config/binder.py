"""The generic YAML -> dataclass binder.

``from_dict`` binds a parsed mapping onto a dataclass: every field is a
straight 1:1 mapping, nested dicts bind onto nested dataclasses, lists of
dicts onto lists of dataclasses. A field type with its own ``from_dict``
classmethod is built through that instead. Unknown keys fail.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Dict, Optional, Type, TypeVar, get_args, get_origin

from src.core.config.loading import load_yaml, reject_unknown_keys
from src.core.config.variables import interpolate

T = TypeVar("T")


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
