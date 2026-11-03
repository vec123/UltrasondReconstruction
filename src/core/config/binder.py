"""The YAML -> dataclass binder that every config goes through.

``from_dict`` binds a parsed mapping onto a dataclass: nested dicts bind onto
nested dataclasses, lists and dicts element by element. It is strict:
unknown keys, missing required keys and values of the wrong type all raise
:class:`ConfigError` with the path of the offending key, so a broken run
fails while reading the config instead of halfway through a fit. Numbers
written as strings ('1e-9') become numbers, and every field typed as
:class:`pathlib.Path` is expanded and made absolute against the project root.
A field type with its own ``from_dict`` classmethod builds itself.
"""

from __future__ import annotations

import dataclasses
import os
import types
import typing
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type, TypeVar, Union, get_args, get_origin

from src.core.config.loading import load_yaml
from src.core.config.variables import interpolate
from src.core.paths import get_project_root

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a config file cannot be turned into a valid config object."""


def _type_name(hint) -> str:
    return getattr(hint, "__name__", str(hint))


def _type_name(hint) -> str:
    return getattr(hint, "__name__", str(hint))


def from_dict(cls: Type[T], data: Optional[Mapping[str, Any]], project_root=None,
              source: str = "<dict>", strict: bool = True) -> T:
    """Builds a (possibly nested) dataclass instance from a dict.

    Missing keys fall back to the dataclass's own default.

    Args:
        cls: the dataclass to bind into
        data: the parsed mapping; None is the same as {}
        project_root: base of relative ``Path`` fields (default: get_project_root())
        source: where ``data`` came from, for the error messages
        strict: False ignores unknown keys instead of raising

    Raises:
        ConfigError: for an unknown key, a missing required key or a wrong type
    """
    return _Binder(project_root, strict).build(cls, data or {}, cls.__name__, source)


class _Binder:
    def __init__(self, project_root, strict: bool):
        self.project_root = project_root
        self.strict = strict

    def build(self, schema, data, trail: str, source: str):
        if not dataclasses.is_dataclass(schema):
            raise ConfigError(f"{source}: '{trail}' expects a dataclass schema, got {_type_name(schema)}")
        if not isinstance(data, Mapping):
            raise ConfigError(f"{source}: '{trail}' expects a mapping, got {type(data).__name__}")

        hints = typing.get_type_hints(schema)
        fields = {f.name: f for f in dataclasses.fields(schema) if f.init}

        unknown = sorted(k for k in data if k not in fields)
        if unknown and self.strict:
            raise ConfigError(
                f"{source}: unknown key(s) {unknown} under '{trail}'. Valid keys: {sorted(fields)}"
            )

        kwargs = {}
        for name, field in fields.items():
            if name in data:
                kwargs[name] = self._coerce(hints[name], data[name], f"{trail}.{name}", source)
            elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
                raise ConfigError(
                    f"{source}: missing required key '{trail}.{name}' ({_type_name(hints[name])})"
                )

        try:
            return schema(**kwargs)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{source}: could not build '{trail}': {exc}") from exc

    def _coerce(self, hint, value, trail: str, source: str):
        if hint is Any or hint is None:
            return value

        # A type with its own from_dict (cross-field validation, aliases)
        # builds itself; this keeps hand-written binders composable.
        if isinstance(hint, type) and callable(getattr(hint, "from_dict", None)) and isinstance(value, Mapping):
            return hint.from_dict(value)

        origin = get_origin(hint)

        if origin is Union or origin is getattr(types, "UnionType", ()):
            args = get_args(hint)
            if value is None and type(None) in args:
                return None
            for arg in args:
                if arg is type(None):
                    continue
                try:
                    return self._coerce(arg, value, trail, source)
                except ConfigError:
                    continue
            raise ConfigError(f"{source}: '{trail}' does not match {hint} (got {value!r})")

        if dataclasses.is_dataclass(hint):
            return self.build(hint, value, trail, source)

        if hint is Path:
            return self._resolve_path(value, trail, source)

        if origin is list:
            if not isinstance(value, (list, tuple)):
                raise ConfigError(f"{source}: '{trail}' expects a list, got {type(value).__name__}")
            args = get_args(hint) or (Any,)
            return [self._coerce(args[0], v, f"{trail}[{i}]", source) for i, v in enumerate(value)]

        if origin is dict:
            if not isinstance(value, Mapping):
                raise ConfigError(f"{source}: '{trail}' expects a mapping, got {type(value).__name__}")
            args = get_args(hint) or (Any, Any)
            return {
                str(k): self._coerce(args[1], v, f"{trail}.{k}", source) for k, v in value.items()
            }

        if hint is bool:
            if isinstance(value, bool):
                return value
            raise ConfigError(f"{source}: '{trail}' expects true/false, got {value!r}")

        # YAML 1.1 only recognises a float when it has *both* a decimal point and
        # a signed exponent: '1.0e-9' parses as a float, while '1e-9', '1e9' and
        # '1.0e9' all arrive as strings. Numeric strings are therefore accepted
        # wherever a number is declared, rather than making the config author
        # remember that rule.
        if hint is int:
            if isinstance(value, bool):
                raise ConfigError(f"{source}: '{trail}' expects an int, got a bool")
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    pass
            raise ConfigError(f"{source}: '{trail}' expects an int, got {value!r}")

        if hint is float:
            if isinstance(value, bool):
                raise ConfigError(f"{source}: '{trail}' expects a float, got a bool")
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    pass
            raise ConfigError(f"{source}: '{trail}' expects a float, got {value!r}")

        if hint is str:
            if isinstance(value, str):
                return value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(value)
            raise ConfigError(f"{source}: '{trail}' expects a string, got {value!r}")

        return value

    def _resolve_path(self, value, trail: str, source: str) -> Path:
        if isinstance(value, Path):
            path = value
        elif isinstance(value, str):
            path = Path(value)
        else:
            raise ConfigError(f"{source}: '{trail}' expects a path string, got {value!r}")

        path = path.expanduser()
        if not path.is_absolute():
            path = Path(self.project_root or get_project_root()) / path
        return Path(os.path.normpath(path))



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
    return from_dict(cls, data, source=str(path))
