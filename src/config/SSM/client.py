"""Client that turns YAML config files into typed config objects.

The client is deliberately strict: unknown keys, missing required keys and
values of the wrong type all raise :class:`ConfigError` with the path of the
offending key, so a broken run fails while reading the config instead of
halfway through a fit.

Strings may reference environment variables with ``${VAR}`` or
``${VAR:-fallback}``. ``${PROJECT_ROOT}`` is always available and defaults to
the repository root when the environment does not define it. Every field typed
as :class:`pathlib.Path` is expanded and made absolute against the project root.
"""

from __future__ import annotations

import dataclasses
import os
import re
import types
import typing
from pathlib import Path
from typing import Any, Mapping, Type, TypeVar, Union, get_args, get_origin

import yaml

T = TypeVar("T")

_VAR = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


class ConfigError(ValueError):
    """Raised when a config file cannot be turned into a valid config object."""


def _type_name(hint) -> str:
    return getattr(hint, "__name__", str(hint))


class ConfigClient:
    """Parses YAML config files into the dataclasses declared in ``schema.py``."""

    def __init__(self, project_root=None, env: Mapping[str, str] | None = None, strict: bool = True):
        self.env = dict(os.environ if env is None else env)
        root = project_root or self.env.get("PROJECT_ROOT") or Path(__file__).resolve().parents[2]
        self.project_root = Path(root).expanduser().resolve()
        self.env["PROJECT_ROOT"] = str(self.project_root)
        self.strict = strict

    def load(self, config_path, schema: Type[T]) -> T:
        """Reads ``config_path`` and returns an instance of ``schema``."""
        path = Path(config_path).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{path}: top level must be a mapping, got {type(raw).__name__}")

        return self.from_dict(raw, schema, source=str(path))

    def from_dict(self, data: Mapping[str, Any], schema: Type[T], source: str = "<dict>") -> T:
        """Same as :meth:`load` but for an already parsed mapping."""
        resolved = self._interpolate(data, source)
        return self._build(schema, resolved, schema.__name__, source)

    # -- internals ---------------------------------------------------------

    def _interpolate(self, value, source: str):
        if isinstance(value, str):
            return _VAR.sub(lambda m: self._lookup(m, value, source), value)
        if isinstance(value, Mapping):
            return {k: self._interpolate(v, source) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._interpolate(v, source) for v in value]
        return value

    def _lookup(self, match: re.Match, raw: str, source: str) -> str:
        name = match.group("name")
        if name in self.env:
            return self.env[name]
        default = match.group("default")
        if default is not None:
            return default
        raise ConfigError(
            f"{source}: undefined variable '${{{name}}}' in {raw!r}. "
            f"Set it in the environment/.env or use '${{{name}:-fallback}}'."
        )

    def _build(self, schema, data, trail: str, source: str):
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
            return self._build(hint, value, trail, source)

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
            path = self.project_root / path
        return Path(os.path.normpath(path))


def load_config(config_path, schema: Type[T], **client_kwargs) -> T:
    """Convenience wrapper: ``load_config("configs/pipeline.yaml", PipelineConfig)``."""
    return ConfigClient(**client_kwargs).load(config_path, schema)
