"""Client that turns YAML config files into typed config objects.

The client reads the file, substitutes the variables and binds the result
with ``binder.from_dict``, which is strict (see there).

Strings may reference environment variables with ``${VAR}`` or
``${VAR:-fallback}``. ``${PROJECT_ROOT}`` is always available and defaults to
the repository root when the environment does not define it. Every field typed
as :class:`pathlib.Path` is expanded and made absolute against the project root.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping, Type, TypeVar

import yaml

from src.shared.core.config.binder import ConfigError, from_dict
from src.shared.core.paths import get_project_root

T = TypeVar("T")

_VAR = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


class ConfigClient:
    """Parses YAML config files into the dataclasses declared in ``schema.py``."""

    def __init__(self, project_root=None, env: Mapping[str, str] | None = None, strict: bool = True):
        self.env = dict(os.environ if env is None else env)
        root = project_root or self.env.get("PROJECT_ROOT") or get_project_root()
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
        return from_dict(schema, resolved, project_root=self.project_root, source=source, strict=self.strict)

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
