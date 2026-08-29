"""Name -> implementation lookup shared by every pluggable part of a fit.

The config names a component and hands over a mapping of keyword arguments;
the registry resolves the name and checks the arguments against the
constructor before building anything. Unknown keys are an error rather than
being ignored, so a typo in the YAML fails immediately instead of silently
producing a fit with default settings.
"""

from __future__ import annotations

import inspect
import typing
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_args, get_origin

T = TypeVar("T")


class RegistryError(ValueError):
    """Raised when a component cannot be resolved or built from its config."""


def _hints(cls) -> Dict[str, Any]:
    """Resolved type hints for a component's constructor arguments."""
    for target in (cls, getattr(cls, "__init__", None)):
        if target is None:
            continue
        try:
            hints = typing.get_type_hints(target)
        except Exception:
            continue
        if hints:
            return hints
    return {}


def _coerce(value, hint):
    """Turns a numeric string into the number its declared type asks for.

    YAML 1.1 only recognises a float when it has *both* a decimal point and a
    signed exponent, so ``1.0e-9`` arrives as a float while ``1e-9``, ``1e9``
    and ``1.0e9`` all arrive as strings. Rather than make every config author
    remember that rule, convert against the type the component declares.
    """
    if not isinstance(value, str):
        return value

    targets = get_args(hint) if get_origin(hint) is Union else (hint,)
    for target in targets:
        if target is float:
            try:
                return float(value)
            except ValueError:
                pass
        elif target is int:
            try:
                return int(value)
            except ValueError:
                pass
    return value


class Registry:
    """Registry for one kind of component (a loss, a solver, ...)."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, type] = {}

    def register(self, name: str):
        """Class decorator that files the class under ``name``."""

        def decorator(cls):
            if name in self._items:
                raise RegistryError(f"{self.kind} '{name}' is already registered")
            cls.name = name
            self._items[name] = cls
            return cls

        return decorator

    def names(self) -> List[str]:
        return sorted(self._items)

    def get(self, name: str) -> type:
        cls = self._items.get(name)
        if cls is None:
            raise RegistryError(f"unknown {self.kind} '{name}'. Available: {self.names()}")
        return cls

    def build(self, name: str, kwargs: Optional[Dict[str, Any]] = None):
        cls = self.get(name)
        kwargs = dict(kwargs or {})

        accepted = [p for p in inspect.signature(cls).parameters]
        unknown = sorted(set(kwargs) - set(accepted))
        if unknown:
            raise RegistryError(
                f"unknown kwarg(s) {unknown} for {self.kind} '{name}'. Accepted: {accepted}"
            )

        hints = _hints(cls)
        kwargs = {key: _coerce(value, hints.get(key)) for key, value in kwargs.items()}

        try:
            return cls(**kwargs)
        except (TypeError, ValueError) as exc:
            raise RegistryError(f"could not build {self.kind} '{name}': {exc}") from exc

    def build_spec(self, spec):
        """Builds from anything carrying ``.name`` and ``.kwargs`` (a config spec)."""
        return self.build(spec.name, getattr(spec, "kwargs", None))
