"""Name -> implementation lookup, shared by every pluggable part of every pipeline.

Two flavors, because they solve different problems:

    TypedRegistry   Eager: ``register(name)`` stores the class itself.
                    ``build()`` validates kwargs against the class's own
                    constructor signature and type hints, and coerces YAML's
                    stringly-typed numbers (see ``_coerce``). Use this when
                    every component is already cheap to import - no heavy
                    optional dependency involved.

    LazyRegistry    Lazy: ``register(category, name, target)`` stores a
                    ``"module.path:ClassName"`` STRING, resolved only on the
                    first ``create()`` call for that entry. Use this when some
                    components pull in a heavy optional dependency (e3nn,
                    torch_geometric, ...) that most runs never need, so
                    importing the registry itself must stay cheap.

Both fail loudly: an unknown name lists the names that ARE available, and a
stale registration (a renamed class, a moved module) is reported as exactly
that instead of as a generic import error.

Each pipeline owns its own registry instance per pluggable axis (one
``TypedRegistry("loss")``, one ``TypedRegistry("solver")``, ...) or per
lazily-loaded component family (one ``LazyRegistry()`` covering
encoder/decoder/etc.) - nothing here is a shared global table, so two
pipelines' component namespaces never collide.
"""

from __future__ import annotations

import inspect
import typing
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple, Union, get_args, get_origin


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


class TypedRegistry:
    """Eager registry for one kind of component (a loss, a solver, ...)."""

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


class LazyRegistry:
    """Maps (category, name) -> "module:QualName", resolved on first use.

    Storing a string instead of a live import is what keeps registration
    itself free of the target's own dependencies.
    """

    def __init__(self):
        self._entries: Dict[Tuple[str, str], str] = {}

    def register(self, category: str, name: str, target: str) -> None:
        """``target`` is a string like ``"pkg.module:ClassName"``."""
        self._entries[(category, name)] = target

    def resolve(self, category: str, name: str) -> type:
        """Imports and returns the class a registration names. Does not instantiate.

        Both failure paths are re-raised naming the registration: the bare
        importlib error reports only that some module or attribute is
        missing, never which (category, name) pointed at it - which is the
        one fact needed to repair the entry.
        """
        key = (category, name)
        if key not in self._entries:
            raise RegistryError(
                f"no {category!r} registered under name {name!r}; "
                f"available: {self.available(category)}"
            )

        target = self._entries[key]
        module_path, qualname = target.split(":")

        try:
            module = import_module(module_path)
        except ModuleNotFoundError as error:
            # Only when the REGISTERED module is the missing one. A module
            # that imports an absent third-party dependency raises the same
            # class, and calling that a stale registration would send the
            # reader to the wrong file.
            if error.name != module_path:
                raise
            raise ImportError(
                f"registration {category}/{name} -> {target!r}: module "
                f"{module_path!r} does not exist; it was moved or renamed "
                f"without updating the registration"
            ) from error

        try:
            return getattr(module, qualname)
        except AttributeError as error:
            raise ImportError(
                f"registration {category}/{name} -> {target!r}: {module_path!r} "
                f"has no {qualname!r}; the class was renamed without updating "
                f"the registration"
            ) from error

    def create(self, category: str, name: str, **kwargs):
        """Resolve, import, instantiate. Unknown name -> RegistryError listing valid names."""
        return self.resolve(category, name)(**kwargs)

    def available(self, category: str) -> List[str]:
        """Names in a category - powers --help and error messages."""
        return sorted(name for (cat, name) in self._entries if cat == category)

    def entries(self) -> Dict[Tuple[str, str], str]:
        """{(category, name): target string} for every registered component.

        A copy, so a caller iterating the registry cannot mutate it by accident.
        """
        return dict(self._entries)
