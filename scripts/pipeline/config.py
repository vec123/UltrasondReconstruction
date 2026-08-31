"""Minimal YAML <-> dataclass binding for pipeline stage configuration.

Every stage's configuration is a plain ``dataclasses.dataclass``. A YAML file
maps 1:1 onto its fields (nested dicts map onto nested dataclasses, and a list
of dicts maps onto a list of them), so the config a stage runs with is always
fully visible in one file, and unknown keys (typos) fail loudly instead of
being silently ignored.

Paths are written relative to the project root using ``${PROJECT_ROOT}``
interpolation, e.g. ``${PROJECT_ROOT}/US_samples/...`` - the same env var
(from ``.env``, via ``python-dotenv``) every script in this repo already uses.
"""

import dataclasses
import os
import re
from typing import Type, TypeVar, get_args, get_origin

import yaml
from dotenv import load_dotenv

T = TypeVar("T")

# Matches a bare cross-stage reference like `nrrd_to_vtp.output_dir` or
# `align_to_reference.aligned_shapes_vtp_dir` - a dotted chain of identifiers,
# nothing else (so it never collides with a `${...}` or `/`-containing path).
REFERENCE_PATTERN = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")


def project_root() -> str:
    load_dotenv()
    root = os.environ.get("PROJECT_ROOT")
    if not root:
        raise RuntimeError("PROJECT_ROOT is not set. Define it in the repo's .env file.")
    return root


def load_yaml(path) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def _substitute_vars(text: str, variables: dict) -> str:
    for name, replacement in variables.items():
        text = text.replace(f"${{{name}}}", replacement)
    if "${" in text:
        raise ValueError(
            f"Unresolved '${{...}}' placeholder in '{text}'. Known variables: {sorted(variables)}. "
            f"Pass missing ones via --set NAME=VALUE."
        )
    return text


def interpolate(value, variables: dict, resolve_ref=None):
    """Recursively resolves every string leaf of a (possibly nested) config value.

    Each string is one of:

    - A bare cross-stage reference, e.g. ``nrrd_to_vtp.output_dir`` (only
      recognized when ``resolve_ref`` is given - see ``pipeline/cli.py``).
      Resolves to that field's actual value, unchanged in type.
    - A path expression: one or more terms joined by `` + ``, each either
      such a reference (stringified) or a literal with ``${NAME}``
      interpolated from ``variables``. Terms are concatenated directly (write
      the separating ``/`` yourself), and the result is normalized with
      ``os.path.normpath`` - so ``${PROJECT_ROOT} + /US_samples`` and
      ``nrrd_to_vtp.output_dir + /aligned`` both work.
    """
    if isinstance(value, str):
        if resolve_ref is not None and REFERENCE_PATTERN.match(value):
            return resolve_ref(value)

        if " + " not in value:
            substituted = _substitute_vars(value, variables)
            return os.path.normpath(substituted) if substituted != value else value

        terms = [t.strip() for t in value.split(" + ")]
        resolved_terms = [
            str(resolve_ref(t)) if resolve_ref is not None and REFERENCE_PATTERN.match(t)
            else _substitute_vars(t, variables)
            for t in terms
        ]
        return os.path.normpath("".join(resolved_terms))
    if isinstance(value, dict):
        return {k: interpolate(v, variables, resolve_ref) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate(v, variables, resolve_ref) for v in value]
    return value


def from_dict(cls: Type[T], data: dict) -> T:
    """Builds a (possibly nested) dataclass instance from a dict.

    Missing keys fall back to the dataclass's own default; keys not present
    on the dataclass raise, so a typo'd YAML field is caught immediately
    instead of silently being ignored.
    """
    data = data or {}
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}

    unknown = set(data) - set(field_types)
    if unknown:
        raise ValueError(f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}")

    kwargs = {}
    for name, ftype in field_types.items():
        if name not in data:
            continue
        value = data[name]
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            value = from_dict(ftype, value)
        elif get_origin(ftype) is list and isinstance(value, list):
            (elem_type,) = get_args(ftype) or (None,)
            if dataclasses.is_dataclass(elem_type):
                value = [from_dict(elem_type, v) if isinstance(v, dict) else v for v in value]
        kwargs[name] = value

    return cls(**kwargs)


def load_config(path, cls: Type[T]) -> T:
    """Loads a stage YAML (with ``${PROJECT_ROOT}`` interpolation applied) into `cls`."""
    data = load_yaml(path)
    data = interpolate(data, {"PROJECT_ROOT": project_root()})
    return from_dict(cls, data)
