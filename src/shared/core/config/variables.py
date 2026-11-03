"""``${NAME}`` variables in config strings.

``interpolate`` substitutes plain ``${NAME}`` variables. ``interpolate_paths``,
which the stage pipelines (scripts/pipeline/cli.py) use, also understands
cross-stage references and path expressions.

Paths are written relative to the project root using ``${PROJECT_ROOT}``
interpolation, e.g. ``${PROJECT_ROOT}/US_samples/...`` - the same env var
(from ``.env``, via ``python-dotenv``) every script in this repo already uses.
"""

from __future__ import annotations

import os
import re
from typing import Dict

from src.shared.core.paths import get_project_root

# Matches a bare cross-stage reference like `nrrd_to_vtp.output_dir` or
# `align_to_reference.aligned_shapes_vtp_dir` - a dotted chain of identifiers,
# nothing else (so it never collides with a `${...}` or `/`-containing path).
REFERENCE_PATTERN = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")


def substitute_variables(text: str, variables: Dict[str, str], hint: str = "") -> str:
    """Substitutes ``${NAME}`` in one string.

    Args:
        text: the string to substitute into
        variables: replacement text for each ``${NAME}``
        hint: appended to the error message, e.g. how to supply a missing variable

    Raises:
        ValueError: if a ``${...}`` placeholder has no matching variable
    """
    for name, replacement in variables.items():
        text = text.replace(f"${{{name}}}", replacement)
    if "${" in text:
        raise ValueError(
            f"Unresolved '${{...}}' placeholder in '{text}'. "
            f"Known variables: {sorted(variables)}." + (f" {hint}" if hint else "")
        )
    return text


def interpolate(value, variables: Dict[str, str]):
    """Recursively substitutes ``${NAME}`` in every string leaf of ``value``.

    Args:
        value: a (possibly nested) dict/list/str/other config value
        variables: replacement text for each ``${NAME}``

    Raises:
        ValueError: if a ``${...}`` placeholder has no matching variable
    """
    if isinstance(value, str):
        return substitute_variables(value, variables)
    if isinstance(value, dict):
        return {k: interpolate(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate(v, variables) for v in value]
    return value


def path_variables() -> dict:
    """The ``${NAME}`` variables every config can use: PROJECT_ROOT, and DATA_ROOT
    (the raw ultrasound data folder) when the repo's .env defines it."""
    variables = {"PROJECT_ROOT": str(get_project_root())}
    if os.environ.get("DATA_ROOT"):
        variables["DATA_ROOT"] = os.environ["DATA_ROOT"]
    return variables


def _substitute_vars(text: str, variables: dict) -> str:
    return substitute_variables(text, variables, hint="Pass missing ones via --set NAME=VALUE.")


def interpolate_paths(value, variables: dict, resolve_ref=None):
    """Recursively resolves every string leaf of a (possibly nested) config value.

    Extends ``interpolate`` (plain ``${NAME}`` substitution) with references
    and path expressions. Each string is one of:

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
        return {k: interpolate_paths(v, variables, resolve_ref) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_paths(v, variables, resolve_ref) for v in value]
    return value
