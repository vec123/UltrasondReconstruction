"""Pipeline-specific parts of stage configuration.

Loading a YAML file and binding it onto a dataclass is shared with every other
pipeline - see ``src.core.config`` (``load_yaml``, ``from_dict``), where unknown
keys (typos) fail loudly instead of being silently ignored. What lives here is
only what a stage pipeline adds on top: the ``${PROJECT_ROOT}`` / ``${DATA_ROOT}``
variables, and an ``interpolate`` that also understands cross-stage references
and path expressions.

Paths are written relative to the project root using ``${PROJECT_ROOT}``
interpolation, e.g. ``${PROJECT_ROOT}/US_samples/...`` - the same env var
(from ``.env``, via ``python-dotenv``) every script in this repo already uses.
"""

import os
import re

from dotenv import load_dotenv

from src.core.config import substitute_variables

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


def path_variables() -> dict:
    """The ``${NAME}`` variables every config can use: PROJECT_ROOT, and DATA_ROOT
    (the raw ultrasound data folder) when the repo's .env defines it."""
    variables = {"PROJECT_ROOT": project_root()}
    if os.environ.get("DATA_ROOT"):
        variables["DATA_ROOT"] = os.environ["DATA_ROOT"]
    return variables


def _substitute_vars(text: str, variables: dict) -> str:
    return substitute_variables(text, variables, hint="Pass missing ones via --set NAME=VALUE.")


def interpolate(value, variables: dict, resolve_ref=None):
    """Recursively resolves every string leaf of a (possibly nested) config value.

    Extends ``src.core.config.interpolate`` (plain ``${NAME}`` substitution) with
    references and path expressions. Each string is one of:

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
