"""The cases of a ``targets:`` block: each case is a set of variable values.

    targets:
      match: ${CASES_ROOT}/${SUBJECT_ID}/${WEEK}/${REFERENCE_SCAN}.nrrd
      first: [REFERENCE_SCAN]
      exclude: [{SUBJECT_ID: 7}]

``match:`` finds the cases on the disk (src/shared/core/config/patterns.py):
each variable without a value is a part to find, and each set of values
found is one case. A variable with a value selects. A list of patterns joins
their cases; a variable that one pattern does not have is empty in its cases
(``WEEK: ""`` for a dataset without week folders). ``first:`` names
variables that do not make cases of their own: a case takes their first
value in sort order (the first scan of a session). ``cases:`` lists the
cases by hand instead. ``only:`` and ``exclude:`` select: a case fits an
entry when it has all of the entry's values.
"""

from dataclasses import dataclass
from typing import Dict

from src.shared.core.config import find_paths, unknown_variables
from src.shared.core.pipeline.cases import CASE_KEY, SUBJECT_ID, WEEK, parse_case_key, with_case_key

KEYS = ("match", "first", "cases", "only", "exclude")


@dataclass(frozen=True)
class Case:
    values: Dict[str, str]  # the variables of this case
    n_files: int = 0  # the paths that ``match:`` found for it (all of them, with ``first:``)

    @property
    def name(self):
        """CASE_KEY (``1_20_semanas``), or the values joined when there is no SUBJECT_ID;
        "case" for the one case of a pattern whose variables all have a value."""
        if SUBJECT_ID in self.values:
            return with_case_key(self.values)[CASE_KEY]
        return "_".join(value.replace(" ", "_") for value in self.values.values()) or "case"


def _values(entry, key, source):
    if not isinstance(entry, dict):
        raise ValueError(f"'targets: {key}:' in '{source}' must list mappings of NAME: value, got {entry!r}.")
    return {str(name): str(value) for name, value in entry.items()}


def _matched(patterns, first, variables, source):
    patterns = [patterns] if isinstance(patterns, str) else list(patterns)
    found = [(values, path) for pattern in patterns
             for values, path in find_paths(pattern, variables, parse={CASE_KEY: parse_case_key})]
    # CASE_KEY gives the variables it is made from.
    names = list(dict.fromkeys(
        part for pattern in patterns for name in unknown_variables(pattern, variables)
        for part in ((SUBJECT_ID, WEEK) if name == CASE_KEY else (name,))))
    # A first: variable with a value already (e.g. given by the caller) has nothing to find.
    missing = sorted(set(first) - set(names) - set(variables))
    if missing:
        raise ValueError(f"'targets: first:' in '{source}' names {missing}, which no 'match:' pattern finds.")
    groups = {}  # the values that make the case -> [its values, its first path, its number of paths]
    for values, path in found:
        values = {name: values.get(name, "") for name in names}
        key = tuple(value for name, value in values.items() if name not in first)
        if key in groups:
            groups[key][2] += 1
        else:
            groups[key] = [values, path, 1]
    ordered = sorted(groups.values(), key=lambda group: group[1].split("/"))
    return [Case(values, n_files) for values, _, n_files in ordered]


def find_case_paths(pattern, source):
    """{case name: (its values, path)} of every existing path that fits ``pattern``, in path
    order. The variables of the pattern (without a value) are the case variables; a stage
    with a pattern field (StageSpec.patterns) finds its cases with this."""
    paths = {}
    for values, path in find_paths(pattern, {}, parse={CASE_KEY: parse_case_key}):
        name = Case(values).name
        if name in paths:
            raise ValueError(f"{source}: case '{name}' fits two paths, '{paths[name][1]}' and '{path}'.")
        paths[name] = (values, path)
    return paths


def find_cases(targets, variables, source):
    """The cases of a ``targets:`` block, ``variables`` filled in (see the module docstring)."""
    if not isinstance(targets, dict):
        raise ValueError(f"'targets:' in '{source}' must be a mapping, got {type(targets).__name__}.")
    unknown = sorted(set(targets) - set(KEYS))
    if unknown:
        raise ValueError(f"'targets:' in '{source}' has unknown key(s) {unknown}; the keys are {list(KEYS)}.")
    if ("match" in targets) == ("cases" in targets):
        raise ValueError(f"'targets:' in '{source}' needs exactly one of 'match:' (find the cases) "
                         f"and 'cases:' (list them).")

    if "match" in targets:
        cases = _matched(targets["match"], [str(name) for name in targets.get("first", [])], variables, source)
        if not cases:
            raise ValueError(f"No path fits 'targets: match: {targets['match']}' in '{source}'.")
    else:
        if "first" in targets:
            raise ValueError(f"'targets: first:' in '{source}' works only with 'match:'.")
        cases = [Case(_values(entry, "cases", source)) for entry in targets["cases"]]

    only = [_values(entry, "only", source) for entry in targets.get("only", [])]
    exclude = [_values(entry, "exclude", source) for entry in targets.get("exclude", [])]

    def fits(case, entry):
        return all(case.values.get(name) == value for name, value in entry.items())

    return [case for case in cases
            if (not only or any(fits(case, entry) for entry in only))
            and not any(fits(case, entry) for entry in exclude)]
