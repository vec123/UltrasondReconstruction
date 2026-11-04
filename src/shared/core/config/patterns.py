"""Path patterns, read backwards: from the files on the disk to the values of their variables.

A path pattern is a path with ``${NAME}`` variables, as every config path is.
``interpolate_paths`` fills it in: the variables of a case give one path.
``find_paths`` does the opposite: the variables with a value are filled in,
each variable without one is a part to find (one folder or file name, not
empty), and every existing path that fits gives one set of values.

    ${CASES_ROOT}/${SUBJECT_ID}/${WEEK}/*.nrrd
    -> [({"SUBJECT_ID": "1", "WEEK": "20 semanas"}, ".../1/20 semanas/1-20s-01.nrrd"), ...]

``*`` is a wildcard inside one name; every other character is literal.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

from src.shared.core.config.variables import VARIABLE_PATTERN, expand_known

WILDCARD = "*"


def _pattern_parts(pattern: str, variables: Dict[str, str]):
    """(glob, regex) of ``pattern`` once its known variables are filled in."""
    text = os.path.normpath(expand_known(pattern, variables))
    glob_parts, regex_parts, seen = [], [], set()
    position = 0
    for match in VARIABLE_PATTERN.finditer(text):
        literal = text[position:match.start()]
        glob_parts.append(WILDCARD.join(glob.escape(piece) for piece in literal.split(WILDCARD)))
        regex_parts.append("[^/]*".join(re.escape(piece) for piece in literal.split(WILDCARD)))
        name = match.group(1)
        glob_parts.append(WILDCARD)
        regex_parts.append(f"(?P={name})" if name in seen else f"(?P<{name}>[^/]+)")
        seen.add(name)
        position = match.end()
    literal = text[position:]
    glob_parts.append(WILDCARD.join(glob.escape(piece) for piece in literal.split(WILDCARD)))
    regex_parts.append("[^/]*".join(re.escape(piece) for piece in literal.split(WILDCARD)))
    return "".join(glob_parts), re.compile("".join(regex_parts))


def unknown_variables(pattern: str, variables: Dict[str, str]) -> List[str]:
    """The variables of ``pattern`` without a value: the parts that find_paths finds."""
    return list(dict.fromkeys(VARIABLE_PATTERN.findall(expand_known(pattern, variables))))


def find_paths(
    pattern: str,
    variables: Dict[str, str],
    parse: Optional[Dict[str, Callable[[str], Dict[str, str]]]] = None,
) -> List[Tuple[Dict[str, str], str]]:
    """[(values, path)] of every existing path that fits ``pattern``, in sort order of the
    path parts (as sorted() sorts Paths: folder ``1`` before ``1-a``).

    Args:
        pattern: a path with ``${NAME}`` variables and ``*`` wildcards
        variables: the variables with a value
        parse: {NAME: function(value) -> {NAME: value}} for a variable that is
            made from others (e.g. CASE_KEY from SUBJECT_ID and WEEK): its
            found value is replaced by the values it is made from. A path
            whose values do not agree with each other, or with ``variables``,
            does not fit.
    """
    glob_pattern, regex = _pattern_parts(pattern, variables)
    known = {name: expand_known(str(value), variables) for name, value in variables.items()}
    found = []
    for path in sorted(glob.glob(glob_pattern), key=lambda p: p.split(os.sep)):
        match = regex.fullmatch(path)
        if match is None:
            continue
        values = match.groupdict()
        for name, parse_value in (parse or {}).items():
            if name in values:
                parts = parse_value(values.pop(name))
                if any({**known, **values}.get(key, value) != value for key, value in parts.items()):
                    break
                values.update(parts)
        else:
            found.append((values, path))
    return found
