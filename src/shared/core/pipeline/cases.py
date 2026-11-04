"""The variables of a case, and the case key that the code makes from them.

A case is one subject at one gestational age: SUBJECT_ID and WEEK (``"1"``,
``"20 semanas"``). A dataset without week folders has WEEK ``""``. The paths
use these variables, and patterns of them (see the sources file). CASE_KEY is
the one spelling that a pattern cannot make: one file or folder name, with
``_`` for the spaces of the week, as the cohort folders and the batch logs use
it (``1_20_semanas``, ``FLA242``).
"""

import re

from src.shared.core.config import expand_known

SUBJECT_ID = "SUBJECT_ID"
WEEK = "WEEK"
CASE_KEY = "CASE_KEY"

_CASE_KEY_PATTERN = re.compile(r"(?P<subject>.+?)(?:_(?P<week>\d+(?:\.\d+)?_semanas))?")


def case_key(subject_id, week=""):
    """``"1", "20 semanas"`` -> ``"1_20_semanas"``; without a week, the subject alone."""
    return "_".join(filter(None, (str(subject_id), str(week or "").replace(" ", "_"))))


def parse_case_key(key):
    """The inverse of case_key: ``"1_20_semanas"`` -> {SUBJECT_ID: "1", WEEK: "20 semanas"}.

    A key that does not end in ``_<number>_semanas`` is a subject alone (WEEK ``""``).
    """
    match = _CASE_KEY_PATTERN.fullmatch(key)
    week = match.group("week")
    return {SUBJECT_ID: match.group("subject"), WEEK: week.replace("_", " ") if week else ""}


def with_case_key(variables):
    """``variables`` with CASE_KEY made from SUBJECT_ID and WEEK, when it has SUBJECT_ID and no CASE_KEY."""
    if SUBJECT_ID not in variables or CASE_KEY in variables:
        return variables
    subject_id, week = (expand_known(str(variables.get(name, "")), variables) for name in (SUBJECT_ID, WEEK))
    return {**variables, CASE_KEY: case_key(subject_id, week)}
