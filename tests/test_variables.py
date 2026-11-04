"""${NAME} variables: built from each other, and path patterns read backwards.

    python -m pytest tests/test_variables.py
    python tests/test_variables.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared.core.config import find_paths, substitute_variables  # noqa: E402
from src.shared.core.pipeline.cases import CASE_KEY, case_key, parse_case_key, with_case_key  # noqa: E402


def _raises(function, *args):
    try:
        function(*args)
    except ValueError as error:
        return str(error)
    raise AssertionError(f"{function.__name__}{args} did not raise")


def test_variables_built_from_others_in_any_order():
    variables = {"OUT": "${ROOT}/${SESSION}", "SESSION": "${SUBJECT_ID}/${WEEK}",
                 "ROOT": "/data", "SUBJECT_ID": "1", "WEEK": "20 semanas"}
    assert substitute_variables("${OUT}/fit", variables) == "/data/1/20 semanas/fit"


def test_variable_loop_and_unknown_variable_raise():
    assert "loop" in _raises(substitute_variables, "${A}", {"A": "${B}", "B": "${A}"})
    assert "Unresolved" in _raises(substitute_variables, "${A}/x", {"A": "${MISSING}"})


def test_case_key_round_trip():
    for subject_id, week, key in [("1", "20 semanas", "1_20_semanas"),
                                  ("Dex-63", "26 semanas", "Dex-63_26_semanas"),
                                  ("FLA242", "", "FLA242")]:
        assert case_key(subject_id, week) == key
        assert parse_case_key(key) == {"SUBJECT_ID": subject_id, "WEEK": week}


def test_with_case_key():
    assert with_case_key({"SUBJECT_ID": "1", "WEEK": "20 semanas"})[CASE_KEY] == "1_20_semanas"
    assert with_case_key({"SUBJECT_ID": "${S}", "S": "FLA242"})[CASE_KEY] == "FLA242"
    assert CASE_KEY not in with_case_key({"WEEK": "20 semanas"})
    assert with_case_key({"SUBJECT_ID": "1", CASE_KEY: "own"})[CASE_KEY] == "own"


def _tree(root, files):
    for name in files:
        path = Path(root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_find_paths():
    with tempfile.TemporaryDirectory() as root:
        _tree(root, ["1/20 semanas/1-20s-01.nrrd", "1/20 semanas/1-20s-02.nrrd",
                     "2/26 semanas/2-26s-01.nrrd", "2/26 semanas/notes.txt", "FLA1/FLA1_01.nrrd",
                     "a[1]/x.vtp", "targets/1_20_semanas.vtp", "targets/FLA242.vtp"])
        variables = {"ROOT": root, "SESSION": "${SUBJECT_ID}/${WEEK}"}

        found = find_paths("${ROOT}/${SESSION}/${SCAN}.nrrd", variables)
        assert [values for values, _ in found] == [
            {"SUBJECT_ID": "1", "WEEK": "20 semanas", "SCAN": "1-20s-01"},
            {"SUBJECT_ID": "1", "WEEK": "20 semanas", "SCAN": "1-20s-02"},
            {"SUBJECT_ID": "2", "WEEK": "26 semanas", "SCAN": "2-26s-01"},
        ]
        assert found[0][1] == f"{root}/1/20 semanas/1-20s-01.nrrd"

        # A known variable selects; a wildcard matches inside one name.
        assert [v for v, _ in find_paths("${ROOT}/${SESSION}/*.nrrd", {**variables, "SUBJECT_ID": "2"})] == [
            {"WEEK": "26 semanas"}]
        # A variable used twice must have the same value both times.
        assert [v for v, _ in find_paths("${ROOT}/${S}/${S}_01.nrrd", variables)] == [{"S": "FLA1"}]
        # Other characters are literal, also in a known value.
        assert [v for v, _ in find_paths("${ROOT}/${DIR}/${F}.vtp", {**variables, "DIR": "a[1]"})] == [{"F": "x"}]
        # A variable made from others gives the values it is made from.
        assert [v for v, _ in find_paths("${ROOT}/targets/${CASE_KEY}.vtp", variables,
                                         parse={CASE_KEY: parse_case_key})] == [
            {"SUBJECT_ID": "1", "WEEK": "20 semanas"}, {"SUBJECT_ID": "FLA242", "WEEK": ""}]
        # ... and a path whose values do not agree does not fit.
        assert find_paths("${ROOT}/targets/${CASE_KEY}.vtp", {**variables, "SUBJECT_ID": "2"},
                          parse={CASE_KEY: parse_case_key}) == []


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
