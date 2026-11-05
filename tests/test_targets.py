"""targets: and batch: - which cases a YAML runs on, and how they run.

    python -m pytest tests/test_targets.py
    python tests/test_targets.py
"""

import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared.core.pipeline import engine  # noqa: E402
from src.shared.core.pipeline.batch import STATUS_FILE, CasesFailed  # noqa: E402
from src.shared.core.pipeline.targets import find_cases  # noqa: E402


class _Spec:
    """A stage that records its config and fails for SUBJECT_ID 2."""

    def __init__(self, runs):
        self.runs = runs
        self.from_dict = lambda raw: raw

    def run(self, cfg):
        if cfg["subject"] == "2":
            raise RuntimeError("subject 2 fails")
        self.runs.append(cfg)


class _Registry:
    def __init__(self, spec):
        self.spec = spec

    def get(self, name):
        return self.spec


def _tree(root, files):
    for name in files:
        path = Path(root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _yaml(root, content):
    path = Path(root, "run.yaml")
    path.write_text(yaml.safe_dump(content))
    return path


def test_find_cases():
    with tempfile.TemporaryDirectory() as root:
        _tree(root, ["1/20 semanas/a.nrrd", "1/20 semanas/b.nrrd", "1/26 semanas/c.nrrd", "FLA1/d.nrrd"])
        targets = {"match": ["${ROOT}/${SUBJECT_ID}/${WEEK}/${SCAN}.nrrd", "${ROOT}/${SUBJECT_ID}/${SCAN}.nrrd"],
                   "first": ["SCAN"]}
        cases = find_cases(targets, {"ROOT": root}, "test")
        assert [(c.name, c.values, c.n_files) for c in cases] == [
            ("1_20_semanas", {"SUBJECT_ID": "1", "WEEK": "20 semanas", "SCAN": "a"}, 2),
            ("1_26_semanas", {"SUBJECT_ID": "1", "WEEK": "26 semanas", "SCAN": "c"}, 1),
            ("FLA1", {"SUBJECT_ID": "FLA1", "WEEK": "", "SCAN": "d"}, 1),
        ]
        assert [c.name for c in find_cases({**targets, "exclude": [{"WEEK": "26 semanas"}]}, {"ROOT": root}, "t")] == [
            "1_20_semanas", "FLA1"]
        assert [c.name for c in find_cases({**targets, "only": [{"SUBJECT_ID": "FLA1"}]}, {"ROOT": root}, "t")] == [
            "FLA1"]
        listed = find_cases({"cases": [{"SUBJECT_ID": 3, "WEEK": "20 semanas"}]}, {}, "t")
        assert [(c.name, c.values) for c in listed] == [("3_20_semanas", {"SUBJECT_ID": "3", "WEEK": "20 semanas"})]
        # All its variables given (e.g. by the caller): the one case, a first: variable included.
        given = {"ROOT": root, "SUBJECT_ID": "1", "WEEK": "20 semanas", "SCAN": "b"}
        assert [(c.name, c.values, c.n_files) for c in find_cases(targets, given, "t")] == [("case", {}, 1)]

        for wrong, message in [({"match": "x", "cases": []}, "exactly one"), ({"cases": [], "where": 1}, "unknown"),
                               ({"match": f"{root}/none/${{X}}"}, "No path fits"),
                               ({"cases": [], "first": ["X"]}, "only with"),
                               ({"match": targets["match"], "first": ["NAME"]}, "no 'match:' pattern finds")]:
            try:
                find_cases(wrong, {"ROOT": root}, "t")
            except ValueError as error:
                assert message in str(error), (message, error)
            else:
                raise AssertionError(f"{wrong} did not raise")


def test_run_main_in_process():
    with tempfile.TemporaryDirectory() as root:
        config = _yaml(root, {"stage": "s", "variables": {"OUT": "/out/${CASE_KEY}"},
                              "targets": {"cases": [{"SUBJECT_ID": 1, "WEEK": "20 semanas"},
                                                    {"SUBJECT_ID": 2}, {"SUBJECT_ID": 3}]},
                              "subject": "${SUBJECT_ID}", "out": "${OUT}"})
        runs = []
        try:
            engine.run_main(config, _Registry(_Spec(runs)))
        except CasesFailed as error:
            assert "1 of 3 cases failed: ['2']" in str(error)
        else:
            raise AssertionError("the failed case was not reported")
        assert runs == [{"subject": "1", "out": "/out/1_20_semanas"}, {"subject": "3", "out": "/out/3"}]

        # One case, as the process of each case runs it: targets: is not used.
        runs.clear()
        engine.run_case(config, _Registry(_Spec(runs)), {"SUBJECT_ID": "7"})
        assert runs == [{"subject": "7", "out": "/out/7"}]


def test_run_main_in_processes():
    with tempfile.TemporaryDirectory() as root:
        _tree(root, ["done/3.txt"])
        config = _yaml(root, {"stage": "s",
                              "targets": {"cases": [{"SUBJECT_ID": n} for n in (1, 2, 3)]},
                              "batch": {"workers": 2, "log_dir": f"{root}/logs",
                                        "skip_if_exists": f"{root}/done/${{SUBJECT_ID}}.txt"}})

        def command(config_path, case):
            code = 1 if case["SUBJECT_ID"] == "2" else 0
            return [sys.executable, "-c", f"print('Running stage: fit'); print({case!r}); raise SystemExit({code})"]

        try:
            engine.run_main(config, _Registry(None), case_command=command)
        except CasesFailed as error:
            assert "1 of 3 cases failed: ['2']" in str(error)
        else:
            raise AssertionError("the failed case was not reported")
        logs = Path(root, "logs")
        assert sorted(p.name for p in logs.iterdir()) == ["1.log", "2.log", STATUS_FILE]
        assert "'SUBJECT_ID': '1'" in (logs / "1.log").read_text()
        rows = sorted(line.split("\t")[::3] for line in (logs / STATUS_FILE).read_text().splitlines()[1:])
        assert rows == [["1", "0"], ["2", "1"]]
        assert "failed at: fit" in (logs / STATUS_FILE).read_text()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
