"""The manifest.json entries that the engine writes into each stage's output folder.

    python -m pytest tests/test_manifest.py
    python tests/test_manifest.py
"""

import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared.core.pipeline import engine  # noqa: E402
from src.shared.core.pipeline.manifest import MANIFEST_FILE, output_folders  # noqa: E402
from src.shared.core.pipeline.stage_spec import StageSpec  # noqa: E402


@dataclass
class _Outputs:
    dir: Optional[str] = None


@dataclass
class _Config:
    output_dir: Optional[str] = None
    output_path: Optional[str] = None
    outputs: _Outputs = field(default_factory=_Outputs)
    value: int = 0


def _write(cfg):
    for folder in (cfg.output_dir, cfg.outputs.dir):
        if folder:
            Path(folder).mkdir(parents=True, exist_ok=True)


class _Registry:
    def __init__(self, specs):
        self.specs = specs

    def get(self, name):
        return self.specs[name]


def _manifest(folder):
    return json.loads(Path(folder, MANIFEST_FILE).read_text())


def test_output_folders():
    with tempfile.TemporaryDirectory() as root:
        Path(root, "a").mkdir()
        Path(root, "a", "x.png").touch()
        cfg = _Config(output_dir=f"{root}/a", output_path=f"{root}/a/x.png", outputs=_Outputs(f"{root}/missing"))
        assert output_folders(cfg, "output_dir") == [f"{root}/a"]
        assert output_folders(cfg, "output_path") == [f"{root}/a"]
        assert output_folders(cfg, ("output_dir", "output_path", "outputs.dir")) == [f"{root}/a"]
        assert output_folders(_Config(output_path=f"{root}/a/not_written.png"), "output_path") == []
        assert output_folders(_Config(), "output_dir") == []


def test_stage_and_pipeline_entries():
    with tempfile.TemporaryDirectory() as root:
        registry = _Registry({
            "s": StageSpec(_Config, _write),
            "nested": StageSpec(_Config, _write, output="outputs.dir"),
        })
        stage = Path(root, "stage.yaml")
        stage.write_text(yaml.safe_dump({"stage": "s", "variables": {"SUBJECT_ID": 1, "WEEK": "20 semanas"},
                                         "output_dir": f"{root}/out/${{CASE_KEY}}", "value": 3}))
        engine.run_main(stage, registry)
        entry = _manifest(f"{root}/out/1_20_semanas")["s:stage"]  # a stage file alone: <stage>:<file name>
        assert entry["config"]["output_dir"] == f"{root}/out/1_20_semanas"
        assert entry["stage"] == "s" and entry["config"]["value"] == 3
        assert entry["variables"]["CASE_KEY"] == "1_20_semanas"
        assert set(entry) == {"stage", "config_file", "variables", "config", "git", "started", "minutes"}

        pipeline = Path(root, "pipeline.yaml")
        pipeline.write_text(yaml.safe_dump({
            "stages": [{"name": "s", "id": "first"}, {"name": "s", "id": "second"},
                       {"name": "nested", "id": "fit"}, {"name": "s", "id": "off", "enabled": False}],
            "paths": {"first": {"output_dir": f"{root}/shared"}, "second": {"output_dir": f"{root}/shared"},
                      "fit": {"outputs": {"dir": f"{root}/fit"}}, "off": {"output_dir": f"{root}/off"}},
        }))
        engine.run_main(pipeline, registry)
        assert sorted(_manifest(f"{root}/shared")) == ["first", "second"]
        assert list(_manifest(f"{root}/fit")) == ["fit"]
        assert not Path(root, "off").exists()

        # A rerun replaces its own entry and keeps the others.
        engine.run_main(pipeline, registry)
        assert sorted(_manifest(f"{root}/shared")) == ["first", "second"]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
