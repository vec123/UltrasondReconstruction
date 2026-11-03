"""Import rules between the layers of src/.

Each package of src/ belongs to one layer (LAYERS), and a layer may import
only itself and the layers in ALLOWED:

    shared       core, geometry, vtk, transforms, correspondence, ...
    ultrasound   the ultrasound processing        -> shared
    gpmm, lddmm  the two models                   -> shared
    evaluation   the metrics on model outputs     -> shared
    apps         the GUIs                         -> all of src/

No file of src/ imports scripts/. Imports inside functions count too.

The code does not obey every rule yet. KNOWN_VIOLATIONS lists the imports
that break a rule today, as (importing file, imported module). The test fails
on a violation that is not in the list, and on an entry of the list that is
no longer a violation, so the list can only get shorter: remove an entry when
the refactor removes its import. When a package moves, change LAYERS.

    python -m pytest tests/test_import_rules.py
    python tests/test_import_rules.py            (prints the report)
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Path prefix -> layer. The longest matching prefix wins.
LAYERS = {
    "src/__init__.py": "shared",
    "src/core": "shared",
    "src/config": "shared",
    "src/geometry": "shared",
    "src/vtk": "shared",
    "src/transforms": "shared",
    "src/landmarks": "shared",
    "src/graphs": "shared",
    "src/nrrd": "shared",
    "src/file_name_matching": "shared",
    "src/deformation_modes": "shared",
    "src/weight_field": "shared",
    "src/SSM/correspondence": "shared",
    "src/ultrasound": "ultrasound",
    "src/config/US": "ultrasound",
    "src/SSM": "gpmm",
    "src/config/SSM": "gpmm",
    "src/resnet_lddmm": "lddmm",
    "src/evaluation": "evaluation",
    "src/GUIs": "apps",
    "scripts": "scripts",
}

ALLOWED = {
    "shared": {"shared"},
    "ultrasound": {"shared", "ultrasound"},
    "gpmm": {"shared", "gpmm"},
    "lddmm": {"shared", "lddmm"},
    "evaluation": {"shared", "evaluation"},
    "apps": {"shared", "ultrasound", "gpmm", "lddmm", "evaluation", "apps"},
}

# Third-party code that is not checked.
SKIP = ("src/SSM/BabyFM",)

KNOWN_VIOLATIONS = {
    # PipelineOutputFiles (the GPMM output names): to models/gpmm/outputs.py
    ("src/SSM/correspondence/loader/layouts.py", "src.config.SSM.schema.pipeline"),
    # load_shape and _robust_penalty: to shared
    ("src/SSM/model/builder/injectors.py", "src.resnet_lddmm.resnet_flow.io"),
    ("src/SSM/nicp/losses/chamfer.py", "src.resnet_lddmm.resnet_flow.losses.data_terms"),
    # the landmark GUI, re-exported as a runner
    ("src/SSM/runners/__init__.py", "src.GUIs.set_landmarks"),
    # Shape, reject_unknown_keys and the landmark readers: to shared
    ("src/weight_field/landmark_weight_field.py", "src.resnet_lddmm.resnet_flow.config"),
    ("src/weight_field/landmark_weight_field.py", "src.resnet_lddmm.resnet_flow.guidance.landmarks"),
    ("src/weight_field/landmark_weight_field.py", "src.resnet_lddmm.resnet_flow.io"),
    ("src/weight_field/target_weight_field.py", "src.resnet_lddmm.resnet_flow.config"),
    ("src/weight_field/target_weight_field.py", "src.resnet_lddmm.resnet_flow.guidance.landmarks"),
    ("src/weight_field/target_weight_field.py", "src.resnet_lddmm.resnet_flow.io"),
}


def layer_of(path: str):
    """The layer of a path relative to ROOT ('src/SSM/nicp/problem.py'), or None."""
    matches = [prefix for prefix in LAYERS if path == prefix or path.startswith(prefix + "/")]
    return LAYERS[max(matches, key=len)] if matches else None


def module_path(module: str) -> str:
    """'src.SSM.nicp' -> 'src/SSM/nicp'."""
    return module.replace(".", "/")


def is_module(module: str) -> bool:
    path = ROOT / module_path(module)
    return path.with_suffix(".py").is_file() or path.is_dir()


def imported_modules(path: Path):
    """Every src. / scripts. module that the file imports, as dotted names."""
    package = ".".join(path.relative_to(ROOT).parent.parts)
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            candidates = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module
            # `from src.SSM import correspondence` imports src.SSM.correspondence.
            candidates = [f"{module}.{alias.name}" if is_module(f"{module}.{alias.name}") else module
                          for alias in node.names]
        else:
            continue
        for module in candidates:
            if module.split(".")[0] in ("src", "scripts"):
                yield module


def find_violations():
    """(importing file, imported module) for every import that breaks a rule."""
    violations = set()
    for path in sorted((ROOT / "src").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(SKIP):
            continue
        own = layer_of(relative)
        assert own is not None, f"{relative} is in no layer: add its package to LAYERS"
        for module in imported_modules(path):
            other = layer_of(module_path(module))
            if other not in ALLOWED[own]:
                violations.add((relative, module))
    return violations


def test_no_new_violations():
    new = find_violations() - KNOWN_VIOLATIONS
    assert not new, "imports that break the layer rules:\n" + "\n".join(
        f"  {file} -> {module}" for file, module in sorted(new))


def test_known_violations_still_exist():
    fixed = KNOWN_VIOLATIONS - find_violations()
    assert not fixed, "these imports are fixed, remove them from KNOWN_VIOLATIONS:\n" + "\n".join(
        f"  {file} -> {module}" for file, module in sorted(fixed))


if __name__ == "__main__":
    violations = find_violations()
    for file, module in sorted(violations):
        own, other = layer_of(file), layer_of(module_path(module))
        print(f"{own:>10} -> {other:<10} {file} -> {module}")
    print(f"{len(violations)} violations, {len(violations - KNOWN_VIOLATIONS)} not in KNOWN_VIOLATIONS")
