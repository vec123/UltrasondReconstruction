"""Reading and writing a :class:`~.spec.ModeSet` as YAML.

The file is plain data - every rule and selection operation is a mapping with
a ``rule:`` / ``op:`` tag - so it can be read, diffed and hand-edited. Loading
validates it completely (unknown tags, bad kinds, dangling references) and,
given the mesh, checks that the file was authored on that mesh: brush strokes
store vertex indices, which mean nothing on another triangulation.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from src.deformation_modes.directions import (
    AwayFrom, AwayFromMidline, Between, DirectionRule, Fixed, MidlineNormal, Reversed,
    SurfaceNormal, Tangential,
)
from src.deformation_modes.mesh import TemplateMesh
from src.deformation_modes.spec import (
    SCHEMA_VERSION, Axis, Brush, ConstraintSet, FieldThreshold, Mirror, Mode, ModeSet,
    NearLandmarks, Operation, Selection, faces_fingerprint,
)


class SpecError(ValueError):
    """A mode-set file that cannot be used: malformed, inconsistent, or for another mesh."""


# -- direction rules --------------------------------------------------------------


def rule_to_dict(rule: DirectionRule) -> Dict[str, Any]:
    if isinstance(rule, Between):
        return {"rule": "between", "start": rule.start, "end": rule.end}
    if isinstance(rule, SurfaceNormal):
        return {"rule": "surface_normal"}
    if isinstance(rule, MidlineNormal):
        return {"rule": "midline_normal"}
    if isinstance(rule, AwayFromMidline):
        return {"rule": "away_from_midline"}
    if isinstance(rule, AwayFrom):
        return {"rule": "away_from", "center": list(rule.center)}
    if isinstance(rule, Fixed):
        return {"rule": "fixed", "vector": [float(c) for c in rule.vector]}
    if isinstance(rule, Tangential):
        return {"rule": "tangential", "of": rule_to_dict(rule.rule)}
    if isinstance(rule, Reversed):
        return {"rule": "reversed", "of": rule_to_dict(rule.rule)}
    raise SpecError(f"cannot serialise direction rule {rule!r}")


def rule_from_dict(data: Mapping[str, Any]) -> DirectionRule:
    kind = data.get("rule")
    try:
        if kind == "between":
            return Between(str(data["start"]), str(data["end"]))
        if kind == "surface_normal":
            return SurfaceNormal()
        if kind == "midline_normal":
            return MidlineNormal()
        if kind == "away_from_midline":
            return AwayFromMidline()
        if kind == "away_from":
            return AwayFrom(tuple(str(c) for c in data["center"]))
        if kind == "fixed":
            return Fixed(tuple(float(c) for c in data["vector"]))
        if kind == "tangential":
            return Tangential(rule_from_dict(data["of"]))
        if kind == "reversed":
            return Reversed(rule_from_dict(data["of"]))
    except KeyError as exc:
        raise SpecError(f"direction rule {kind!r} is missing {exc}") from None
    raise SpecError(f"unknown direction rule {kind!r}")


# -- selections ---------------------------------------------------------------------


def _op_to_dict(op: Operation) -> Dict[str, Any]:
    if isinstance(op, Brush):
        out = {"op": "brush", "vertex": int(op.vertex), "radius": float(op.radius)}
    elif isinstance(op, NearLandmarks):
        out = {"op": "near_landmarks", "labels": list(op.labels), "radius": float(op.radius)}
    elif isinstance(op, FieldThreshold):
        out = {"op": "field", "field": op.field}
        if op.above is not None:
            out["above"] = float(op.above)
        if op.below is not None:
            out["below"] = float(op.below)
    elif isinstance(op, Mirror):
        return {"op": "mirror"}
    else:
        raise SpecError(f"cannot serialise selection op {op!r}")
    if op.subtract:
        out["subtract"] = True
    return out


def _op_from_dict(data: Mapping[str, Any]) -> Operation:
    kind = data.get("op")
    subtract = bool(data.get("subtract", False))
    try:
        if kind == "brush":
            return Brush(int(data["vertex"]), float(data["radius"]), subtract)
        if kind == "near_landmarks":
            return NearLandmarks(tuple(str(l) for l in data["labels"]), float(data["radius"]), subtract)
        if kind == "field":
            return FieldThreshold(str(data["field"]),
                                  None if data.get("above") is None else float(data["above"]),
                                  None if data.get("below") is None else float(data["below"]),
                                  subtract)
        if kind == "mirror":
            return Mirror()
    except KeyError as exc:
        raise SpecError(f"selection op {kind!r} is missing {exc}") from None
    raise SpecError(f"unknown selection op {kind!r}")


def _selection_to_list(selection: Selection):
    return [_op_to_dict(op) for op in selection.ops]


def _selection_from_list(data) -> Selection:
    return Selection(tuple(_op_from_dict(op) for op in (data or [])))


# -- the mode set ---------------------------------------------------------------------


def modeset_to_dict(modeset: ModeSet) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "description": modeset.description,
        "template": modeset.template,
        "mesh": {"n_vertices": modeset.n_vertices, "faces_sha1": modeset.faces_sha1},
        "defaults": {"sigma": modeset.sigma, "spacing": modeset.spacing, "nu": modeset.nu},
        "axes": [{"name": a.name, **rule_to_dict(a.rule)} for a in modeset.axes],
        "constraint_sets": [
            {"name": c.name, "kind": c.kind, "weight": c.weight,
             "select": _selection_to_list(c.selection)}
            for c in modeset.constraint_sets
        ],
        "modes": [
            {"name": m.name, "kind": m.kind, "axis": m.axis,
             **({"magnitude": m.magnitude} if m.kind == "displacement" else {"stretch": list(m.stretch)}),
             "constraints": list(m.constraints),
             **({"sigma": m.sigma} if m.sigma is not None else {}),
             "sites": _selection_to_list(m.sites)}
            for m in modeset.modes
        ],
    }


def modeset_from_dict(data: Mapping[str, Any]) -> ModeSet:
    """Build and validate a ModeSet.

    Raises:
        SpecError: wrong schema version, malformed entries, or a mode referring
            to an axis or constraint set that does not exist.
    """
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SpecError(f"schema_version {version!r} is not supported (expected {SCHEMA_VERSION})")
    try:
        mesh = data["mesh"]
        defaults = data.get("defaults", {})
        axes = tuple(Axis(str(a["name"]), rule_from_dict(a)) for a in data.get("axes", []))
        sets = tuple(ConstraintSet(str(c["name"]), str(c.get("kind", "anchor")),
                                   _selection_from_list(c.get("select")), float(c.get("weight", 1.0)))
                     for c in data.get("constraint_sets", []))
        modes = tuple(Mode(str(m["name"]), str(m.get("kind", "displacement")),
                           _selection_from_list(m.get("sites")), str(m.get("axis", "")),
                           float(m.get("magnitude", 1.0)),
                           tuple(m.get("stretch", (1.0, 0.0, 0.0))),
                           tuple(str(c) for c in m.get("constraints", [])),
                           None if m.get("sigma") is None else float(m["sigma"]))
                      for m in data.get("modes", []))
        modeset = ModeSet(template=str(data.get("template", "")), n_vertices=int(mesh["n_vertices"]),
                          faces_sha1=str(mesh["faces_sha1"]), sigma=float(defaults["sigma"]),
                          spacing=float(defaults.get("spacing", 0.5)), nu=float(defaults.get("nu", 1e-4)),
                          axes=axes, constraint_sets=sets, modes=modes,
                          description=str(data.get("description", "")))
    except KeyError as exc:
        raise SpecError(f"missing key {exc}") from None
    except (TypeError, ValueError) as exc:
        raise SpecError(str(exc)) from None

    for mode in modeset.modes:
        if mode.axis and not modeset.has("axes", mode.axis):
            raise SpecError(f"mode {mode.name!r} uses axis {mode.axis!r}, which is not defined")
        for name in mode.constraints:
            if not modeset.has("constraint_sets", name):
                raise SpecError(f"mode {mode.name!r} uses constraint set {name!r}, which is not defined")
    return modeset


def check_mesh(modeset: ModeSet, mesh: TemplateMesh) -> None:
    """Raise SpecError unless ``modeset`` was authored on ``mesh``'s triangulation."""
    if modeset.n_vertices != mesh.n_vertices or modeset.faces_sha1 != faces_fingerprint(mesh.faces):
        raise SpecError(
            f"this mode set was authored on a mesh with {modeset.n_vertices} vertices "
            f"(faces sha1 {modeset.faces_sha1[:10]}...), not on this one ({mesh.n_vertices} "
            f"vertices, sha1 {faces_fingerprint(mesh.faces)[:10]}...): its vertex indices "
            f"would point at the wrong places"
        )


def load_modeset(path, mesh: Optional[TemplateMesh] = None) -> ModeSet:
    """Read a mode-set YAML file (and check it against ``mesh`` when given)."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise SpecError(f"{path}: top level must be a mapping")
    try:
        modeset = modeset_from_dict(data)
    except SpecError as exc:
        raise SpecError(f"{path}: {exc}") from None
    if mesh is not None:
        check_mesh(modeset, mesh)
    return modeset


def save_modeset(path, modeset: ModeSet, backup: bool = True) -> Path:
    """Write ``modeset`` as YAML; the previous file (if any) is kept as ``<name>.bak``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Deformation modes - written by src/GUIs/deformation_modes; hand edits are\n"
                     "# fine (comments are not preserved). See src/deformation_modes/spec.py.\n")
        yaml.safe_dump(modeset_to_dict(modeset), handle, sort_keys=False, default_flow_style=None,
                       width=100)
    return path
