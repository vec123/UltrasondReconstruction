"""The kinds of point field the point_fields stage adds to a mesh.

A kind is a class with its settings as fields and ``apply(polydata)``, which
adds its arrays to the mesh in place (every array the mesh has stays) and
returns a one-line report. The calculations are the shared functions that
scan_fusion and the weight-field scripts use too, so a field has one
definition.

To add a kind: write the class, register it under a name. Nothing else changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from src.shared.core.config import from_dict
from src.shared.core.registry import TypedRegistry
from src.shared.geometry.shape import shape_from_polydata
from src.shared.vtk.fields import add_point_field
from src.shared.vtk.surface_measures import compute_point_cloud_edges, compute_vertex_normals
from src.shared.weight_field.landmark_weight_field import LandmarkWeightFieldCfg, compute_landmark_weight_field
from src.shared.weight_field.target_weight_field import TargetWeightFieldCfg, compute_target_weight_field

FIELD_KINDS = TypedRegistry("point field kind")
register_field = FIELD_KINDS.register
KIND = "kind"  # the key of a `fields:` entry that names its kind


class FieldKind(ABC):
    """One kind of point field."""

    @classmethod
    def from_settings(cls, settings, mesh_path, source):
        """The kind with ``settings`` (its `fields:` entry without ``kind``) for the mesh ``mesh_path``."""
        return from_dict(cls, settings, source=source)

    @abstractmethod
    def apply(self, polydata) -> str:
        """Adds this kind's point arrays to ``polydata``; returns a one-line report."""


def build_field(entry, mesh_path, source):
    """The FieldKind of one `fields:` entry: ``{kind: <name>, <its settings>}``."""
    if not isinstance(entry, dict) or KIND not in entry:
        raise ValueError(f"Each entry of 'fields:' in '{source}' needs '{KIND}: <name>' "
                         f"(one of {FIELD_KINDS.names()}), got {entry!r}.")
    settings = {key: value for key, value in entry.items() if key != KIND}
    return FIELD_KINDS.get(entry[KIND]).from_settings(settings, mesh_path, f"{source} ({entry[KIND]})")


@register_field("normals")
@dataclass
class Normals(FieldKind):
    """The unit normal of the surface at each vertex, from the faces (surface_measures.compute_vertex_normals)."""

    field: str = "normal"
    weighting: str = "unit"  # how the incident faces are averaged: "unit" or "area"

    def apply(self, polydata):
        normals = compute_vertex_normals(polydata, field_name=self.field, weighting=self.weighting)
        lengths = np.linalg.norm(normals, axis=1)
        degenerate = int((lengths < 1e-6).sum())
        return (f"'{self.field}' ({self.weighting}-weighted) on {len(lengths)} vertices, "
                f"{degenerate} zero-length ({degenerate / len(lengths):.2%})")


@register_field("edge")
@dataclass
class Edge(FieldKind):
    """The rim and hole borders of a point-cloud surface (surface_measures.compute_point_cloud_edges):
    edge, edge_distance, edge_weight, and <field>_edge = <field> x edge_weight for each ``combine`` field."""

    k: int = 16  # neighbours per point for the gap test
    gap_deg: float = 120.0  # largest angular gap an interior point may show
    ramp_spacings: float = 3.0  # ramp width in point spacings
    combine: List[str] = field(default_factory=lambda: ["support"])

    def apply(self, polydata):
        result = compute_point_cloud_edges(polydata, k=self.k, gap_deg=self.gap_deg,
                                           ramp_spacings=self.ramp_spacings)
        for name in self.combine:
            array = polydata.GetPointData().GetArray(name)
            if array is None:
                raise ValueError(f"The mesh has no '{name}' point array to combine with edge_weight")
            add_point_field(polydata, (vtk_to_numpy(array) * result["weight"]).astype(np.float32),
                            field_name=f"{name}_edge")
        return (f"{result['edge'].sum()} edge points ({result['edge'].mean():.1%}), spacing {result['spacing']:.3g}, "
                f"ramp {result['ramp']:.3g} (file units), {(result['weight'] < 1.0).mean():.1%} of points "
                f"below full weight; wrote edge, edge_distance, edge_weight"
                + "".join(f", {name}_edge" for name in self.combine))


def describe_entry(entry):
    """One landmark of a weight field, with its settings, for the report."""
    base = f"{entry.label}(scale={entry.decay_scale}, max={entry.max_weight}"
    if entry.decay_axes:
        axes = ", ".join(f"{axis}=[{neg},{pos}]" for axis, (neg, pos) in entry.decay_axes.items())
        base += f", axes={{{axes}}}"
    return base + ")"


class WeightField(FieldKind):
    """A weight field of src/shared/weight_field. Its own config class reads the settings
    (the keys of the old YAMLs, without ``vtp`` and ``output``: the stage's input and output)."""

    config_cls = None
    compute = None  # function(Shape, cfg) -> [N] tensor

    def __init__(self, cfg):
        self.cfg = cfg

    @classmethod
    def from_settings(cls, settings, mesh_path, source):
        given = sorted({"vtp", "output"} & set(settings))
        if given:
            raise ValueError(f"{source}: {given} are the input and output of the stage; remove them from the entry.")
        return cls(cls.config_cls.from_dict({**settings, "vtp": mesh_path}))

    def apply(self, polydata):
        weight = type(self).compute(shape_from_polydata(polydata), self.cfg).numpy()
        add_point_field(polydata, weight, field_name=self.cfg.output_field)
        details = ", ".join(describe_entry(entry) for entry in self.cfg.entries)
        return (f"{self.cfg.output_field!r} from {len(self.cfg.entries)} landmark(s) ({details}), "
                f"min_weight={self.cfg.min_weight}: range [{weight.min():.4g}, {weight.max():.4g}]")


@register_field("landmark_weight")
class LandmarkWeight(WeightField):
    """The strongest of the included landmarks' peaked Gaussians on a template
    (weight_field.landmark_weight_field); the landmarks come from its heatmap fields."""

    config_cls = LandmarkWeightFieldCfg
    compute = staticmethod(compute_landmark_weight_field)


@register_field("target_weight")
class TargetWeight(WeightField):
    """Geodesic landmark proximity x a reliability field on a scan (weight_field.target_weight_field)."""

    config_cls = TargetWeightFieldCfg
    compute = staticmethod(compute_target_weight_field)
