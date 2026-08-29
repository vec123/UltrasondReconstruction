"""I/O for shapes and trajectories (STEPS T13+, T32)."""

from dataclasses import dataclass, field
from typing import Optional
import os

import torch
import numpy as np

from src.vtk.io import load_vtp, load_obj, save_vtp
from src.vtk.extract import (
    extract_all_vtp_point_fields,
    extract_vtp_points_cells,
    extract_vtp_point_fields,
)
from src.vtk.create import create_polydata
from src.vtk.fields import add_point_field, add_point_fields


@dataclass(frozen=True)
class FrameTransform:
    """Affine normalization transform: center and scale (STEPS T14, T32).

    Applies the transformation: (x - center) / scale
    Inverts via: x * scale + center

    Optional pose (T32): includes encoder-predicted rotation/translation.
    When present, invert() applies: x * scale + center, then rotates and translates.
    """

    center: torch.Tensor  # [3]
    scale: torch.Tensor  # scalar or [1]
    rotation: Optional[torch.Tensor] = None  # [3, 3] or [B, 3, 3] or None
    translation: Optional[torch.Tensor] = None  # [3] or [B, 3] or None

    @classmethod
    def identity(cls):
        """The no-op transform, for staying in the normalised frame.

        Exporters take a transform rather than a flag, so "write these points as
        they are" is spelled by passing this instead of by branching inside every
        exporter.
        """
        return cls(center=torch.zeros(3), scale=torch.tensor(1.0))

    def apply(self, points):
        """Apply the transform: (points - center) / scale.

        Args:
            points: [B, N, 3] or [N, 3]

        Returns:
            Transformed points in the same shape
        """
        return (points - self.center) / self.scale

    def invert(self, points):
        """Invert the transform: points * scale + center, then apply pose if available.

        Args:
            points: [N, 3] or [B, N, 3]
            rotation: [3, 3] or None (learned encoder pose)
            translation: [3] or None (learned encoder pose)

        Returns:
            Inverted points (with learned pose applied if available)
        """
        # First denormalize: x * scale + center
        world_points = points * self.scale + self.center

        # Then apply encoder-learned rotation and translation if available
        if self.rotation is not None:
            # Rotation: points @ R^T (works for any shape ending in 3)
            world_points = torch.matmul(world_points, self.rotation.T)

        if self.translation is not None:
            world_points = world_points + self.translation

        return world_points


@dataclass
class Shape:
    """A shape with points, faces, and optional weights and normals.

    Attributes:
        points: [1, N, 3] point cloud in normalized coordinates.
        faces: [F, 3] triangular face indices.
        weights: [1, N] per-point weights (e.g., area) or None (caller uses uniform).
        normals: [1, N, 3] per-point normals or None (caller computes if needed).
        point_fields: {name: numpy array} of every OTHER point array the file carried
            -- heatmaps, mode scores, labels -- or None. Passenger data: the model
            never reads it, it is kept so the exporters can put it back on the
            deformed shape, where it is still valid because a deformation moves the
            points without reordering or resampling them. numpy rather than torch,
            and uncast, so an int label array stays int.
    """

    points: torch.Tensor  # [1, N, 3]
    faces: torch.Tensor  # [F, 3]
    weights: Optional[torch.Tensor] = None  # [1, N] or None
    normals: Optional[torch.Tensor] = None  # [1, N, 3] or None
    point_fields: Optional[dict] = None  # {name: [N] or [N, C] numpy} or None


def load_shape(path):
    """Load a shape from a VTP or OBJ file.

    Extracts points, faces, and optional point fields (area, normal).
    Missing fields return None (caller handles uniform fallback).

    Args:
        path: path to .vtp or .obj file

    Returns:
        Shape with points [1,N,3], faces [F,3], weights/normals or None
    """
    # Detect format by extension
    _, ext = os.path.splitext(path.lower())

    if ext == ".vtp":
        poly = load_vtp(path)
    elif ext == ".obj":
        poly = load_obj(path)
    else:
        raise ValueError(f"Unsupported format: {ext}. Supported: .vtp, .obj")

    points_np, faces_np = extract_vtp_points_cells(poly)
    fields = extract_vtp_point_fields(poly, ["area", "normal"])

    # Reshape points to [1, N, 3]
    # The run's precision, not a hardcoded one: everything downstream derives its
    # dtype from these tensors, so this is where cfg.dtype actually takes effect.
    points = torch.tensor(points_np, dtype=torch.get_default_dtype()).unsqueeze(0)

    # Reshape faces to [F, 3]
    faces = torch.tensor(faces_np, dtype=torch.int64)

    # Extract weights from "area" field (or None if absent)
    weights = None
    if fields["area"] is not None:
        weights = torch.tensor(fields["area"],
                               dtype=torch.get_default_dtype()).unsqueeze(0)

    # Extract normals (or None if absent)
    normals = None
    if fields["normal"] is not None:
        normals = torch.tensor(fields["normal"],
                               dtype=torch.get_default_dtype()).unsqueeze(0)

    # Everything else the file carried, kept verbatim so the exporters can put it
    # back on the deformed shape. "area"/"normal" are in here too and that is
    # deliberate: they are still point data, and a template exported without them
    # would have lost arrays its input had.
    point_fields = extract_all_vtp_point_fields(poly) if ext == ".vtp" else {}

    return Shape(points=points, faces=faces, weights=weights, normals=normals,
                 point_fields=point_fields)


def joint_normalize(shapes, domain=(0, 1)):
    """Normalize a list of shapes to fit jointly into a domain (STEPS T14).

    Computes one bounding box over all shapes' union, then fits it to the
    target domain with a single center and scale applied to all.

    Args:
        shapes: list of Shape objects
        domain: (min, max) tuple for target domain (default (0, 1))

    Returns:
        (normalized_shapes, transform) where normalized_shapes is a list of
        Shape objects with normalized points, and transform is the FrameTransform
        used (for exporting back to world coordinates)
    """
    if not shapes:
        raise ValueError("shapes list cannot be empty")

    # Collect all points: [B, N, 3] -> flatten to [total_points, 3]
    all_points = torch.cat([s.points.reshape(-1, 3) for s in shapes], dim=0)

    # Compute bbox
    min_pt = all_points.min(dim=0).values  # [3]
    max_pt = all_points.max(dim=0).values  # [3]

    # Bbox size
    bbox_size = max_pt - min_pt  # [3]
    # Use the max dimension for uniform scaling (fits tightest box into cube)
    scale = bbox_size.max()

    # Center of bbox
    center = (min_pt + max_pt) / 2

    # Fit to domain: normalize to [-0.5, 0.5], then scale to domain
    domain_min, domain_max = domain
    domain_scale = domain_max - domain_min
    domain_center = (domain_max + domain_min) / 2

    # Fold the domain into the transform, so invert() is a TRUE inverse of the
    # normalisation. Applying the domain shift afterwards instead would leave
    # transform.invert() short by domain_center * scale -- 0.718 for the rabbit under
    # the default [0,1] domain, i.e. half the bounding box in every axis, silently
    # displacing every exported "world" shape.
    #   (p - c)/s * D + d  ==  (p - c')/s'   with   c' = c - d*s/D,  s' = s/D
    transform = FrameTransform(center=center - domain_center * scale / domain_scale,
                               scale=scale / domain_scale)

    # Normalize all shapes
    normalized_shapes = []
    for shape in shapes:
        norm_points = transform.apply(shape.points)

        normalized_shapes.append(Shape(
            points=norm_points,
            faces=shape.faces,
            weights=shape.weights,
            normals=shape.normals,
            # Per-point passenger data: normalisation moves the points, not the
            # values attached to them, so it survives the frame change untouched.
            point_fields=shape.point_fields
        ))

    return normalized_shapes, transform


def load_cohort(folder, template):
    """Load a cohort of shapes from a folder and normalize jointly with template (STEPS T26).

    Loads all .vtp files from a folder, assigns sequential shape_ids, and normalizes
    the cohort jointly with a template using a single bounding box (one frame).

    Args:
        folder: path to folder containing .vtp files
        template: Shape object or path to .vtp/.obj template file

    Returns:
        (cohort_shapes, template_norm, transform) where:
        - cohort_shapes: list of (shape_id, Shape) tuples for each cohort member
        - template_norm: normalized template Shape
        - transform: FrameTransform used for joint normalization
    """
    # Load template if it's a path
    if isinstance(template, str):
        template = load_shape(template)

    # Collect all .vtp files from cohort folder
    cohort_filenames = sorted([f for f in os.listdir(folder) if f.lower().endswith('.vtp')])
    if not cohort_filenames:
        raise ValueError(f"No .vtp files found in {folder}")

    cohort_shapes_unnorm = []
    for fname in cohort_filenames:
        fpath = os.path.join(folder, fname)
        shape = load_shape(fpath)
        cohort_shapes_unnorm.append(shape)

    # Joint normalization: template + all cohort shapes in one frame
    all_shapes = [template] + cohort_shapes_unnorm
    normalized_all, transform = joint_normalize(all_shapes)

    # Extract normalized template and cohort
    template_norm = normalized_all[0]
    cohort_shapes = [(i, normalized_all[i + 1]) for i in range(len(cohort_shapes_unnorm))]

    return cohort_shapes, template_norm, transform


def to_frame(points, transform, label="points"):
    """``transform.invert(points)``, with a loud guard on non-finite values.

    NaN/Inf crash ParaView rather than showing up as bad geometry, so every export
    path funnels through here instead of writing whatever the model produced.

    Args:
        points: [N, 3] in the normalised frame
        transform: FrameTransform; pass an identity one to stay in that frame
        label: name used in the warning

    Returns:
        [N, 3] in the frame ``transform`` inverts into
    """
    points = transform.invert(points)
    bad = torch.isnan(points) | torch.isinf(points)
    if bad.any():
        print(f"[WARNING] {label} points contain NaN/Inf, replacing with zeros")
        points = torch.where(bad, torch.tensor(0.0, device=points.device), points)
    return points


def export_shape(points, faces, transform, path, label="Shape", point_fields=None):
    """Write ONE cloud (or mesh) to a .vtp file.

    Connectivity is written only when it addresses these exact vertices --
    ``faces`` is dropped when it is empty or indexes beyond the cloud, which is
    what happens whenever the points have been subsampled. A face array that
    survives a subsample joins points that are not the ones it names, and the
    result looks like a plausible mesh rather than like an error, so the check is
    here rather than at the call sites.

    ``point_fields`` is written under the same rule, for the same reason: a
    deformation moves points without reordering them, so row i of every array still
    describes point i and the arrays carry over exactly -- but only while the cloud
    is still the one they were measured on. See _fields_for.

    Args:
        points: [N, 3] in the normalised frame
        faces: [F, 3] triangle indices into the FULL shape, or None
        transform: FrameTransform, applied via invert()
        path: destination .vtp path
        label: name used in warnings
        point_fields: {name: [N] or [N, C] array} to write alongside the geometry,
            typically the source Shape's ``point_fields``; None writes bare geometry

    Returns:
        The path written
    """
    points = to_frame(points, transform, label)

    if faces is not None and (len(faces) == 0 or int(faces.max()) >= points.shape[0]):
        faces = None

    polydata = create_polydata(points, faces=faces)
    if point_fields:
        add_point_fields(polydata, _fields_for(point_fields, points.shape[0], label))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_vtp(polydata, path, binary=True)
    return path


def _fields_for(point_fields, n_points, label):
    """The subset of ``point_fields`` whose arrays still address ``n_points`` points.

    The face guard's argument applied to point data. A per-point array carried across
    a subsample labels points that are not the ones it was measured on, and unlike
    broken connectivity that does not even look wrong -- it looks like data. So the
    mismatched arrays are dropped, loudly, rather than written.
    """
    kept = {name: values for name, values in point_fields.items()
            if len(values) == n_points}

    dropped = sorted(set(point_fields) - set(kept))
    if dropped:
        print(f"[WARNING] {label}: dropped {len(dropped)} point field(s) that do not "
              f"address {n_points} points: {', '.join(dropped)}")
    return kept


def _faces_of(shape):
    """The face array of a Shape as a tensor, or None if it has no usable one.

    ``faces`` is a per-shape list on a cohort batch and a bare tensor on a single
    shape, so a caller that has already sliced one shape out can still be holding
    either. Normalising here keeps that off the export path.
    """
    faces = getattr(shape, "faces", None)
    if isinstance(faces, list):
        faces = faces[0] if len(faces) > 0 else None
    return faces


def export_reference_shapes(template, sample, transform, out_dir):
    """Export the template and sample reference shapes as VTP files.

    Written alongside the trajectory frames so all three can be loaded together:
    ``template_points.vtp`` is where the flow starts, ``sample_points.vtp`` is what it
    is being fitted to.

    The template keeps its connectivity when it still has any -- at full resolution
    the file is a MESH, which is what makes it comparable to ``deformed.vtp`` rather
    than a cloud beside a surface. It falls back to points whenever the faces stop
    addressing these exact vertices, which is the subsampled training case; the
    decision lives in export_shape, so both files answer it the same way. The name
    stays ``template_points.vtp`` either way: it is what the manifest lists and what
    every existing reader opens.

    The output frame is whatever ``transform`` inverts into -- pass the run's
    FrameTransform for world coordinates, or an identity transform to stay in the
    normalised frame. TrajectoryExporter calls this once per space, so the caller,
    not this function, decides.

    Args:
        template: Shape with points [1, N, 3], the canonical reference
        sample: Shape with points [1, M, 3], the (augmented) target
        transform: FrameTransform, applied via invert()
        out_dir: directory to write template_points.vtp and sample_points.vtp into
    """
    os.makedirs(out_dir, exist_ok=True)

    # The template's own point data goes with it, so it and deformed.vtp can be
    # coloured by the same array and compared -- the pair is the whole point of
    # writing this file.
    export_shape(template.points[0, :, :], _faces_of(template), transform,
                 os.path.join(out_dir, "template_points.vtp"), label="Template",
                 point_fields=getattr(template, "point_fields", None))

    sample_points = to_frame(sample.points[0, :, :], transform, "Sample")
    save_vtp(create_polydata(sample_points, faces=None),
             os.path.join(out_dir, "sample_points.vtp"), binary=True)

    print(f"[io.export_reference_shapes] Exported template "
          f"({template.points.shape[1]} points) and sample "
          f"({sample_points.shape[0]} points)")


def export_trajectory(traj, faces, transform, out_dir):
    """Export trajectory steps as VTP files in world coordinates (STEPS T15).

    Saves one VTP file per integration step with velocity as a point field.
    Points are denormalized from normalized domain back to world coordinates.

    Trajectories are written as POINT CLOUDS -- no connectivity, ever. The
    exporter is handed the template's faces whether or not the trajectory was
    subsampled, and in the subsampled case those indices address the full mesh,
    so they would join points that are not the ones they name. Rather than
    carrying faces sometimes, the format is the same every time.

    Args:
        traj: Trajectory object with points [K+1, B, N, 3] and velocities [K, B, N, 3]
        faces: accepted for call-site symmetry with the other exporters and
            deliberately unused; see above
        transform: FrameTransform used for normalization (inverted to denormalize)
        out_dir: directory to save step_XXXX.vtp files

    Returns:
        List of saved file paths
    """
    os.makedirs(out_dir, exist_ok=True)

    K_plus_1, B, N, _ = traj.points.shape
    K = K_plus_1 - 1

    saved_paths = []

    for step in range(K_plus_1):
        # Extract points for this step, batch 0: [N, 3]
        points_norm = traj.points[step, 0, :, :]  # [N, 3]

        # Denormalize to world coordinates
        points_world = transform.invert(points_norm)  # [N, 3]

        # Validate points (NaN/Inf crash ParaView)
        if torch.isnan(points_world).any() or torch.isinf(points_world).any():
            print(f"[WARNING] Step {step}: points contain NaN/Inf, replacing with zeros")
            points_world = torch.where(torch.isnan(points_world) | torch.isinf(points_world),
                                       torch.tensor(0.0, device=points_world.device), points_world)

        # Create PolyData without faces (subsampled trajectories have invalid face indices)
        polydata = create_polydata(points_world, faces=None)

        # Add velocity as point field (zero for step 0, actual velocity for steps 1..K)
        if step == 0:
            velocity = torch.zeros(N, 3, dtype=traj.points.dtype)
        else:
            velocity = traj.velocities[step - 1, 0, :, :]  # [N, 3]

        # Validate velocity
        if torch.isnan(velocity).any() or torch.isinf(velocity).any():
            print(f"[WARNING] Step {step}: velocity contains NaN/Inf, replacing with zeros")
            velocity = torch.where(torch.isnan(velocity) | torch.isinf(velocity),
                                  torch.tensor(0.0, device=velocity.device), velocity)

        polydata = add_point_field(polydata, velocity, field_name="velocity")

        # Save as step_XXXX.vtp
        filename = f"step_{step:04d}.vtp"
        filepath = os.path.join(out_dir, filename)
        save_vtp(polydata, filepath, binary=True)
        saved_paths.append(filepath)

    return saved_paths
