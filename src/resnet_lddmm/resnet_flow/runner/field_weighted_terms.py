"""Field-weighted Chamfer terms for a pair and for a cohort."""

import os
import torch
from src.resnet_lddmm.resnet_flow.io import export_shape
from src.resnet_lddmm.resnet_flow.guidance.field_weights import resolve_point_weight_field
from src.resnet_lddmm.resnet_flow.losses import BidirectionalMappingError

from .data import cohort_paths


# Data terms taking the per-direction ChamferSideCfg config shape
# ({source_to_target: {...}, target_to_source: {...}} in data_kwargs) and owning
# their precision internally. Membership here decides three things, and a new
# member that misses any one of them fails SILENTLY rather than loudly:
#   * _add_field_weighted_chamfer resolves field_name/field_source at all
#     (missing -> pred_w/tgt_w stay None, the configured field never applies),
#   * build() forces the loss.sigma-derived composer weight to 1.0
#     (missing -> loss.sigma compounds with the per-direction base_sigmas),
#   * _build_cohort refuses the term (missing -> fields silently ignored there).
FIELD_WEIGHTED_DATA_NAMES = ("field_weighted_chamfer",
                             "field_weighted_point2plane_chamfer")


def _ensure_source_normals(cfg, source_norm):
    """Give the source per-vertex normals, derived from its faces if its file has none.

    The backward half of ``direction: bidirectional`` projects onto the SOURCE's
    tangent planes, as the forward half does onto the target's. Derived here
    rather than demanded from the file: the source is typically a template the
    pipeline rewrites every run (similarity_transform), so an array baked onto it
    by hand would not survive, and a template mesh's face normals are exact.
    joint_normalize's single uniform scale leaves their directions untouched.

    Args:
        cfg: ExperimentCfg, for the file name in messages
        source_norm: the NORMALISED source Shape; gets .normals set IN PLACE when
            it has none

    Raises:
        ValueError: the source has neither normals nor faces, or its normals
            are non-finite.
    """
    if source_norm.normals is None:
        if source_norm.faces is None or source_norm.faces.numel() == 0:
            raise ValueError(
                f"direction='bidirectional' with data_name={cfg.loss.data_name!r} "
                f"needs normals on the source ({cfg.source}), which has neither a "
                "'normal' point array nor faces to derive one from."
            )
        # Local import: geometry pulls in heavier deps than this module's own.
        from src.shared.geometry import mesh_vertex_normals
        normals = mesh_vertex_normals(verts=source_norm.points[0].detach().cpu().numpy(),
                                      faces=source_norm.faces.cpu().numpy())
        source_norm.normals = torch.tensor(normals, dtype=source_norm.points.dtype).unsqueeze(0)
        n_zero = int((source_norm.normals[0].norm(dim=-1) < 1e-6).sum())
        print(f"[field_weighted_chamfer] source normals derived from "
              f"{source_norm.faces.shape[0]} faces ({n_zero} zero-length: no "
              "tangent plane, point-to-point only there)")

    if not torch.isfinite(source_norm.normals).all():
        raise ValueError(
            f"the source's 'normal' array has non-finite rows ({cfg.source}). A NaN "
            "normal silently destroys the whole data term; drop the array to have "
            "it derived from faces, or re-bake it with scripts/write_surface_normals.py."
        )


def _add_field_weighted_chamfer(cfg, source_norm, target_norm, transform, mapping_error, data_term):
    """Resolve each direction's point-weight field (if any) and attach it to mapping_error.

    A no-op whenever cfg.loss.data_name is not in FIELD_WEIGHTED_DATA_NAMES -- the
    field_name/field_source keys only exist in THAT data term's own config shape
    (ChamferSideCfg), so nothing here applies to any other data_name.

    Under ``direction: bidirectional`` it also hands the mapping error the
    mirrored data term for the backward half (see FieldWeightedCDData.mirrored),
    and for point-to-plane makes sure the source has normals.

    Lives here rather than in _build_loss_terms because, like landmark
    resolution, it needs the NORMALISED shapes and cannot run until after
    load_pair -- Registry.create builds ``data_term`` itself earlier, with only
    the raw config dict available.

    Args:
        cfg: ExperimentCfg
        source_norm, target_norm: NORMALISED shapes (point_fields AND faces
            intact -- pass the Shape itself, not a stripped batch wrapper)
        transform: the run's FrameTransform, forwarded to the sanity-check
            export so the visualised mesh lands back in world coordinates
        mapping_error: the already-built mapping error for this run, either
            direction; gets its point-weight fields attached IN PLACE
        data_term: the already-built FieldWeightedCDData, read back only for
            its already-validated field_name/field_source/field_transform/
            field_scale attributes -- avoids re-parsing cfg.loss.data_kwargs a
            second time.
    """
    if cfg.loss.data_name not in FIELD_WEIGHTED_DATA_NAMES:
        return

    bidirectional = isinstance(mapping_error, BidirectionalMappingError)

    # Fail at BUILD time, not at the first training step: the target's normals are
    # a static input, so "the file has none" is knowable before a single gradient
    # step has been paid for.
    if cfg.loss.data_name == "field_weighted_point2plane_chamfer":
        if target_norm.normals is None:
            raise ValueError(
                f"data_name={cfg.loss.data_name!r} needs a 'normal' point array on "
                f"the target ({cfg.target}), but it carries none. Either re-run "
                "scan_fusion (it writes the field itself under compute_normals: "
                "true, on the final post-min_support topology), or bake it onto "
                "the existing file with:\n"
                f"    python scripts/write_surface_normals.py {cfg.target}\n"
                "(the target has faces, so these are exact mesh normals, not "
                "estimated ones). io.load_shape reads the array back automatically "
                "and joint_normalize carries it through unchanged -- its single "
                "uniform scale leaves normal DIRECTIONS untouched."
            )
        # A NaN normal is unrecoverable and does not announce itself until the
        # whole loss goes NaN, some steps in. Both producers guard their own
        # division, so reaching this means the array was written by something
        # else -- catch it here rather than after a wasted run.
        if not torch.isfinite(target_norm.normals).all():
            n_bad = int((~torch.isfinite(target_norm.normals).all(dim=-1)).sum())
            raise ValueError(
                f"the target's 'normal' array has {n_bad} non-finite row(s) "
                f"({cfg.target}). A NaN normal propagates through the entire data "
                "term and silently destroys the run. Re-bake it with "
                "scripts/write_surface_normals.py, which returns degenerate "
                "vertices as the ZERO vector instead (scored as zero plane-distance, "
                "falling back to the point-to-point component)."
            )
        if bidirectional:
            _ensure_source_normals(cfg, source_norm)

    # The mirrored backward half would put the inheritance on the OTHER claim
    # (source points taking the target's field), which is not what was configured.
    if data_term.source_to_target_inherit_field_name is not None:
        raise NotImplementedError(
            f"source_to_target.inherit_target_field: {data_term.source_to_target_inherit_field_name} "
            f"(a named target field) is supported in train.mode='cohort' only; use true "
            f"to inherit the target_to_source field."
        )
    if bidirectional and (data_term.target_to_source_inherit_source_field
                          or data_term.source_to_target_inherit_target_field):
        raise NotImplementedError(
            "target_to_source.inherit_source_field / source_to_target.inherit_target_field "
            "are not supported under direction='bidirectional' yet: the mirrored "
            "backward half has no defined meaning for them. Use direction='forward', "
            "or drop the key."
        )

    if bidirectional:
        mapping_error.set_backward_data_term(data_term.mirrored())
        print("[field_weighted_chamfer] bidirectional: D(phi^-1(target), source) is "
              "scored with the sides mirrored -- each side's settings stay with its claim")

    template_field = resolve_point_weight_field(
        data_term.source_to_target_field_name, data_term.source_to_target_field_source,
        fallback_shape=source_norm,
        field_transform=data_term.source_to_target_field_transform,
        field_scale=data_term.source_to_target_field_scale,
    )
    sample_field = resolve_point_weight_field(
        data_term.target_to_source_field_name, data_term.target_to_source_field_source,
        fallback_shape=target_norm,
        field_transform=data_term.target_to_source_field_transform,
        field_scale=data_term.target_to_source_field_scale,
    )
    mapping_error.set_point_weight_fields(template_field=template_field, sample_field=sample_field)

    # A visual sanity check, same spirit as landmarks_template.vtp: the
    # RESOLVED weight (post-transform, if any), written onto the FULL mesh it
    # came from -- with connectivity, so it opens as an actual surface in
    # ParaView rather than a bare point cloud, colourable by "chamfer_weight".
    for label, shape, field, filename in (
        ("source_to_target (template)", source_norm, template_field, "field_weighted_chamfer_source.vtp"),
        ("target_to_source (target)", target_norm, sample_field, "field_weighted_chamfer_target.vtp"),
    ):
        if field is None:
            print(f"[field_weighted_chamfer] {label}: no field configured (constant)")
            continue
        print(f"[field_weighted_chamfer] {label}: field over {field.shape[0]} "
              f"vertices, range [{field.min().item():.4g}, {field.max().item():.4g}]")
        export_shape(shape.points[0], shape.faces, transform,
                     os.path.join(cfg.output_dir, filename),
                     label=f"FieldWeightedChamfer ({label})",
                     point_fields={"chamfer_weight": field.cpu().numpy()})


def _add_cohort_field_weighted_chamfer(cfg, cohort_norm, template_norm, mapping_error, data_term):
    """_add_field_weighted_chamfer for a cohort: one template field, one field per member.

    In cohort mode the template is cfg.target and the members are cfg.source, so the
    data term's source_to_target side is the TEMPLATE's and its target_to_source side
    each MEMBER's. The template's field is attached to the mapping error as in pair
    mode; each member's field becomes that member's per-vertex weight (times its
    "area" weights, if any), which CohortBatchLoader stacks and the mapping error
    gathers on the same draw as its points -- the same product _combine_weights forms
    in pair mode. A no-op for data terms outside FIELD_WEIGHTED_DATA_NAMES.

    Args:
        cfg: ExperimentCfg in cohort mode
        cohort_norm: the NORMALISED members (point_fields intact); their .weights
            are set IN PLACE
        template_norm: the NORMALISED template
        mapping_error: gets the template field attached IN PLACE
        data_term: the already-built FieldWeightedCDData
    """
    if cfg.loss.data_name not in FIELD_WEIGHTED_DATA_NAMES:
        return

    if cfg.loss.data_name == "field_weighted_point2plane_chamfer":
        for path, member in zip(cohort_paths(cfg), cohort_norm):
            if member.normals is None or not torch.isfinite(member.normals).all():
                raise ValueError(
                    f"data_name={cfg.loss.data_name!r} needs finite 'normal' point arrays "
                    f"on every cohort member; {path} has none or non-finite ones. Bake "
                    "them with scripts/write_surface_normals.py."
                )

    template_field = resolve_point_weight_field(
        data_term.source_to_target_field_name, data_term.source_to_target_field_source,
        fallback_shape=template_norm,
        field_transform=data_term.source_to_target_field_transform,
        field_scale=data_term.source_to_target_field_scale,
    )
    mapping_error.set_point_weight_fields(template_field=template_field, sample_field=None)

    for member in cohort_norm:
        field = resolve_point_weight_field(
            data_term.target_to_source_field_name, data_term.target_to_source_field_source,
            fallback_shape=member,
            field_transform=data_term.target_to_source_field_transform,
            field_scale=data_term.target_to_source_field_scale,
        )
        if field is not None:
            field = field.to(member.points.dtype).unsqueeze(0)
            member.weights = field if member.weights is None else member.weights * field

    match_name = data_term.source_to_target_inherit_field_name
    if match_name is not None:
        for member in cohort_norm:
            field = resolve_point_weight_field(match_name, None, fallback_shape=member)
            member.match_weights = field.to(member.points.dtype).unsqueeze(0)
        print(f"[field_weighted_chamfer] source_to_target inherits {match_name!r} of each member's "
              f"nearest point: range [{min(m.match_weights.min().item() for m in cohort_norm):.4g}, "
              f"{max(m.match_weights.max().item() for m in cohort_norm):.4g}]")

    member_fields = [m.weights for m in cohort_norm if m.weights is not None]
    for label, fields in (("source_to_target (template)", [] if template_field is None else [template_field]),
                          (f"target_to_source ({len(cohort_norm)} members)", member_fields)):
        if not fields:
            print(f"[field_weighted_chamfer] {label}: no field configured (constant)")
        else:
            print(f"[field_weighted_chamfer] {label}: range "
                  f"[{min(f.min().item() for f in fields):.4g}, {max(f.max().item() for f in fields):.4g}]")
