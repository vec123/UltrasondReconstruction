"""Assembles the loss terms of a registration: landmarks, kinetic field, fshape and per-part kinetic terms."""

import os
import numpy as np
import torch
from src.models.lddmm.resnet_flow.config import FshapeCfg, PointWeightFieldCfg, SHARED_DISCOUNT_ONCE
from src.models.lddmm.resnet_flow.losses.fshape import build_fshape_kwargs
from src.models.lddmm.resnet_flow.io import export_shape
from src.models.lddmm.resnet_flow.guidance.landmarks import build_landmark_term_kwargs, resolve_template_points
from src.models.lddmm.resnet_flow.guidance.field_weights import normalize_unit_mean, resolve_point_weight_field
from src.models.lddmm.learning.registry import Registry
from src.models.lddmm.learning.losses.composer import LossTerm


def _build_loss_terms(loss_cfg):
    """cfg.loss.terms -> (composer entries, {name: term module}).

    Zero-weight entries are dropped rather than computed and multiplied by zero,
    matching how isometry has always been skipped when disabled.

    Args:
        loss_cfg: LossCfg with a .terms list of {kind, weight, name, kwargs} dicts

    Returns:
        (list of LossTerm, dict mapping metric name -> instantiated term)

    Raises:
        ValueError: on a missing 'kind', or a name colliding with another term
    """
    specs = list(loss_cfg.terms)

    # First-class weights, configured like every other term (isometry_weight,
    # kinetic_weight, ...). Strictly > 0 to be built at all: at 0 the term is never
    # instantiated and its extra work never runs, rather than being computed and
    # multiplied by zero. A new gated term is one entry here.
    for prefix, kind in (("equivariant_deformation", "equivariant_deformation_loss"),
                         ("pose_supervision", "pose_supervision_loss"),
                         ("flow_rotation_penalty", "flow_rotation_penalty")):
        weight = float(getattr(loss_cfg, f"{prefix}_weight", 0.0) or 0.0)
        if weight > 0:
            specs.append({
                "kind": kind,
                "weight": weight,
                "kwargs": dict(getattr(loss_cfg, f"{prefix}_kwargs", None) or {}),
            })

    entries, modules = [], {}
    for spec in specs:
        if "kind" not in spec:
            raise ValueError(f"loss term needs a 'kind': {spec}")
        name = spec.get("name", spec["kind"])
        if name in modules:
            raise ValueError(
                f"duplicate loss term name {name!r}; give one of them an explicit "
                f"'name' so their metrics stay distinguishable"
            )
        weight = float(spec.get("weight", 1.0))
        if weight <= 0:
            continue
        # Registry.create raises with the list of valid names on an unknown kind.
        modules[name] = Registry.create("loss_term", spec["kind"], **spec.get("kwargs", {}))
        entries.append(LossTerm(name, weight=weight))
    return entries, modules


def _add_landmark_term(cfg, template_shape, transform, composer, loss_terms):
    """Resolve + attach the landmark-guidance term, if cfg.loss.landmark_weight > 0.

    Lives here rather than in ``_build_loss_terms`` because it needs data that
    only exists AFTER shapes are loaded -- the template's ``point_fields`` and
    the FrameTransform both come out of ``load_pair``/``joint_normalize``, which
    runs later than ``_build_loss_terms`` in ``build()``. See landmarks.py and
    ``losses.terms.LandmarkGuidanceLoss``.

    Args:
        cfg: ExperimentCfg
        template_shape: the NORMALISED template Shape (must still carry
            ``point_fields`` -- pass the Shape itself, not a stripped batch
            wrapper such as ``_build_pair``'s ``SimpleBatch``)
        transform: the FrameTransform training runs in
        composer: LossComposer already built; a "landmark" LossTerm is appended
            to it in place when enabled
        loss_terms: {name: module} dict already built by ``_build_loss_terms``

    Returns:
        ``loss_terms`` unchanged when disabled; otherwise a NEW dict (the
        caller's dict is not mutated) with a "landmark" entry added.

    Raises:
        ValueError: ``loss.landmark_weight > 0`` but ``cfg.landmarks`` is unset.
    """
    if cfg.loss.landmark_weight <= 0:
        return loss_terms

    if not cfg.landmarks:
        raise ValueError(
            "loss.landmark_weight > 0 but no `landmarks:` folder is configured; "
            "set `landmarks: <path to landmark CSVs>` or drop landmark_weight."
        )

    kwargs = build_landmark_term_kwargs(cfg, template_shape, transform, **cfg.loss.landmark_kwargs)
    labels = kwargs.pop("labels")

    loss_terms = dict(loss_terms or {})
    loss_terms["landmark"] = Registry.create("loss_term", "landmark_guidance", **kwargs)
    composer.terms.append(LossTerm("landmark", weight=cfg.loss.landmark_weight))

    # A quick visual sanity check -- did the argmax-resolved template vertices
    # land on the right anatomy? Written once, at build time, next to the other
    # reference shapes so it can be opened alongside template_points.vtp.
    export_shape(kwargs["template_points"][0], None, transform,
                 os.path.join(cfg.output_dir, "landmarks_template.vtp"),
                 label="Landmarks (template)")

    print(f"[landmarks] guiding on {len(labels)} labels: {', '.join(labels)}")
    return loss_terms


def _add_kinetic_field_term(cfg, source_norm, transform, composer, loss_terms):
    """Resolve + attach the point-weighted kinetic term, if cfg.loss.kinetic_field_weight > 0.

    Post-load for the same reason as ``_add_landmark_term``: the weight field is
    a point field on the NORMALISED source, which does not exist when
    ``_build_loss_terms`` runs. See losses/kinetic.py for the term itself.

    Args:
        cfg: ExperimentCfg
        source_norm: the NORMALISED source Shape, point_fields intact
        transform: the FrameTransform training runs in, for the sanity export
        composer: LossComposer; a "kinetic_field" LossTerm is appended in place
            when enabled
        loss_terms: {name: module} dict already built

    Returns:
        ``loss_terms`` unchanged when disabled; otherwise a NEW dict with a
        "kinetic_field" entry added.

    Raises:
        ValueError: invalid kinetic_field_kwargs, a missing/misshapen field, or a
            configured term already named "kinetic_field".
    """
    if cfg.loss.kinetic_field_weight <= 0:
        return loss_terms

    name = "kinetic_field"
    if loss_terms and name in loss_terms:
        raise ValueError(f"a loss term is already named {name!r}; rename it so the "
                         f"two metrics stay distinguishable")

    field_cfg = PointWeightFieldCfg.from_dict(cfg.loss.kinetic_field_kwargs,
                                              what="loss.kinetic_field_kwargs")
    vertex_weights = resolve_point_weight_field(
        field_cfg.field_name, field_cfg.field_source, fallback_shape=source_norm,
        field_transform=field_cfg.field_transform, field_scale=field_cfg.field_scale,
        dtype=source_norm.points.dtype,
    )
    raw_min, raw_max = vertex_weights.min().item(), vertex_weights.max().item()
    if field_cfg.normalize:
        vertex_weights = normalize_unit_mean(vertex_weights, label=f"kinetic field {field_cfg.field_name!r}")

    loss_terms = dict(loss_terms or {})
    loss_terms[name] = Registry.create("loss_term", "point_weighted_kinetic",
                                       vertex_weights=vertex_weights)
    composer.terms.append(LossTerm(name, weight=cfg.loss.kinetic_field_weight))

    lo, hi = vertex_weights.min().item(), vertex_weights.max().item()
    print(f"[kinetic_field] {field_cfg.field_name!r} over {vertex_weights.shape[0]} "
          f"vertices: raw range [{raw_min:.4g}, {raw_max:.4g}], used range "
          f"[{lo:.4g}, {hi:.4g}] (ratio {hi / max(lo, 1e-12):.3g}), "
          f"weight {cfg.loss.kinetic_field_weight}, on top of kinetic_weight "
          f"{cfg.loss.kinetic_weight}")
    if cfg.loss.direction == "bidirectional":
        print("[kinetic_field] bidirectional: weights the FORWARD trajectory only; "
              "the backward one starts on the target, which the field does not "
              "describe, and keeps the plain kinetic term.")

    # Same spirit as field_weighted_chamfer_source.vtp: the weight actually used,
    # on the full source mesh, colourable by "kinetic_weight" in ParaView.
    export_shape(source_norm.points[0], source_norm.faces, transform,
                 os.path.join(cfg.output_dir, "kinetic_field_source.vtp"),
                 label="PointWeightedKinetic (template)",
                 point_fields={"kinetic_weight": vertex_weights.detach().cpu().numpy()})
    return loss_terms


def _add_fshape_term(cfg, source_norm, target_norm, transform, composer, loss_terms):
    """Resolve + attach the windowed fshape data term, if cfg.loss.fshape_weight > 0.

    Post-load for the same reason as the landmark and kinetic-field terms: it
    needs the NORMALISED meshes, their faces and the template's heatmaps. See
    losses/fshape.py for the term itself.

    Args:
        cfg: ExperimentCfg
        source_norm, target_norm: NORMALISED Shapes with faces and point_fields
        transform: the run's FrameTransform (mm <-> normalised, sanity exports)
        composer: LossComposer; an "fshape" LossTerm is appended in place
        loss_terms: {name: module} dict already built

    Returns:
        ``loss_terms`` unchanged when disabled, else a NEW dict with "fshape".

    Raises:
        ValueError: missing faces, invalid fshape_kwargs, a name clash, or a
            configuration the term cannot honour (augmentation or pose: it
            compares against the fixed, unaugmented target).
    """
    if cfg.loss.fshape_weight <= 0:
        return loss_terms

    name = "fshape"
    if loss_terms and name in loss_terms:
        raise ValueError(f"a loss term is already named {name!r}")
    if cfg.augmentation.kind != "none" or getattr(cfg, "use_encoder_pose", False):
        raise ValueError("loss.fshape_weight > 0 compares the deformed template with the FIXED, "
                         "unaugmented target; it needs augmentation.kind: none and no pose.")
    for label, shape in (("source", source_norm), ("target", target_norm)):
        if shape.faces is None or len(shape.faces) == 0:
            raise ValueError(f"loss.fshape_weight > 0 needs triangles on the {label}, which has none")

    fcfg = FshapeCfg.from_dict(dict(cfg.loss.fshape_kwargs))
    scale = float(transform.scale)
    center = resolve_template_points(source_norm, fcfg.landmarks, fcfg.heatmap_prefix).mean(axis=0)
    kwargs, info = build_fshape_kwargs(
        source_norm.points[0].detach().cpu().numpy().astype(np.float64), source_norm.faces.cpu().numpy(),
        target_norm.points[0].detach().cpu().numpy().astype(np.float64), target_norm.faces.cpu().numpy(),
        center=center, r_in=fcfg.radius_mm / scale, r_out=(fcfg.radius_mm + fcfg.falloff_mm) / scale,
        margin=fcfg.margin_mm / scale, sigmas=[s / scale for s in fcfg.sigma_mm],
        n_faces=int(fcfg.n_faces), seed=int(fcfg.seed), signal=fcfg.signal,
        sigma_signal=fcfg.sigma_signal, curvature_radius_mm=fcfg.curvature_radius_mm,
        scale=scale, dtype=source_norm.points.dtype, signal_normalize=fcfg.signal_normalize,
    )

    loss_terms = dict(loss_terms or {})
    term = Registry.create("loss_term", "fshape_data", **kwargs)
    loss_terms[name] = term
    composer.terms.append(LossTerm(name, weight=cfg.loss.fshape_weight))

    with torch.no_grad():
        initial = term.distance(term.template_vertices[None]).item()
    kind = "varifold" if fcfg.signal == "none" else f"fshape ({fcfg.signal})"
    print(f"[fshape] {kind}: window {fcfg.radius_mm:g}+{fcfg.falloff_mm:g} mm around {fcfg.landmarks}; "
          f"template {info['template_faces']} of {info['template_crop_faces']} cropped triangles, "
          f"target {info['target_faces']} of {info['target_window_faces']}; sigma_mm {fcfg.sigma_mm}; "
          f"relative distance at identity {initial:.4g} (x weight {cfg.loss.fshape_weight:g} "
          f"= {initial * cfg.loss.fshape_weight:.4g})")
    if fcfg.signal != "none":
        s_med, s_95, t_med, t_95 = info["signal_stats"]
        print(f"[fshape] {fcfg.signal} (1/mm, r={fcfg.curvature_radius_mm:g} mm): template median "
              f"{s_med:.3g} p95 {s_95:.3g}, target median {t_med:.3g} p95 {t_95:.3g}; compared as "
              f"{'percentiles' if fcfg.signal_normalize == 'rank' else 'raw values'}, sigma_signal "
              f"{info['sigma_signal']:.3g}; undetermined vertices set flat {info['missing']}"
              f"{'; target sign flipped to match orientation' if info['sign_flipped'] else ''}")

    # What the term actually compares, colourable by "fshape_signal" in ParaView.
    for label, (points, faces, signal) in (("template", info["template_export"]),
                                           ("target", info["target_export"])):
        fields = None if signal is None else {"fshape_signal": np.asarray(signal, dtype=np.float32)}
        export_shape(torch.as_tensor(points, dtype=source_norm.points.dtype), torch.as_tensor(faces).long(),
                     transform, os.path.join(cfg.output_dir, f"fshape_{label}.vtp"),
                     label=f"Fshape ({label})", point_fields=fields)
    return loss_terms


def priced_parts(cfg, num_members=None) -> list:
    """The flow's priced parts: (metric name, part name, discount), one per configured part.

    The part names are the keys fields/base.py split_velocity returns. ``num_members``
    is the cohort size M, which field.shared.discount "once" (= 1/M) needs.
    """
    parts = []
    if cfg.field.modes is not None:
        parts.append(("mode_kinetic", "modes", cfg.field.modes["discount"]))
    if cfg.field.prior is not None:
        parts.append(("prior_kinetic", "prior", cfg.field.prior["discount"]))
    if cfg.field.symmetry is not None and cfg.field.symmetry["mode"] == "split":
        parts.append(("symmetric_kinetic", "symmetric", cfg.field.symmetry["discount"]))
    if cfg.field.shared is not None:
        parts.append(("shared_kinetic", "shared", _shared_discount(cfg.field.shared["discount"], num_members)))
    return parts


def _shared_discount(discount, num_members) -> float:
    """field.shared.discount as a number: itself, or 1/M for "once".

    Raises:
        ValueError: "once" without a cohort size.
    """
    if discount != SHARED_DISCOUNT_ONCE:
        return discount
    if not num_members:
        raise ValueError(f"field.shared.discount {SHARED_DISCOUNT_ONCE!r} is 1/M and needs the "
                         f"number of cohort members M")
    return 1.0 / num_members


def _add_part_kinetic_terms(cfg, composer, loss_terms, num_members=None):
    """Charge each priced part's kinetic energy at ``discount * kinetic_weight``.

    The priced parts are the designed modes (field.modes), the frozen prior field
    (field.prior), the symmetric part (field.symmetry, mode split) and the shared
    flow (field.shared; ``num_members`` resolves its discount "once").
    ``Trajectory.kinetic_energy`` leaves them out, so without these terms they would
    be free. A part is dropped like any zero-weight term when its price is 0.

    Returns:
        ``loss_terms`` unchanged without priced parts; else a NEW dict with
        "mode_kinetic", "prior_kinetic", "symmetric_kinetic" and/or "shared_kinetic".

    Raises:
        ValueError: a configured term already has one of those names.
    """
    parts = priced_parts(cfg, num_members)
    if not parts:
        return loss_terms
    loss_terms = dict(loss_terms or {})
    for name, part, discount in parts:
        if name in loss_terms:
            raise ValueError(f"a loss term is already named {name!r}; rename it so the "
                             f"two metrics stay distinguishable")
        weight = discount * cfg.loss.kinetic_weight
        if weight <= 0:
            print(f"[{part}] discount 0: moving along the {part} part is free")
            continue
        print(f"[{part}] kinetic price: discount {discount:.6g} * kinetic_weight = {weight:.6g}")
        loss_terms[name] = Registry.create("loss_term", "part_kinetic", part=part)
        composer.terms.append(LossTerm(name, weight=weight))
    return loss_terms
