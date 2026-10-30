"""Wraps the flow with the optional parts: local mask, symmetry and deformation modes."""

import os
import numpy as np
import torch
from src.resnet_lddmm.resnet_flow.io import export_shape
from src.resnet_lddmm.resnet_flow.landmarks import resolve_template_points
from src.resnet_lddmm.resnet_flow.fields.composite.modes import network_of, symmetric_field_of


def _place_local_mask(cfg, source_norm, transform, flow):
    """Place time_varying_local's region: centred on the template's own landmarks.

    The centre is the mean of ``field.local.landmarks`` on the NORMALISED
    template (heatmap argmax, as everywhere else), the radii are converted from
    world units (mm here) with the run's FrameTransform scale. Training only:
    apply restores the placement from the checkpoint's buffers instead.

    Args:
        cfg: ExperimentCfg
        source_norm: the NORMALISED source Shape, point_fields intact
        transform: the run's FrameTransform (normalised = (world - center) / scale)
        flow: the built flow; its field's mask is placed IN PLACE
    """
    if cfg.field.kind != "time_varying_local":
        return

    local = cfg.field.local
    points = resolve_template_points(source_norm, local["landmarks"], local["heatmap_prefix"])
    center = torch.tensor(points.mean(axis=0), dtype=source_norm.points.dtype)
    scale = float(transform.scale)
    r_in = local["radius_mm"] / scale
    r_out = (local["radius_mm"] + local["falloff_mm"]) / scale
    mask = network_of(flow.field).mask
    mask.place(center, r_in, r_out)

    with torch.no_grad():
        values = mask(source_norm.points[0])[:, 0]
    world_center = transform.invert(center[None])[0].tolist()
    print(f"[local_field] region centred on mean of {local['landmarks']} at "
          f"{[round(c, 2) for c in world_center]} (world); full strength within "
          f"{local['radius_mm']:g} mm, zero beyond {local['radius_mm'] + local['falloff_mm']:g} mm; "
          f"template vertices: {(values >= 1.0).float().mean().item():.1%} full, "
          f"{(values > 0).float().mean().item():.1%} touched")

    # Same spirit as landmarks_template.vtp: where the fine field acts, on the mesh.
    export_shape(source_norm.points[0], source_norm.faces, transform,
                 os.path.join(cfg.output_dir, "local_field_mask.vtp"),
                 label="LocalRefinementField mask (template)",
                 point_fields={"local_mask": values.cpu().numpy()})


def _place_symmetry(cfg, template_norm, transform, flow):
    """Place field.symmetry's mirror plane from the template's own landmarks, in the training frame.

    The normal is the mean direction right -> left over field.symmetry.pairs; the
    plane passes through the mean of the pairs' midpoints and the midline landmarks
    (heatmap argmax, as everywhere else). Prints how well the landmarks and the
    template itself fit the plane, as a fraction of the template's size (the world
    unit is mm for scans, but not for a shape model's frame). Training only: apply
    restores the plane from the checkpoint's buffers.

    Args:
        cfg: ExperimentCfg
        template_norm: the NORMALISED flowed shape (pair: source, cohort: template),
            point_fields intact
        transform: the run's FrameTransform (normalised = (world - center) / scale)
        flow: the built flow; its SymmetricField's plane is placed IN PLACE
    """
    symmetry = cfg.field.symmetry
    if symmetry is None:
        return
    prefix = symmetry["heatmap_prefix"]
    right = resolve_template_points(template_norm, [p[0] for p in symmetry["pairs"]], prefix)
    left = resolve_template_points(template_norm, [p[1] for p in symmetry["pairs"]], prefix)
    directions = left - right
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    normal = directions.mean(axis=0)
    normal /= np.linalg.norm(normal)
    on_plane = 0.5 * (left + right)
    if symmetry["midline"]:
        on_plane = np.concatenate([on_plane, resolve_template_points(template_norm, symmetry["midline"], prefix)])
    offset = float((on_plane @ normal).mean())
    field = symmetric_field_of(flow.field)
    field.plane.place(torch.as_tensor(normal), offset)

    angles = np.degrees(np.arccos(np.clip(directions @ normal, -1.0, 1.0)))
    with torch.no_grad():
        points = template_norm.points[0]
        size = float((points.max(dim=0).values - points.min(dim=0).values).norm())
        sample = points[torch.randperm(len(points), generator=torch.Generator().manual_seed(0), device="cpu")[:4000]]
        mirrored = field.plane.reflect_points(sample.to(field.plane.normal))
        gap = torch.cat([torch.cdist(chunk, points.to(chunk)).min(dim=1).values
                         for chunk in mirrored.split(500)]) / size
    print(f"[symmetry] mode {symmetry['mode']}"
          + (f", symmetric part at {symmetry['discount']:g} x kinetic_weight" if symmetry["mode"] == "split" else "")
          + f"; mirror plane from {len(symmetry['pairs'])} pair(s) and {len(symmetry['midline'])} midline "
          f"landmark(s), normal {[round(float(c), 3) for c in normal]}: pair directions within "
          f"{angles.max():.2f} deg of it, landmarks {np.abs(on_plane @ normal - offset).max() / size:.2%} "
          f"of the template diagonal from the plane at most; mirrored template to template: mean "
          f"{gap.mean():.2%}, 95% {gap.quantile(0.95):.2%} of the diagonal")


def _place_modes(cfg, source_norm, transform, flow):
    """Place field.modes' designed fields on the source, in the training frame.

    The modes are compiled on the template their spec was drawn on and carried
    onto the source by the similarity that maps one onto the other -- the source
    must be a posed copy of that template, as similarity_transform writes (see
    src/deformation_modes/transport.py). Each field is scaled so its peak over
    the source is 1 world unit (mm) per unit time, so an amplitude reads as mm.
    Training only: apply restores the fields from the checkpoint's buffers.

    Args:
        cfg: ExperimentCfg
        source_norm: the NORMALISED source Shape
        transform: the run's FrameTransform (normalised = (world - center) / scale)
        flow: the built flow; its ModalField's basis is placed IN PLACE

    Raises:
        ValueError / SpecError: the source is not a posed copy of the spec's
            template, or the spec lacks a named mode.
    """
    modes_cfg = cfg.field.modes
    if modes_cfg is None:
        return
    # Local import: compiling pulls in the geodesic solver, which only mode runs need.
    from src.deformation_modes import transport_modes

    field = flow.field
    scale = float(transform.scale)
    vertices = source_norm.points[0].detach().cpu().double().numpy()
    transported, similarity = transport_modes(modes_cfg["spec"], modes_cfg["template"],
                                              modes_cfg["names"], vertices, unit=1.0 / scale)
    field.basis.place([mode.expansion for mode in transported])

    with torch.no_grad():
        values = field.basis(source_norm.points[:1].to(field.amplitude.dtype))[0]   # [N, M, 3]
    print(f"[modes] {len(transported)} designed mode(s) from {os.path.basename(modes_cfg['spec'])}, "
          f"carried from {os.path.basename(modes_cfg['template'])} onto the source (scale "
          f"{similarity.scale:.4g}, fit residual {similarity.rms:.2g}); kinetic price "
          f"{modes_cfg['discount']:g} x kinetic_weight, amplitude lr {modes_cfg['lr']:g}")
    fields = {}
    for m, mode in enumerate(transported):
        speed = values[:, m].norm(dim=-1) * scale                                     # mm per unit amplitude
        print(f"[modes]   {mode.name}: {len(mode.expansion.centers)} kernel centres, sigma "
              f"{mode.expansion.sigma * scale:.3g} mm; moves {(speed > 0.1).float().mean().item():.1%} "
              f"of the source by > 10% of its peak")
        for warning in mode.warnings:
            print(f"[modes]   {mode.name}: spec warning - {warning}")
        fields[f"mode_{mode.name}"] = (values[:, m] * scale).cpu().numpy()
        fields[f"mode_{mode.name}_speed"] = speed.cpu().numpy()

    # Same spirit as local_field_mask.vtp: each mode's field (mm per unit amplitude) on the source.
    export_shape(source_norm.points[0], source_norm.faces, transform,
                 os.path.join(cfg.output_dir, "deformation_modes_source.vtp"),
                 label="Designed deformation modes (template)", point_fields=fields)
