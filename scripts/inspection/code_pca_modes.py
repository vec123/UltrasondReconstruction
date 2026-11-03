"""Show what the first principal components of a cohort auto-decoder's codes do to
the face: decode synthetic codes at the code mean plus -2 .. +2 standard deviations
along PC1 and PC2, deform the cohort template with each, and render the results.

Each component gets a frontal, a left and a right row, its columns going from -2 to +2 SD,
all with one common camera. A third row per component colours the face by how far
each point moved from the mean code's face, signed along its outward normal
(blue: inward, red: outward), so subtle changes stay visible.

With --region (direct-grid runs only, code.grid_direct) the PCA uses only the
code entries of the grid nodes around the named landmarks, e.g. the nose, and the
other entries stay at their mean, so the modes are that region's own.

Run with:
    python scripts/inspection/code_pca_modes.py \
        --apply-config configs/resnet_lddmm/cohort_autodecoder_casos_mar_fieldweighted_nopos_apply.yaml
    python scripts/inspection/code_pca_modes.py --region n,prn,sn,aL,aR,acL,acR \
        --apply-config configs/resnet_lddmm/cohort_autodecoder_casos_mar_fieldweighted_direct_g4c8_apply.yaml
"""

import argparse
import os

import numpy as np
import torch
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from scripts.inspection import render_frontal_views as views  # noqa: E402
from scripts.inspection.code_pca_views import similarity_map, template_landmarks  # noqa: E402
from src.resnet_lddmm.resnet_flow import registrations  # noqa: E402,F401  (registers all components)
from src.resnet_lddmm.resnet_flow.apply.client import DeformationClient  # noqa: E402
from src.shared.core.pipeline import no_pipeline_context  # noqa: E402
from src.shared.core.config import interpolate_paths, path_variables  # noqa: E402
from src.shared.core.config import load_yaml  # noqa: E402
from src.resnet_lddmm.resnet_flow.apply.config import ApplyCfg, load_training_config  # noqa: E402
from src.resnet_lddmm.resnet_flow.callbacks.checkpoints import load_checkpoint, resolve_checkpoint  # noqa: E402
from src.resnet_lddmm.resnet_flow.io import load_shape  # noqa: E402
from src.resnet_lddmm.resnet_flow.runner import _normalization_domain, load_cohort, set_default_dtype  # noqa: E402
from src.shared.vtk.io import load_vtp  # noqa: E402

COMPONENTS = (0, 1)
STEPS = (-2.0, -1.0, 0.0, 1.0, 2.0)
DISPLACEMENT_RANGE_MM = 3.0  # colour scale limit of the displacement rows


def mode_codes(codes, steps, entries=None):
    """{(component, step): code} at the code mean + step * SD along each component,
    and the variance ratio of each component. With `entries` (code indices) the
    PCA runs on those entries only and the others stay at their mean."""
    mean = codes.mean(axis=0)
    entries = np.arange(codes.shape[1]) if entries is None else entries
    _, s, vt = np.linalg.svd(codes[:, entries] - mean[entries], full_matrices=False)
    sd = s / np.sqrt(len(codes) - 1)
    synthetic = {}
    for k in COMPONENTS:
        for step in steps:
            code = mean.copy()
            code[entries] += step * sd[k] * vt[k]
            synthetic[(k, step)] = code
    return synthetic, s ** 2 / np.sum(s ** 2)


def region_entries(train_cfg, transform, lmk, labels):
    """Code indices of the direct grid's nodes at the corners of the cells that hold
    the landmarks `labels`, and those nodes' (i, j, k) grid indices."""
    if not train_cfg.code.grid_direct:
        raise ValueError("--region needs a direct-grid run (code.grid_direct: true)")
    g, channels = train_cfg.code.grid, train_cfg.code.grid_channels
    lo, hi = _normalization_domain(train_cfg)
    points = torch.as_tensor(np.stack([lmk[label] for label in labels]),
                             dtype=transform.center.dtype, device=transform.center.device)
    t = ((transform.apply(points) - lo) / (hi - lo) * (g - 1)).cpu().numpy()
    cells = np.clip(np.floor(t), 0, g - 2).astype(int)
    nodes = sorted({tuple(int(v) for v in c + np.array(d)) for c in cells for d in np.ndindex(2, 2, 2)})
    entries = np.concatenate([np.arange(channels) + ((i * g + j) * g + k) * channels for i, j, k in nodes])
    return entries, nodes


def decode(apply_config, steps, region=None, extra=None):
    """{(component, step): world points [N, 3]}, the template polydata, the run
    folder, the variance ratios and, with `region` (landmark labels), its grid nodes.

    `extra` ({key: function of the [M, n_z] codes -> one code}) decodes further
    codes into the same dict under their own keys, e.g. the zero code."""
    # As run_stage.py reads a stage config: drop the stage name, fill ${PROJECT_ROOT} etc.
    raw = load_yaml(apply_config)
    raw.pop("stage", None)
    apply_cfg = ApplyCfg.from_dict(interpolate_paths(raw, path_variables(),
                                                     resolve_ref=no_pipeline_context))
    train_cfg = load_training_config(apply_cfg)
    ref = resolve_checkpoint(apply_cfg.checkpoint, apply_cfg.step)
    cfg = apply_cfg.resolved(train_cfg, ref)
    set_default_dtype(train_cfg)

    _, _, _, transform = load_cohort(train_cfg)
    checkpoint = load_checkpoint(ref, map_location=cfg.device)
    codes = checkpoint["codes"]["codes.weight"].detach().cpu().numpy()
    run = os.path.dirname(os.path.normpath(apply_cfg.train_config))
    entries = nodes = None
    if region:
        names = sorted(n for n in os.listdir(os.path.join(run, "apply")) if os.path.isdir(os.path.join(run, "apply", n)))
        entries, nodes = region_entries(train_cfg, transform, template_landmarks(run, names[0]), region)
    synthetic, ratio = mode_codes(codes, steps, entries)
    synthetic.update({key: make(codes) for key, make in (extra or {}).items()})

    template = load_shape(cfg.template)
    points = {}
    for key, code in synthetic.items():
        client = DeformationClient.from_checkpoint(
            train_cfg, checkpoint, transform,
            code=torch.as_tensor(code, dtype=torch.get_default_dtype(), device=cfg.device)[None],
            pose=(None, None), device=cfg.device, chunk_size=cfg.chunk_size,
        )
        deformation = client.deform(template)
        points[key] = transform.invert(deformation.points.to(transform.center.device)).detach().cpu().numpy()
    return points, load_vtp(cfg.template), run, ratio, nodes


def with_points(template, points, scalars=None):
    """A copy of the template mesh with its points replaced, optionally carrying scalars."""
    poly = vtk.vtkPolyData()
    poly.DeepCopy(template)
    poly.GetPoints().SetData(numpy_to_vtk(points.astype(np.float64), deep=True))
    if scalars is not None:
        array = numpy_to_vtk(scalars.astype(np.float64), deep=True)
        array.SetName("displacement")
        poly.GetPointData().SetScalars(array)
    return poly


def normals_of(poly):
    filt = vtk.vtkPolyDataNormals()
    filt.SetInputData(poly)
    filt.ComputePointNormalsOn()
    filt.SplittingOff()
    filt.ConsistencyOn()
    filt.Update()
    return vtk_to_numpy(filt.GetOutput().GetPointData().GetNormals())


def heat_actor(poly):
    """The mesh coloured by its 'displacement' scalars on a blue-white-red scale."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetTableRange(-DISPLACEMENT_RANGE_MM, DISPLACEMENT_RANGE_MM)
    for i in range(256):
        t = i / 255 * 2 - 1
        white = 1 - abs(t)
        pole = (0.13, 0.40, 0.67) if t < 0 else (0.70, 0.09, 0.17)
        lut.SetTableValue(i, *[white + (1 - white) * c for c in pole], 1.0)
    lut.Build()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(-DISPLACEMENT_RANGE_MM, DISPLACEMENT_RANGE_MM)
    mapper.ScalarVisibilityOn()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply-config", required=True, help="The run's cohort apply .yaml.")
    parser.add_argument("--output", default=None, help="PNG to write (default: <run>/code_pca_modes.png).")
    parser.add_argument("--panel-px", type=int, default=320, help="Width and height of each panel in pixels.")
    parser.add_argument("--region", default=None,
                        help="Comma-separated landmark labels; PCA over their grid nodes only (direct grid).")
    parser.add_argument("--region-name", default=None,
                        help="Short name of the region for the file name and header (default: the labels).")
    parser.add_argument("--similarity", default=None,
                        help="similarity.json mapping the run's world frame to mm (a shape-model-frame run, "
                             "see scripts/SSM/target_to_ssm_frame.py); default: world is mm already.")
    args = parser.parse_args()

    region = args.region.split(",") if args.region else None
    points, template, run, ratio, nodes = decode(args.apply_config, STEPS, region)
    to_mm = similarity_map(args.similarity)
    points = {key: to_mm(p) for key, p in points.items()}
    names = sorted(n for n in os.listdir(os.path.join(run, "apply")) if os.path.isdir(os.path.join(run, "apply", n)))
    lmk = template_landmarks(run, names[0], to_mm)

    meshes = {key: with_points(template, p) for key, p in points.items()}
    frontal, left, right = (views.view_camera(lmk, list(meshes.values()), view) for view in ("frontal", "left", "right"))

    # The displacement rows reuse add_panel's mesh slot with an already coloured actor.
    plain_actor = views.mesh_actor
    def mesh_or_heat(poly, color):
        return heat_actor(poly) if poly.GetPointData().GetScalars() is not None else plain_actor(poly, color)
    views.mesh_actor = mesh_or_heat

    rows = []
    for k in COMPONENTS:
        mean_points = points[(k, 0.0)]
        normals = normals_of(meshes[(k, 0.0)])
        titles = [f"PC{k + 1} {step:+.0f} SD" for step in STEPS]
        rows.append((frontal, [(t, meshes[(k, s)], views.MESH_COLOR, []) for t, s in zip(titles, STEPS)]))
        for side in (left, right):
            rows.append((side, [(t, meshes[(k, s)], views.MESH_COLOR, []) for t, s in zip(titles, STEPS)]))
        heat = [with_points(template, points[(k, s)], np.sum((points[(k, s)] - mean_points) * normals, axis=1))
                for s in STEPS]
        rows.append((frontal, [(f"{t}: normal shift", h, views.MESH_COLOR, []) for t, h in zip(titles, heat)]))

    region_name = args.region_name or "_".join(region or [])
    suffix = f"_region_{region_name}" if region else ""
    out_path = args.output or os.path.join(run, f"code_pca_modes{suffix}.png")
    scope = f"{region_name}: {len(nodes)} nodes" if region else "whole code"
    header = (f"{os.path.basename(run)} ({scope}): mean + n SD, PC1 {100 * ratio[0]:.1f} %, PC2 "
              f"{100 * ratio[1]:.1f} %; shift: blue in, red out, +-{DISPLACEMENT_RANGE_MM:g} mm")
    views.write_grid(rows, len(STEPS), header, out_path, args.panel_px)
    if nodes:
        print(f"Region nodes (i, j, k): {nodes}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
        main()
