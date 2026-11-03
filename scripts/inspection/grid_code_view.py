"""Show where the codes of a direct-grid cohort run sit and how strongly each is
regularised: one sphere per grid node, coloured by its code-penalty weight
(code.grid_reg, 0..1), over the slightly transparent template. Seen from the
front, the left and the right, each view tilted by TILT_DEG so the nodes behind
each other stay apart.

Distance and region weights depend only on the template, so the view is the same
for every member. Coverage weights are per member: name one with --member, which
also draws that member's target (transparent).

Run with:
    python scripts/inspection/grid_code_view.py \
        --run outputs/Casos_Mar/resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g4c8_dist
"""

import argparse
import json
import os

import numpy as np
import torch
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from scripts.inspection.code_pca_views import target_path  # noqa: E402
from scripts.inspection.render_frontal_views import MESH_COLOR, SCAN_COLOR, mesh_actor, text_actor, view_camera  # noqa: E402
from src.resnet_lddmm.resnet_flow import registrations  # noqa: E402,F401  (registers all components)
from src.resnet_lddmm.resnet_flow.conditioning.grid_regularization import grid_penalty_weights  # noqa: E402
from src.resnet_lddmm.resnet_flow.config import ExperimentCfg  # noqa: E402
from src.resnet_lddmm.resnet_flow.io import load_shape  # noqa: E402
from src.resnet_lddmm.resnet_flow.guidance.landmarks import resolve_template_points  # noqa: E402
from src.resnet_lddmm.resnet_flow.runner import _normalization_domain, cohort_paths, load_cohort  # noqa: E402
from src.shared.vtk.io import load_vtp  # noqa: E402

VIEWS = ("frontal", "left", "right")
TEMPLATE_OPACITY = 0.25
TARGET_OPACITY = 0.35
TILT_DEG = 15.0  # camera turn away from the exact front/side views, in both directions


def load_run(run):
    """The run's ExperimentCfg, as apply reads it (without the provenance 'mode' key)."""
    data = json.load(open(os.path.join(run, "config.json")))
    data.pop("mode", None)
    return ExperimentCfg.from_dict(data)


def node_values(cfg, run, member):
    """World positions [g³, 3] of the grid nodes, the penalty weight per node [g³]
    (the member's, for coverage weights; member may be None otherwise) and the grid
    resolution."""
    if not cfg.code.grid_direct:
        raise ValueError("grid_code_view needs a direct-grid run (code.grid_direct: true)")
    g, channels = cfg.code.grid, cfg.code.grid_channels
    _, cohort_norm, template_norm, transform = load_cohort(cfg)
    domain = _normalization_domain(cfg)

    axis = torch.linspace(domain[0], domain[1], g, dtype=transform.center.dtype)
    nodes = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    world = transform.invert(nodes.to(transform.center.device)).cpu().numpy()

    weights = grid_penalty_weights(cfg.code, [m.points[0] for m in cohort_norm], template_norm.points[0],
                                   lambda labels: resolve_template_points(template_norm, labels), domain)
    weights = weights[:, ::channels].cpu().numpy()
    if cfg.code.grid_reg.kind == "coverage":
        if member is None:
            raise ValueError("coverage weights are per member: name one with --member")
        members = [os.path.splitext(os.path.basename(p))[0] for p in cohort_paths(cfg)]
        return world, weights[members.index(member)], g
    return world, weights[0], g  # the same row for every member


def own_landmarks(template_path):
    """{label: point} of the template, from its own heatmap_<label> fields (world frame)."""
    shape = load_shape(template_path)
    labels = [name[len("heatmap_"):] for name in shape.point_fields
              if name.startswith("heatmap_") and name != "heatmap_landmarks"]
    return dict(zip(labels, resolve_template_points(shape, labels)))


def node_glyphs(world, values, value_range, radius):
    """Spheres at the nodes, coloured by `values` on a light-to-dark blue scale."""
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(world.astype(np.float64), deep=True))
    poly.SetPoints(points)
    scalars = numpy_to_vtk(values.astype(np.float64), deep=True)
    scalars.SetName("value")
    poly.GetPointData().SetScalars(scalars)
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(16)
    sphere.SetPhiResolution(16)
    glyphs = vtk.vtkGlyph3D()
    glyphs.SetInputData(poly)
    glyphs.SetSourceConnection(sphere.GetOutputPort())
    glyphs.ScalingOff()
    glyphs.SetColorModeToColorByScalar()
    glyphs.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(glyphs.GetOutput())
    mapper.SetLookupTable(blue_ramp())
    mapper.SetScalarRange(*value_range)
    mapper.ScalarVisibilityOn()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor, mapper


def blue_ramp():
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    light, dark = np.array([0.87, 0.92, 0.97]), np.array([0.03, 0.19, 0.42])
    for i in range(256):
        lut.SetTableValue(i, *(light + (dark - light) * i / 255), 1.0)
    lut.Build()
    return lut


def grid_lines(world, g):
    """Thin lines along the grid's three axes."""
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(world.astype(np.float64), deep=True))
    poly.SetPoints(points)
    lines = vtk.vtkCellArray()
    index = np.arange(g ** 3).reshape(g, g, g)
    for along in (index, index.transpose(1, 0, 2), index.transpose(2, 0, 1)):
        for line in along.reshape(g, -1).T:  # every line of g nodes along one axis
            lines.InsertNextCell(g, [int(n) for n in line])
    poly.SetLines(lines)
    actor = mesh_actor(poly, (0.6, 0.6, 0.6))
    actor.GetProperty().SetOpacity(0.5)
    return actor


def transparent(poly, color, opacity):
    actor = mesh_actor(poly, color)
    actor.GetProperty().SetOpacity(opacity)
    return actor


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Direct-grid cohort run folder.")
    parser.add_argument("--member", default=None,
                        help="Member whose target is drawn; required for coverage weights, which are per member.")
    parser.add_argument("--output", default=None,
                        help="PNG to write (default: <run>/grid_weights.png, or grid_weights_<member>.png).")
    parser.add_argument("--panel-px", type=int, default=500, help="Width and height of each panel in pixels.")
    args = parser.parse_args()

    cfg = load_run(args.run)
    world, weight, g = node_values(cfg, args.run, args.member)
    template = load_vtp(cfg.target)
    target = load_vtp(target_path(args.run, args.member)) if args.member else None
    lmk = own_landmarks(cfg.target)
    spacing = np.linalg.norm(world[1] - world[0])

    nodes_poly = vtk.vtkPolyData()
    nodes_poly.SetPoints(vtk.vtkPoints())
    nodes_poly.GetPoints().SetData(numpy_to_vtk(world.astype(np.float64), deep=True))
    cameras = [view_camera(lmk, [template, nodes_poly], view) for view in VIEWS]
    for camera in cameras:
        # Tilted a little, so the nodes behind each other along the view do not
        # overlap, and nothing clipped: the nodes in front of the face must show.
        camera.Azimuth(TILT_DEG)
        camera.Elevation(TILT_DEG)
        camera.OrthogonalizeViewUp()
        camera.SetParallelScale(camera.GetParallelScale() * 1.15)
        camera.SetClippingRange(1.0, 5000.0)

    header_px, panel = 40, args.panel_px
    width, height = panel * len(VIEWS), header_px + panel
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(width, height)
    header = vtk.vtkRenderer()
    header.SetViewport(0, 1 - header_px / height, 1, 1)
    header.SetBackground(1, 1, 1)
    header.AddViewProp(text_actor(
        f"{os.path.basename(os.path.normpath(args.run))}: code-penalty weight per node, {g}x{g}x{g} grid, "
        f"grid_reg {cfg.code.grid_reg.kind}; template (beige)"
        + (f", target {args.member} (blue)" if args.member else ""),
        16, 0.005, 0.85))
    window.AddRenderer(header)

    top = 1 - header_px / height
    for c, (camera, view) in enumerate(zip(cameras, VIEWS)):
        renderer = vtk.vtkRenderer()
        renderer.SetViewport(c / len(VIEWS), 0, (c + 1) / len(VIEWS), top)
        renderer.SetBackground(0.97, 0.97, 0.97)
        renderer.SetActiveCamera(camera)
        renderer.AddActor(transparent(template, MESH_COLOR, TEMPLATE_OPACITY))
        if target is not None:
            renderer.AddActor(transparent(target, SCAN_COLOR, TARGET_OPACITY))
        renderer.AddActor(grid_lines(world, g))
        glyphs, mapper = node_glyphs(world, weight, (0.0, 1.0), radius=0.08 * spacing)
        renderer.AddActor(glyphs)
        renderer.AddViewProp(text_actor(view, 14, 0.02, 0.98))
        if c == len(VIEWS) - 1:
            bar = vtk.vtkScalarBarActor()
            bar.SetLookupTable(mapper.GetLookupTable())
            bar.SetNumberOfLabels(3)
            bar.SetWidth(0.08)
            bar.SetHeight(0.5)
            bar.SetPosition(0.9, 0.25)
            bar.GetLabelTextProperty().SetColor(0.1, 0.1, 0.1)
            bar.GetLabelTextProperty().SetFontSize(12)
            renderer.AddViewProp(bar)
        window.AddRenderer(renderer)

    window.Render()
    grabber = vtk.vtkWindowToImageFilter()
    grabber.SetInput(window)
    grabber.Update()
    out_path = args.output or os.path.join(
        args.run, f"grid_weights_{args.member}.png" if args.member else "grid_weights.png")
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(out_path)
    writer.SetInputConnection(grabber.GetOutputPort())
    writer.Write()
    window.Finalize()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
