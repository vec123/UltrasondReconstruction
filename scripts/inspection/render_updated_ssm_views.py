"""Render PNGs of a model updated by scripts/SSM/update_ssm_em.py or update_code_em.py: draws, and fits.

generations.png   the base model and the updated model: mean, then --n-samples
                  random draws of each; frontal, then the patient's left and right
fit_<case>[_heldout].png
                  one per case: Target (fused scan) | LDDMM recon | base model fit |
                  updated model fit, frontal, left and right; under the frontal
                  row, the target coloured by its distance to each fit, light (0)
                  -> dark (--distance-max mm and above)

Next to each PNG, a folder of the same name holds its meshes as .vtp, in the
frame the PNG shows them in: generations/ base_mean, base_draw_<k>,
updated_mean, updated_draw_<k> (mm, model frame); fit_<case>[_heldout]/
target (point fields distance_to_base_fit_mm, distance_to_updated_fit_mm),
lddmm_recon, base_fit, updated_fit (the case's eval frame).

A fit is the posterior mean given the observations, pose and noise the update
used (the settings in its em_summary.json). It is shown in the LDDMM recon's
eval frame, as in render_ssm_posterior_views.py. The header gives the gain in
log marginal likelihood (mm) of the updated fit over the base fit. The cases
default to --n-cases spread over that gain, lowest first. With --run, the cases
come from another run that deformed the same base mean; a case the update did
not use is held out, and its PNG says so.

For an update_code_em.py output (a code prior of a frozen flow, no model.npz),
the draws and fits are the ones it wrote (generations/, fits/), drawn the same way.

Usage:
    python scripts/inspection/render_updated_ssm_views.py \
        --model-dir outputs/UpdatedSSM/<run>/<variant>
"""

import argparse
import json
import os

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from scripts.inspection import render_ssm_posterior_views as post  # noqa: E402  (also lets a panel take color=None)
from scripts.SSM.update_ssm_em import load_cases  # noqa: E402
from src.SSM.model.model_memory_io import load_model_file  # noqa: E402
from src.SSM.model.posterior import posterior, posterior_with_pose  # noqa: E402
from src.transforms.similarity import SimilarityTransform  # noqa: E402
from src.vtk.create import create_polydata  # noqa: E402
from src.vtk.fields import add_point_fields  # noqa: E402
from src.vtk.io import load_vtp, save_vtp  # noqa: E402

views = post.views
VIEWS = ("frontal", "left", "right")


def save_meshes(out_path, meshes):
    """Each ``{name: polydata}`` as ``<out_path without .png>/<name>.vtp``."""
    folder = os.path.splitext(out_path)[0]
    os.makedirs(folder, exist_ok=True)
    for name, mesh in meshes.items():
        save_vtp(mesh, os.path.join(folder, f"{name}.vtp"))


def model_landmarks(model, shape):
    points = np.asarray(shape).reshape(-1, 3)
    return {name: points[v] for name, v in zip(model.landmarks["names "], model.landmarks["verts "])}


def render_generations(base, updated, mm, n_samples, seed, header, out_path, panel_px):
    meshes = {}
    for label, model in (("Base", base), ("Updated", updated)):
        shapes = np.concatenate([model.mean[None], model.sample(n_samples, rng=seed)]) * mm
        meshes[label] = [create_polydata(shape.reshape(-1, 3), faces=model.faces) for shape in shapes]
    render_generation_grid(meshes, model_landmarks(base, base.mean * mm), header, out_path, panel_px)


def render_generation_grid(meshes, lmk, header, out_path, panel_px):
    """``meshes`` {"Base": [mean, draw 1, ...], "Updated": [...]} (mm, one frame), ``lmk`` for the camera."""
    n_samples = len(meshes["Base"]) - 1
    rows = []
    for view in VIEWS:
        camera = views.view_camera(lmk, meshes["Base"] + meshes["Updated"], view)
        suffix = "" if view == "frontal" else f" ({view})"
        for label, row in meshes.items():
            titles = [f"{label} mean{suffix}"] + [f"{label} draw {k + 1}{suffix}" for k in range(n_samples)]
            rows.append((camera, [(title, mesh, views.MESH_COLOR, []) for title, mesh in zip(titles, row)]))
    views.write_grid(rows, n_samples + 1, header, out_path, panel_px)
    save_meshes(out_path, {f"{label.lower()}_{name}": mesh for label, row in meshes.items()
                           for name, mesh in zip(["mean"] + [f"draw_{k + 1}" for k in range(n_samples)], row)})


def fit_world(model, case, settings):
    """Posterior mean of ``model`` given the case's observations, in the case's world
    frame, and the log marginal likelihood of the observations in mm."""
    if settings["pose"] == "none":
        result = posterior(model, case.observations)
    else:
        result = posterior_with_pose(model, case.observations, with_scale=settings["pose"] == "similarity",
                                     iterations=settings["pose_iterations"])
    points = case.to_model.inverse().apply_points(result.posed(result.mean).reshape(-1, 3))
    length = case.mm * (result.pose.scale if result.pose is not None else 1.0)
    lml_mm = result.log_marginal_likelihood - 3 * case.observations.n * np.log(length)
    return create_polydata(points, faces=model.faces), lml_mm


def distance_to(points, surface):
    """(P,) distance of each point of ``points`` to the surface ``surface``."""
    distance = vtk.vtkDistancePolyDataFilter()
    distance.SetInputData(0, points)
    distance.SetInputData(1, surface)
    distance.SignedDistanceOff()
    distance.ComputeSecondDistanceOff()
    distance.Update()
    return vtk_to_numpy(distance.GetOutput().GetPointData().GetArray("Distance")).astype(float)


def render_fit(case, case_dir, models, settings, distance_max, header, out_path, panel_px):
    """``header`` is formatted with ``gain``; returns the target-to-fit medians and the gain."""
    worlds, lml = {}, {}
    for label, model in models.items():
        worlds[label], lml[label] = fit_world(model, case, settings)
    gain = lml["Updated"] - lml["Base"]
    return render_fit_worlds(case_dir, worlds, distance_max, header.format(gain=gain), out_path, panel_px), gain


def render_fit_worlds(case_dir, worlds, distance_max, header, out_path, panel_px):
    """The fit PNG of {label: fit in the case's world frame}; returns the target-to-fit medians."""
    target = load_vtp(os.path.join(case_dir, "eval", "target_w_distance.vtp"))
    lddmm = load_vtp(os.path.join(case_dir, "eval", "recon_w_distance.vtp"))
    ref_lmk = views.read_landmarks(os.path.join(case_dir, "eval", "reference_landmarks.vtp"))
    deformed = load_vtp(os.path.join(case_dir, "world", "deformed.vtp"))
    # world -> the LDDMM eval's landmark frame, exact: same vertices on both sides.
    to_eval = SimilarityTransform.from_correspondences(post.points_of(deformed), post.points_of(lddmm))

    fits = {}
    for label, world in worlds.items():
        mesh = to_eval.apply_polydata(world)
        fits[label] = (mesh, distance_to(target, mesh))

    rows = []
    for view in VIEWS:
        camera = views.view_camera(ref_lmk, [target, lddmm] + [mesh for mesh, _ in fits.values()], view)
        suffix = "" if view == "frontal" else f" ({view})"
        rows.append((camera, [(f"Target (fused scan){suffix}", target, views.SCAN_COLOR, []),
                              (f"LDDMM recon{suffix}", lddmm, views.MESH_COLOR, [])]
                     + [(f"{label} model fit{suffix}", mesh, views.MESH_COLOR, [])
                        for label, (mesh, _) in fits.items()]))
        if view == "frontal":
            rows.append((camera, [None, None] + [
                (f"Target to {label.lower()} fit, 0-{distance_max:g} mm (median {np.median(d):.2f})",
                 post.with_colors(target, post.ramp_colors(d, distance_max)), None, [])
                for label, (_, d) in fits.items()]))
    views.write_grid(rows, 4, header, out_path, panel_px)
    target_out = vtk.vtkPolyData()
    target_out.DeepCopy(target)
    add_point_fields(target_out, {f"distance_to_{label.lower()}_fit_mm": d for label, (_, d) in fits.items()})
    save_meshes(out_path, {"target": target_out, "lddmm_recon": lddmm,
                           **{f"{label.lower()}_fit": mesh for label, (mesh, _) in fits.items()}})
    return {label: float(np.median(d)) for label, (_, d) in fits.items()}


def render_code_em(model_dir, summary, output, args):
    """An update_code_em.py output: its generations/ and fits/ as written, drawn as for a linear model."""
    settings = summary["settings"]
    variant = os.path.basename(os.path.normpath(model_dir))
    base = load_model_file(settings["model"])
    mm = summary["mm_per_model_unit"]
    folder = os.path.join(model_dir, "generations")
    n_samples = len([f for f in os.listdir(folder) if f.startswith("base_draw_")])
    meshes = {label: [load_vtp(os.path.join(folder, f"{label.lower()}_{name}.vtp"))
                      for name in ["mean"] + [f"draw_{k + 1}" for k in range(n_samples)]]
              for label in ("Base", "Updated")}
    out_path = os.path.join(output, "generations.png")
    render_generation_grid(meshes, model_landmarks(base, base.mean * mm),
                           f"Base vs updated code prior: mean and {n_samples} random draws (seed {settings['seed']}) "
                           f"- {variant}", out_path, args.panel_px)
    print(f"Saved {out_path}")

    with open(os.path.join(model_dir, "fits", "fits.json")) as handle:
        fits = json.load(handle)
    for name, info in fits.items():
        held_out = info["status"] == "held out"
        gain = info["lml_mm_updated"] - info["lml_mm_base"]
        worlds = {label: load_vtp(os.path.join(model_dir, "fits", name, f"{label.lower()}_fit.vtp"))
                  for label in ("Base", "Updated")}
        out_path = os.path.join(output, f"fit_{name}{'_heldout' if held_out else ''}.png")
        header = (f"{name} ({info['status']}) - log-likelihood gain {gain:.1f} - base vs updated code prior fit - "
                  f"{os.path.basename(os.path.normpath(info['run']))} - {variant}")
        medians = render_fit_worlds(os.path.join(info["run"], "apply", name), worlds, args.distance_max, header,
                                    out_path, args.panel_px)
        print(f"Saved {out_path} ({info['status']}, log-likelihood gain {gain:.1f}, target-to-fit median: "
              + ", ".join(f"{label.lower()} {value:.2f} mm" for label, value in medians.items()) + ")")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True,
                        help="update_ssm_em.py output (model.npz, em_summary.json), or update_code_em.py output "
                             "(em_summary.json, generations/, fits/: drawn as written, other options ignored)")
    parser.add_argument("--cases", nargs="*", default=None, help="case names (default: spread over the gain)")
    parser.add_argument("--cases-json", default=None,
                        help="JSON with a 'cases' list, e.g. outputs/Casos_Mar/challenging_cases.json")
    parser.add_argument("--n-cases", type=int, default=6)
    parser.add_argument("--run", default=None,
                        help="run to take the cases from (default: the update's own run)")
    parser.add_argument("--n-samples", type=int, default=5, help="random draws per model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distance-max", type=float, default=3.0, help="mm at the dark end of the ramp")
    parser.add_argument("--output", default=None, help="Folder for the PNGs (default: <model-dir>/views).")
    parser.add_argument("--panel-px", type=int, default=400, help="Width and height of each panel in pixels.")
    args = parser.parse_args()

    with open(os.path.join(args.model_dir, "em_summary.json")) as handle:
        summary = json.load(handle)
    settings = summary["settings"]
    variant = os.path.basename(os.path.normpath(args.model_dir))
    output = args.output or os.path.join(args.model_dir, "views")
    os.makedirs(output, exist_ok=True)
    if not os.path.exists(os.path.join(args.model_dir, "model.npz")):
        if not os.path.exists(os.path.join(args.model_dir, "fits", "fits.json")):
            raise FileNotFoundError(f"{args.model_dir} has neither model.npz (update_ssm_em.py) nor "
                                    f"fits/fits.json (update_code_em.py)")
        render_code_em(args.model_dir, summary, output, args)
        return

    base = load_model_file(settings["base_model"])
    updated = load_model_file(os.path.join(args.model_dir, "model.npz"))

    out_path = os.path.join(output, "generations.png")
    render_generations(base, updated, summary["mm_per_model_unit"], args.n_samples, args.seed,
                       f"Base vs updated: mean and {args.n_samples} random draws (seed {args.seed}) - {variant}",
                       out_path, args.panel_px)
    print(f"Saved {out_path}")

    run = args.run or settings["run"]
    in_update = os.path.normpath(run) == os.path.normpath(settings["run"])
    gains = {c["name"]: c["lml_mm_new"] - c["lml_mm_base"] for c in summary["cases"]}
    names = args.cases
    if args.cases_json:
        with open(args.cases_json) as handle:
            names = (names or []) + json.load(handle)["cases"]
    if not names:
        ranked = sorted(gains, key=gains.get)
        picks = np.unique(np.linspace(0, len(ranked) - 1, min(args.n_cases, len(ranked))).round().astype(int))
        names = [ranked[i] for i in picks]

    loaded = load_cases(argparse.Namespace(
        run=run, error_direction=settings["error_direction"], noise=settings["noise"]["name"],
        sigma=settings["noise"]["sigma"], threshold=settings["noise"]["threshold"],
        spacing=settings["spacing"]), base, names)
    by_name = {case.name: case for case in loaded}
    models = {"Base": base, "Updated": updated}
    for name in names:
        held_out = not (in_update and name in gains)
        status = "held out" if held_out else "in update"
        out_path = os.path.join(output, f"fit_{name}{'_heldout' if held_out else ''}.png")
        header = (f"{name} ({status}) - log-likelihood gain {{gain:.1f}} - base vs updated fit - "
                  f"{os.path.basename(os.path.normpath(run))} - {variant}")
        medians, gain = render_fit(by_name[name], os.path.join(run, "apply", name), models, settings,
                                   args.distance_max, header, out_path, args.panel_px)
        print(f"Saved {out_path} ({status}, log-likelihood gain {gain:.1f}, target-to-fit median: "
              + ", ".join(f"{label.lower()} {value:.2f} mm" for label, value in medians.items()) + ")")


if __name__ == "__main__":
    main()
