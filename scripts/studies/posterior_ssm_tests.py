"""Sanity test of the ssm_posterior stage on partial observations the model itself generates.

Each test draws a ground-truth shape from the model, puts it in a mm frame
with a random pose (rotation, scale, translation - as a registration with its
own pose would leave it), adds iid noise and observes only part of it:

    face_center   everything within 35 mm of the nose tip
    right_half    the patient's right half of the face (as a side-on scan)
    lower_face    mouth and chin
    upper_face    eyes and forehead
    sparse_face   20 % of the face's vertices (as a target_to_recon pick)

The observations are written as the stage's inputs (template_vtp,
observed_vtp, error_vtp) and fitted by the stage itself, once per --sigmas,
so this checks exactly the procedure the pipeline runs. With no model mismatch
and iid noise, a tight sigma should recover the truth: implausible shapes or
extreme coefficients here would point at the procedure, not at the data.

Per test, output/<test>/ holds the inputs, ground_truth.vtp and one stage output
folder per sigma, named by ssm_posterior.variant_name; output/<test>.png shows,
from the front, left and right:
ground truth | observations | posterior mean per sigma, plus a frontal row with
the ground truth coloured by observed (blue) / free and each posterior mean
coloured by its distance to the truth. output/summary.json
has the numbers.

Run with:
    python scripts/studies/posterior_ssm_tests.py
"""

import argparse
import io
import json
import os
from contextlib import redirect_stdout

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from src.shared.vtk import render as views  # noqa: E402
from src.shared.vtk.process import sub_surface  # noqa: E402
from src.shared.vtk.render import ramp_colors, with_colors  # noqa: E402
from src.shared.core.config import from_dict  # noqa: E402
from scripts.pipeline.registry import STAGE_REGISTRY  # noqa: E402
from src.models.gpmm.stages.ssm_posterior import variant_name  # noqa: E402
from src.models.gpmm.model.model_memory_io import load_model_file  # noqa: E402
from src.shared.transforms.similarity import SimilarityTransform  # noqa: E402
from src.shared.vtk.create import create_polydata  # noqa: E402
from src.shared.vtk.fields import add_point_field  # noqa: E402
from src.shared.vtk.io import load_vtp, save_vtp  # noqa: E402

DEFAULT_MODEL = "assets/shape_models/baby/baby_pca.npz"
MM_PER_UNIT = 5335.0  # the baby models' frame against the mm templates
UNOBSERVED_ERROR = 10.0
ERROR_MAX_MM = 3.0


TESTS = ("face_center", "right_half", "lower_face", "upper_face", "sparse_face")


def region(name, points, lmk, rng):
    """(N,) bool: which vertices of one shape (mm) test `name` observes."""
    def ball(label, radius):
        return np.linalg.norm(points - lmk[label], axis=1) < radius

    if name == "face_center":
        return ball("prn", 35.0)
    if name == "right_half":
        right = views.face_frame(lmk)[1][0]  # screen-right of the frontal view, i.e. the patient's left
        return ball("prn", 45.0) & ((points - lmk["prn"]) @ right < 5.0)
    if name == "lower_face":
        return ball("ls", 25.0)
    if name == "upper_face":
        return ball("n", 30.0)
    if name == "sparse_face":
        return ball("prn", 40.0) & (rng.random(len(points)) < 0.2)
    raise ValueError(f"unknown test region '{name}'")


def random_pose(rng, centre):
    """s R (x - centre) + centre + t: 3-9 deg about a random axis, s in [0.9, 1], |t| ~ 2 mm."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(rng.uniform(3.0, 9.0))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    scale = rng.uniform(0.9, 1.0)
    matrix = np.eye(4)
    matrix[:3, :3] = scale * R
    matrix[:3, 3] = centre - scale * R @ centre + rng.normal(scale=2.0 / np.sqrt(3), size=3)
    return SimilarityTransform(matrix), np.degrees(angle), scale


def mesh(points, faces, fields=None):
    poly = create_polydata(np.asarray(points, dtype=np.float64), faces=faces)
    for name, values in (fields or {}).items():
        add_point_field(poly, np.asarray(values), name)
    return poly


def run_stage(model_path, test_dir, sigma, threshold):
    """The ssm_posterior stage on one test's inputs; returns its output dir."""
    spec = STAGE_REGISTRY.get("ssm_posterior")
    cfg = from_dict(spec.config_cls, {
        "shape_model_mat": model_path,
        "template_vtp": os.path.join(test_dir, "template.vtp"),
        "observed_vtp": os.path.join(test_dir, "observed.vtp"),
        "error_vtp": os.path.join(test_dir, "observed.vtp"),
        "error_array": "synthetic_error",
        "noise": {"name": "cutoff", "kwargs": {"sigma": sigma, "threshold": threshold}},
        "pose": "similarity",
    })
    out = cfg.output_dir = os.path.join(test_dir, variant_name(cfg))
    with redirect_stdout(io.StringIO()):
        spec.run(cfg)
    return out


def main(ARGS):
    model = load_model_file(ARGS.model)
    rng = np.random.default_rng(ARGS.seed)
    faces = model.faces
    lam = np.clip(model.eigenvalues, 0.0, None)
    names, idx = model.landmark_names(), model.landmark_indices()

    template_points = model.mean.reshape(-1, 3) * MM_PER_UNIT
    os.makedirs(ARGS.output, exist_ok=True)
    summary = {"model": ARGS.model, "noise_mm": ARGS.noise_mm, "sigmas_mm": ARGS.sigmas, "tests": {}}

    # One truth, pose and noise draw per test; the region is the test's own.
    for t, name in enumerate(TESTS):
        alpha = rng.standard_normal(model.n_components) * np.sqrt(lam)
        truth = (model.mean + model.eigenfunctions @ alpha).reshape(-1, 3) * MM_PER_UNIT
        pose, angle, scale = random_pose(rng, truth.mean(axis=0))
        posed_truth = pose.apply_points(truth)
        observed_points = posed_truth + rng.normal(scale=ARGS.noise_mm, size=truth.shape)

        lmk = {n: posed_truth[i] for n, i in zip(names, idx)}
        mask = region(name, posed_truth, lmk, rng)

        test = f"test_{t:02d}_{name}"
        test_dir = os.path.join(ARGS.output, test)
        os.makedirs(test_dir, exist_ok=True)
        save_vtp(mesh(template_points, faces), os.path.join(test_dir, "template.vtp"))
        save_vtp(mesh(posed_truth, faces), os.path.join(test_dir, "ground_truth.vtp"))
        save_vtp(mesh(observed_points, faces, {"synthetic_error": np.where(mask, 0.0, UNOBSERVED_ERROR),
                                                "observed": mask.astype(np.uint8)}),
                 os.path.join(test_dir, "observed.vtp"))

        # What knowing the pose but nothing of the shape would give, on the free side.
        mean_posed = pose.apply_points(template_points)
        entry = {"region": name, "n_observed": int(mask.sum()),
                 "true_pose": {"rotation_deg": float(angle), "scale": float(scale)},
                 "truth_mahalanobis_sq": float(np.sum(alpha ** 2 / np.where(lam > 0, lam, np.inf))),
                 "n_components": int(model.n_components),
                 "mean_shape_error_free_median_mm": float(np.median(
                     np.linalg.norm(mean_posed - posed_truth, axis=1)[~mask])),
                 "fits": {}}

        panels, errors_row = {}, []
        for sigma in ARGS.sigmas:
            out = run_stage(ARGS.model, test_dir, sigma, threshold=1.0)
            fit = json.load(open(os.path.join(out, "posterior_summary.json")))
            post = load_vtp(os.path.join(out, "posterior_mean.vtp"))
            post_points = vtk_to_numpy(post.GetPoints().GetData()).astype(float)
            error = np.linalg.norm(post_points - posed_truth, axis=1)
            entry["fits"][f"{sigma:g}"] = {
                "found_pose": fit["pose"],
                "mahalanobis_sq": fit["mahalanobis_sq"],
                "error_observed_median_mm": float(np.median(error[mask])),
                "error_free_median_mm": float(np.median(error[~mask])),
                "error_free_p95_mm": float(np.percentile(error[~mask], 95)),
                "posterior_std_free_median_mm": float(np.median(
                    vtk_to_numpy(post.GetPointData().GetArray("posterior_std_mm"))[~mask])),
            }
            f = entry["fits"][f"{sigma:g}"]
            panels[sigma] = (f"Posterior, sigma {sigma:g} mm: Maha^2 {f['mahalanobis_sq']:.0f}", post)
            errors_row.append((f"Error, sigma {sigma:g}: free median {f['error_free_median_mm']:.2f} mm",
                               with_colors(post, ramp_colors(error, ERROR_MAX_MM)), None, []))
        summary["tests"][test] = entry

        truth_mesh = load_vtp(os.path.join(test_dir, "ground_truth.vtp"))
        observed_mesh = sub_surface(load_vtp(os.path.join(test_dir, "observed.vtp")), mask)
        cameras = {view: views.view_camera(lmk, [truth_mesh], view) for view in ("frontal", "left", "right")}
        rows = []
        for view, camera in cameras.items():
            suffix = "" if view == "frontal" else f" ({view})"
            rows.append((camera, [
                (f"Ground truth (Maha^2 {entry['truth_mahalanobis_sq']:.0f}){suffix}", truth_mesh, views.MESH_COLOR, []),
                (f"Observations: {mask.sum()}/{len(mask)}{suffix}", observed_mesh, views.SCAN_COLOR, []),
            ] + [(title + suffix, poly, views.MESH_COLOR, []) for title, poly in panels.values()]))
        rgb = np.where(mask[:, None], views.SCAN_COLOR, views.MESH_COLOR) * 255
        truth_observed = with_colors(truth_mesh, rgb.round().astype(np.uint8))
        rows.append((cameras["frontal"], [("Ground truth: observed (blue) / free", truth_observed, None, []),
                                          None] + errors_row))
        header = (f"{test}: pose {angle:.1f} deg, scale {scale:.3f}, noise {ARGS.noise_mm:g} mm; "
                  f"error light 0 -> dark {ERROR_MAX_MM:g} mm")
        png = os.path.join(ARGS.output, f"{test}.png")
        views.write_grid(rows, 2 + len(ARGS.sigmas), header, png, ARGS.panel_px)

        fits = "  ".join(f"sigma {s}: pose {v['found_pose']['rotation_deg']:.1f} deg/{v['found_pose']['scale']:.3f}, "
                         f"Maha^2 {v['mahalanobis_sq']:.0f}, err obs/free {v['error_observed_median_mm']:.2f}/"
                         f"{v['error_free_median_mm']:.2f} mm"
                         for s, v in entry["fits"].items())
        print(f"{test} ({mask.sum()} obs, truth Maha^2 {entry['truth_mahalanobis_sq']:.0f}, pose {angle:.1f} deg/"
              f"{scale:.3f}, mean-shape free err {entry['mean_shape_error_free_median_mm']:.2f} mm)\n    {fits}")

    with open(os.path.join(ARGS.output, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved {ARGS.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Prior model .npz.")
    parser.add_argument("--output", default="outputs/posterior_SSM_tests", help="Folder for the tests.")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.5, 3.0], help="Cutoff sigma(s) to fit with (mm).")
    parser.add_argument("--noise-mm", type=float, default=0.3, help="iid noise std added to the observations (mm).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--panel-px", type=int, default=360, help="Width and height of each panel in pixels.")
    main(parser.parse_args())
