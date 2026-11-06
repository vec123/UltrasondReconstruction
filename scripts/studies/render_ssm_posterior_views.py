"""Render one PNG per case of an SSM posterior fit on a cohort LDDMM run's apply.

Each PNG has, for the frontal view and then for the patient's left and right:
  * Target (fused scan) | LDDMM recon | SSM posterior mean
  * (blank) | the part of the LDDMM recon the posterior trusts: its faces with
    an "informative" vertex (weight at least half the best observation's) |
    the posterior mean coloured by its per-vertex std, light (0) -> dark
    (--std-max mm and above).

Every mesh is shown in the frame of the LDDMM recon's own eval (the
reference-scan landmark frame recon_target_distance writes): the target and the
LDDMM recon from that eval, the posterior and its observations carried there
by the same similarity, fitted from world/deformed.vtp onto the eval's
recon_w_distance.vtp. One camera per view, shared by every panel.

Usage:
    python scripts/studies/render_ssm_posterior_views.py \
        --run outputs/Casos_Mar/resnet_lddmm_cohort_autodecoder_fieldweighted_nopos_kin8000 \
        --cases 14_26_semanas 11_36_semanas
"""

import argparse
import os

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from src.shared.vtk import render as views  # noqa: E402
from src.shared.transforms.similarity import SimilarityTransform  # noqa: E402
from src.shared.vtk.io import load_vtp  # noqa: E402
from src.shared.vtk.process import sub_surface  # noqa: E402

OBSERVED_COLOR = (0.45, 0.65, 0.88)


def points_of(poly):
    return vtk_to_numpy(poly.GetPoints().GetData()).astype(float)


def render_case(case_dir, posterior_name, out_path, std_max, panel_px):
    posterior_dir = os.path.join(case_dir, posterior_name)
    target = load_vtp(os.path.join(case_dir, "eval", "target_w_distance.vtp"))
    lddmm = load_vtp(os.path.join(case_dir, "eval", "recon_w_distance.vtp"))
    ref_lmk = views.read_landmarks(os.path.join(case_dir, "eval", "reference_landmarks.vtp"))

    # world -> the LDDMM eval's landmark frame, exact: same vertices on both sides.
    deformed = load_vtp(os.path.join(case_dir, "world", "deformed.vtp"))
    to_eval = SimilarityTransform.from_correspondences(points_of(deformed), points_of(lddmm))
    residual = np.abs(to_eval.apply_points(points_of(deformed)) - points_of(lddmm)).max()
    if residual > 1e-3:
        raise ValueError(f"{case_dir}: deformed.vtp is not recon_w_distance.vtp up to a similarity ({residual:.3g})")

    posterior = to_eval.apply_polydata(load_vtp(os.path.join(posterior_dir, "posterior_mean.vtp")))
    std = vtk_to_numpy(posterior.GetPointData().GetArray("posterior_std_mm"))
    informative = vtk_to_numpy(posterior.GetPointData().GetArray("informative")).astype(bool)
    observed = sub_surface(lddmm, informative)
    std_mesh = views.with_colors(posterior, views.ramp_colors(std, std_max))

    cameras = {view: views.view_camera(ref_lmk, [target, lddmm, posterior], view)
               for view in ("frontal", "left", "right")}
    rows = []
    for view, camera in cameras.items():
        suffix = "" if view == "frontal" else f" ({view})"
        rows.append((camera, [
            (f"Target (fused scan){suffix}", target, views.SCAN_COLOR, []),
            (f"LDDMM recon{suffix}", lddmm, views.MESH_COLOR, []),
            (f"SSM posterior mean{suffix}", posterior, views.MESH_COLOR, []),
        ]))
        rows.append((camera, [
            None,
            (f"Informative observations: {informative.sum()}/{len(informative)}{suffix}",
             observed, OBSERVED_COLOR, []),
            (f"Posterior std, 0-{std_max:g} mm (median {np.median(std):.2f}){suffix}", std_mesh, None, []),
        ]))

    run = os.path.basename(os.path.dirname(os.path.dirname(case_dir)))
    # Run names are long; the parts that differ between PNGs go first, so a cut keeps them.
    header = f"{posterior_name} - {os.path.basename(case_dir)} - {run}"
    views.write_grid(rows, 3, header, out_path, panel_px)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="The cohort run's output dir (holds apply/<case>/).")
    parser.add_argument("--cases", nargs="*", default=None,
                        help="apply/<case> names (default: every case with a posterior).")
    parser.add_argument("--posterior", required=True,
                        help="Posterior folder inside each case, e.g. posterior_baby-pca_pose_t2r_cutoff1.5mm_sigma3mm "
                             "(ssm_posterior.variant_name).")
    parser.add_argument("--output", default=None, help="Folder for the PNGs (default: <run>/ssm_posterior_views).")
    parser.add_argument("--std-max", type=float, default=2.5, help="Posterior std (mm) at the dark end of the ramp.")
    parser.add_argument("--panel-px", type=int, default=400, help="Width and height of each panel in pixels.")
    args = parser.parse_args()

    apply_dir = os.path.join(args.run, "apply")
    cases = args.cases or sorted(c for c in os.listdir(apply_dir)
                                 if os.path.exists(os.path.join(apply_dir, c, args.posterior, "posterior_mean.vtp")))
    output = args.output or os.path.join(args.run, f"{args.posterior}_views")
    os.makedirs(output, exist_ok=True)
    for case in cases:
        out_path = os.path.join(output, f"{case}.png")
        render_case(os.path.join(apply_dir, case), args.posterior, out_path, args.std_max, args.panel_px)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
