"""Compare code fits of one real scan into an SSM-trained flow (e.g. a regularisation ablation).

Each fit is a freeze_flow run (one member: the scan in the SSM frame, written by
scripts/SSM/target_to_ssm_frame.py). For each fit, the fitted code is decoded on the full
template, carried back to the scan's frame (mm) with the similarity of
target_to_ssm_frame.py, and compared with the scan:

  * scan -> fit: distance of each scan point to the nearest fit vertex (mean, support-
    weighted mean, 95th percentile) - how well the scan is explained;
  * fit -> scan: distance of each fit vertex to the nearest scan point, weighted by the
    template's landmark_weight (the face region the data term looks at) - how far the
    face leaves the scan;
  * the code: its norm and squared Mahalanobis distance under a Gaussian of the
    training codes (training codes: about n_z on average), and the kinetic energy of
    the last logged step.

The mean code (start of every fit) is the reference row. Writes <fit-root>/comparison.json
and <fit-root>/views/comparison.png: the scan, the mean code and each fit as columns,
frontal, left and right rows.

Run with:
    python scripts/inspection/compare_ssm_code_fits.py \\
        --fit-root outputs/baby_ssm_cohort/fit_16_26 \\
        --train-run outputs/baby_ssm_cohort/resnet_lddmm_cohort_autodecoder_n5000_global_l2_cont20k --train-step 38000 \\
        --scan-mesh outputs/Casos_Mar/cohort/fused_targets/16_26_semanas.vtp
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import torch
from scipy.spatial import cKDTree
from vtk.util.numpy_support import vtk_to_numpy

from src.shared.vtk import render as views  # noqa: E402
from scripts.inspection.eval_ssm_cohort import decode, load_run  # noqa: E402
from src.models.gpmm.model.model_memory_io import load_model_file  # noqa: E402
from src.shared.vtk.create import create_polydata  # noqa: E402
from src.shared.vtk.io import load_vtp  # noqa: E402

SCAN_COLOR = views.SCAN_COLOR


def last_kinetic(log_path):
    """Kinetic energy of the last 'Step' line of a run log, or None."""
    if not os.path.exists(log_path):
        return None
    steps = [line for line in open(log_path) if line.startswith("Step")]
    match = re.search(r"kinetic: ([0-9.e+-]+)", steps[-1]) if steps else None
    return float(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fit-root", required=True, help="folder with targets/, template.vtp, similarity.json, the fit runs")
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--train-step", type=int, default=None)
    parser.add_argument("--scan-mesh", default=None,
                        help="meshed version of the scan in the same frame, for the render (default: the scan)")
    parser.add_argument("--runs", nargs="+", default=None, help="fit runs to include (default: all under --fit-root)")
    parser.add_argument("--extra", nargs="+", default=[],
                        help="label=path of other methods' results for the scan, in the scan's frame (mm)")
    parser.add_argument("--name", default="comparison", help="base name of the .json and the views/ .png")
    parser.add_argument("--model", default="assets/shape_models/baby/baby_pca.npz")
    parser.add_argument("--panel-px", type=int, default=240)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)

    root = args.fit_root
    with open(os.path.join(root, "similarity.json")) as f:
        sim = json.load(f)
    scale, rot, trans = sim["scale"], np.array(sim["rotation"]), np.array(sim["translation"])

    def to_mm(points):
        return scale * points @ rot.T + trans

    scan_name = os.path.basename(sorted(glob.glob(os.path.join(root, "targets", "*.vtp")))[0])
    scan_mm = load_vtp(next(p for p in sim["scans"] if os.path.basename(p) == scan_name))
    scan_pts = vtk_to_numpy(scan_mm.GetPoints().GetData()).astype(np.float64)
    support = vtk_to_numpy(scan_mm.GetPointData().GetArray("support")).astype(np.float64)
    template = load_vtp(os.path.join(root, "template.vtp"))
    face_w = vtk_to_numpy(template.GetPointData().GetArray("landmark_weight")).astype(np.float64)
    scan_tree = cKDTree(scan_pts)

    train_codes = torch.load(os.path.join(args.train_run, "checkpoints",
                                          "final.pt" if args.train_step is None else f"step_{args.train_step}.pt"),
                             map_location="cpu", weights_only=False)["codes"]["codes.weight"].double().numpy()
    z_mean, cov_inv = train_codes.mean(0), np.linalg.inv(np.cov(train_codes, rowvar=False))
    train_norm = float(np.linalg.norm(train_codes, axis=1).mean())

    model = load_model_file(args.model)
    faces = model.faces
    names = [str(n).strip() for n in model.landmarks["names "]]
    mean_mm = to_mm(model.mean.reshape(-1, 3))
    lmk = {n: mean_mm[i] for n, i in zip(names, model.landmarks["verts "])}

    def distances(fit_pts):
        """Scan -> fit and (face-weighted, when the fit has the template's vertices) fit -> scan, in mm."""
        d_scan, _ = cKDTree(fit_pts).query(scan_pts)
        d_fit, _ = scan_tree.query(fit_pts)
        return {
            "scan_to_fit_mm_mean": float(d_scan.mean()),
            "scan_to_fit_mm_support_weighted": float((d_scan * support).sum() / support.sum()),
            "scan_to_fit_mm_p95": float(np.percentile(d_scan, 95)),
            "fit_to_scan_mm_face_weighted": (float((d_fit * face_w).sum() / face_w.sum())
                                             if len(fit_pts) == len(face_w) else None),
        }

    runs = sorted(d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "checkpoints", "final.pt")))
    rows = {"mean code": torch.as_tensor(z_mean, dtype=torch.float32)[None]}
    for name in (args.runs or runs):
        rows[name] = torch.load(os.path.join(root, name, "checkpoints", "final.pt"), map_location="cpu",
                                weights_only=False)["codes"]["codes.weight"].float()
    # Every fit shares the frozen flow and the frame, so the first run's serve all codes.
    _, flow, _, _, template_norm, transform = load_run(os.path.join(root, runs[0]), None, device)
    results, polys = {}, {}
    for name, code in rows.items():
        _, pred = next(decode(flow, template_norm, transform, code, device, 1))
        fit_mm = to_mm(pred[0].double().cpu().numpy())
        z = code[0].double().numpy()
        results[name] = {
            **distances(fit_mm),
            "code_norm": float(np.linalg.norm(z)),
            "code_mahalanobis2": float((z - z_mean) @ cov_inv @ (z - z_mean)),
            "kinetic_last_step": last_kinetic(os.path.join(root, f"{name}.log")),
        }
        polys[name] = create_polydata(fit_mm, faces)
    # Other methods' results for the same scan, already in the scan's frame.
    for label, path in (item.split("=", 1) for item in args.extra):
        poly = load_vtp(path)
        results[label] = {**distances(vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)),
                          "code_norm": None, "code_mahalanobis2": None, "kinetic_last_step": None, "path": path}
        polys[label] = poly
    summary = {"scan": scan_name, "train_code_norm_mean": train_norm, "n_z": int(train_codes.shape[1]), "fits": results}
    with open(os.path.join(root, f"{args.name}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    def fmt(value, spec):
        return "-" if value is None else format(value, spec)

    width = max(len(n) for n in results) + 2
    print(f"{'fit':<{width}}{'scan->fit':>10}{'(supp-w)':>10}{'p95':>8}{'fit->scan':>11}{'|z|':>8}{'d2':>9}{'kinetic':>10}")
    for name, r in results.items():
        print(f"{name:<{width}}{r['scan_to_fit_mm_mean']:>10.2f}{r['scan_to_fit_mm_support_weighted']:>10.2f}"
              f"{r['scan_to_fit_mm_p95']:>8.2f}{fmt(r['fit_to_scan_mm_face_weighted'], '.2f'):>11}"
              f"{fmt(r['code_norm'], '.3f'):>8}{fmt(r['code_mahalanobis2'], '.0f'):>9}"
              f"{fmt(r['kinetic_last_step'], '.2e'):>10}")
    print(f"training codes: |z| {train_norm:.3f} on average, d2 about {train_codes.shape[1]}")

    # The fitted scan may be a point cloud (no polys, nothing to render): show its mesh.
    scan_poly = load_vtp(args.scan_mesh) if args.scan_mesh else scan_mm
    panels = [(f"scan {os.path.splitext(scan_name)[0]}", scan_poly, SCAN_COLOR)]
    panels += [(f"{name} ({results[name]['scan_to_fit_mm_mean']:.2f} mm)", polys[name], views.MESH_COLOR)
               for name in results]
    cams = {view: views.view_camera(lmk, [p for _, p, _ in panels], view) for view in ("frontal", "left", "right")}
    grid = [(cams[view], [(title, poly, color, []) for title, poly, color in panels]) for view in cams]
    os.makedirs(os.path.join(root, "views"), exist_ok=True)
    out = os.path.join(root, "views", f"{args.name}.png")
    views.write_grid(grid, len(panels), f"Fits of {scan_name}: code fits into the frozen BabySSM flow and other "
                     "methods (title: mean scan -> fit distance). Rows: frontal, left, right.", out, args.panel_px)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
