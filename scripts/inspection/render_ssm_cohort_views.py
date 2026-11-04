"""Render the 3D shapes of a cohort auto-decoder trained on SSM samples, into <output>/views/.

One PNG per case, each with a frontal, a left and a right row (one camera per view,
shared by the row, so the panels compare at the same scale):

  * train_<rank>_<index>.png, test_<rank>_<index>.png: the best, median and worst
    training member / new sample by the flow's mean vertex error (from the
    eval_ssm_cohort.py arrays), with the columns
        target | flow reconstruction | flow error | PCA-K | PCA-K error | mean shape
    where PCA-K is the target projected onto the first K PCA modes, K the number of
    modes with the same mean error as the flow (eval summary). The error panels colour
    the reconstruction by its per-vertex distance to the target, light (0) to dark
    (--max-error-pct of the bounding-box diagonal of the mean shape, or more).
  * generation_vs_pca.png: shapes decoded from codes drawn from a Gaussian fitted to
    the training codes (top three rows), next to new samples of the PCA model
    (bottom three rows).

The SSM lives in units of about 0.02 (bbox diagonal); the points are scaled by
RENDER_SCALE for rendering, because the camera of src/shared/vtk/render.py works in millimetres.

Run with:
    python scripts/inspection/render_ssm_cohort_views.py \\
        --run outputs/baby_ssm_cohort/resnet_lddmm_cohort_autodecoder_n5000_global_l2_cont20k \\
        --step 38000 \\
        --fit-run outputs/baby_ssm_cohort/resnet_lddmm_cohort_autodecoder_n5000_global_l2_test200_code_fit \\
        --train-data /home/victor/Escritorio/Data/BabySSMData \\
        --test-data /home/victor/Escritorio/Data/BabySSMData_test200 \\
        --eval-dir outputs/baby_ssm_cohort/eval_step38000
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from vtk.util.numpy_support import numpy_to_vtk

from src.shared.vtk import render as views  # noqa: E402
from src.models.lddmm.resnet_flow.apply.trained_run import decode, load_run  # noqa: E402
from src.models.lddmm.resnet_flow.runner import cohort_paths  # noqa: E402
from src.models.gpmm.model.model_memory_io import load_model_file  # noqa: E402
from src.models.gpmm.model.sampling import draw_cohort  # noqa: E402
from src.shared.vtk.create import create_polydata  # noqa: E402

RENDER_SCALE = 1e4         # model units -> about millimetres (bbox diagonal 0.02 -> 200)
VIEWS = ("frontal", "left", "right")
# Sequential blue ramp, light (no error) -> dark (max error).
ERROR_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def mesh(points, faces, colors=None):
    """vtkPolyData of [N, 3] model-unit points, scaled for rendering; RGB scalars if given."""
    poly = create_polydata(np.asarray(points, dtype=np.float64) * RENDER_SCALE, faces)
    if colors is not None:
        array = numpy_to_vtk(colors, deep=True)
        array.SetName("Colors")
        poly.GetPointData().SetScalars(array)
    return poly


def view_cameras(landmark_points, polys):
    """{view: camera} framing all `polys`, oriented by the landmarks of the first shape."""
    return {view: views.view_camera(landmark_points, polys, view) for view in VIEWS}


def landmark_dict(points, landmarks):
    """{name: [3] rendered point} for the named landmark vertices."""
    names = [str(n).strip() for n in landmarks["names "]]
    return {name: points[i] * RENDER_SCALE for name, i in zip(names, landmarks["verts "])}


def write_case(out_path, header, panels, landmark_points):
    """One PNG: `panels` [(title, poly, colour)] as columns, a frontal, left and right row."""
    cams = view_cameras(landmark_points, [p for _, p, _ in panels])
    rows = [(cams[view], [(title, poly, color, []) for title, poly, color in panels]) for view in VIEWS]
    views.write_grid(rows, len(panels), header, out_path, PANEL_PX)


def main():
    global PANEL_PX
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--fit-run", default=None, help="freeze_flow code fit of --test-data (omit: no test cases)")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", default=None)
    parser.add_argument("--eval-dir", default=None,
                        help="eval_ssm_cohort.py output (arrays.npz, summary.json); omit to compute the errors here")
    parser.add_argument("--pca-k", type=int, default=None,
                        help="K of the PCA-K column (default: the eval's train_equivalent_pca_modes)")
    parser.add_argument("--model", default="assets/shape_models/baby/baby_pca.npz")
    parser.add_argument("--output-dir", default=None, help="default: <eval-dir>/views")
    parser.add_argument("--n-generate", type=int, default=6)
    parser.add_argument("--max-error-pct", type=float, default=2.0)
    parser.add_argument("--panel-px", type=int, default=280)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    PANEL_PX = args.panel_px
    if args.output_dir is None and args.eval_dir is None:
        parser.error("give --output-dir when there is no --eval-dir")
    if args.eval_dir is None and args.pca_k is None:
        parser.error("give --pca-k when there is no --eval-dir")
    out_dir = args.output_dir or os.path.join(args.eval_dir, "views")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)

    model = load_model_file(args.model)
    faces = model.faces
    mean = model.mean
    basis = model.eigenfunctions * np.sqrt(model.eigenvalues)
    v = mean.reshape(-1, 3)
    diag = float(np.linalg.norm(v.max(0) - v.min(0)))
    pct = 100.0 / diag
    arrays = np.load(os.path.join(args.eval_dir, "arrays.npz")) if args.eval_dir else None
    if args.pca_k is None:
        with open(os.path.join(args.eval_dir, "summary.json")) as f:
            k_eq = int(round(json.load(f)["reconstruction"]["train_equivalent_pca_modes"]))
    else:
        k_eq = args.pca_k
    basis_k = basis[:, :k_eq]
    pinv_k = np.linalg.pinv(basis_k)

    cfg, flow, z_train, _, template_norm, transform = load_run(args.run, args.step, device)

    def member_sigmas(paths, data_dir):
        """True PCA coefficients of each member, by its sample_<index> file name."""
        sigmas = np.load(os.path.join(data_dir, "coefficients.npz"))["sigmas"]
        return sigmas[[int(os.path.basename(p)[len("sample_"):-len(".vtp")]) for p in paths]]

    def flow_errors(codes, sigmas):
        """Mean vertex error of phi_z(template) against mean + basis @ s, per member."""
        errors = []
        for i, pred in decode(flow, template_norm, transform, codes, device, 25):
            true = torch.as_tensor((mean + sigmas[i:i + len(pred)] @ basis.T).reshape(len(pred), -1, 3),
                                   dtype=pred.dtype, device=device)
            errors.append((pred - true).norm(dim=-1).mean(1).cpu().numpy())
        return np.concatenate(errors)

    s_train = member_sigmas(cohort_paths(cfg), args.train_data)
    sets = {"train": (z_train, s_train, arrays["err_train"][:, 0] if arrays is not None
                      else flow_errors(z_train, s_train), "Training member")}
    if args.fit_run:
        z_test = torch.load(os.path.join(args.fit_run, "checkpoints", "final.pt"), map_location="cpu",
                            weights_only=False)["codes"]["codes.weight"].float()
        test_paths = sorted(glob.glob(os.path.join(args.test_data, "members", "*.vtp")))
        s_test = member_sigmas(test_paths, args.test_data)
        sets["test"] = (z_test, s_test, arrays["err_test"][:, 0] if arrays is not None
                        else flow_errors(z_test, s_test), "New sample (code fit, flow frozen)")
    mean_poly = mesh(v, faces)
    max_err = args.max_error_pct / pct
    legend = f"error colour: light 0 -> dark {args.max_error_pct:g} % of bbox diagonal"

    for name, (codes, sigmas, errors, label) in sets.items():
        order = np.argsort(errors)
        for rank, idx in (("best", order[0]), ("median", order[len(order) // 2]), ("worst", order[-1])):
            target = (mean + basis @ sigmas[idx]).reshape(-1, 3)
            _, pred = next(decode(flow, template_norm, transform, codes[idx:idx + 1], device, 1))
            recon = pred[0].double().cpu().numpy()
            pca_k = (mean + basis_k @ (pinv_k @ (target.reshape(-1) - mean))).reshape(-1, 3)
            err_flow = np.linalg.norm(recon - target, axis=1)
            err_pca = np.linalg.norm(pca_k - target, axis=1)
            panels = [
                ("target", mesh(target, faces), views.MESH_COLOR),
                ("flow reconstruction", mesh(recon, faces), views.MESH_COLOR),
                (f"flow error (mean {err_flow.mean() * pct:.2f} %)", mesh(recon, faces, views.ramp_colors(err_flow, max_err, ERROR_RAMP)), None),
                (f"PCA-{k_eq}", mesh(pca_k, faces), views.MESH_COLOR),
                (f"PCA-{k_eq} error (mean {err_pca.mean() * pct:.2f} %)", mesh(pca_k, faces, views.ramp_colors(err_pca, max_err, ERROR_RAMP)), None),
                ("mean shape", mean_poly, views.MESH_COLOR),
            ]
            header = (f"{label} {idx} ({rank} of {len(errors)} by flow error); "
                      f"max vertex error flow {err_flow.max() * pct:.1f} %, PCA-{k_eq} {err_pca.max() * pct:.1f} %; {legend}")
            path = os.path.join(out_dir, f"{name}_{rank}_{idx:05d}.png")
            write_case(path, header, panels, landmark_dict(target, model.landmarks))
            print(f"wrote {path}")

    # Generation: Gaussian codes vs new PCA samples, all under one camera per view.
    rng = np.random.default_rng(args.seed)
    zt = z_train.numpy().astype(np.float64)
    z_gen = torch.as_tensor(rng.multivariate_normal(zt.mean(0), np.cov(zt, rowvar=False), size=args.n_generate),
                            dtype=torch.float32)
    generated = [p.double().cpu().numpy() for _, batch in decode(flow, template_norm, transform, z_gen, device,
                                                                 args.n_generate) for p in batch]
    _, _, pca_samples = draw_cohort(model, args.n_generate, seed=args.seed + 3)
    gen_polys = [mesh(p, faces) for p in generated]
    pca_polys = [mesh(p, faces) for p in pca_samples]
    cams = view_cameras(landmark_dict(v, model.landmarks), gen_polys + pca_polys)
    rows = [(cams[view], [(f"generated {i + 1}", p, views.MESH_COLOR, []) for i, p in enumerate(gen_polys)])
            for view in VIEWS]
    rows += [(cams[view], [(f"PCA sample {i + 1}", p, views.MESH_COLOR, []) for i, p in enumerate(pca_polys)])
             for view in VIEWS]
    path = os.path.join(out_dir, "generation_vs_pca.png")
    views.write_grid(rows, args.n_generate, "Top: decoded from codes ~ Gaussian of the training codes. "
                     "Bottom: new samples of the PCA model. Rows: frontal, left, right.", path, PANEL_PX)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
