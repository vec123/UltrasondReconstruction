"""Evaluate a cohort auto-decoder trained on samples of a linear shape model (SSM).

The training members were drawn by scripts/SSM/sample_cohort.py, so every member has
known PCA coefficients s (in sigma units) and is in vertex correspondence with the
template (the model mean). That makes four comparisons with the PCA model possible:

  1. codes:       how linearly the learned codes Z carry the coefficients s
                  (R^2 per mode of a linear map Z -> s, fitted on 80 % of the members, tested
                  on the other 20 % and on the codes fitted to new samples), and the
                  spectrum / effective dimension of Z next to the PCA eigenvalues.
  2. reconstruction of the training members: vertex error of phi_z(template) on the
                  full mesh, next to the error of PCA truncated to K modes, and how much
                  of each PCA mode the reconstructions reproduce (R^2 per mode).
  3. generation:  codes drawn from a Gaussian fitted to Z, decoded, projected onto the
                  PCA basis: per-mode std (PCA samples: 1), off-model residual, per-vertex
                  std next to the PCA model's, and specificity (RMS distance to the
                  nearest training member) next to that of new PCA samples.
  4. new samples: codes fitted with the flow frozen (a train.freeze_flow run): their
                  reconstruction error, their Mahalanobis distance under the Gaussian of
                  Z, and the map of 1. predicting their true coefficients.

Errors are in model units and as % of the bounding-box diagonal of the mean shape.

Run with:
    python scripts/inspection/eval_ssm_cohort.py \\
        --run outputs/baby_ssm_cohort/resnet_lddmm_cohort_autodecoder_n5000_global_l2_cont20k \\
        --step 38000 \\
        --fit-run outputs/baby_ssm_cohort/resnet_lddmm_cohort_autodecoder_n5000_global_l2_test200_code_fit \\
        --train-data /home/victor/Escritorio/Data/BabySSMData \\
        --test-data /home/victor/Escritorio/Data/BabySSMData_test200 \\
        --output-dir outputs/baby_ssm_cohort/eval_step38000
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.SSM.sample_cohort import draw_cohort  # noqa: E402
from src.shared.core.config import load_yaml  # noqa: E402
from src.resnet_lddmm.resnet_flow import registrations  # noqa: E402,F401  (registers all components)
from src.resnet_lddmm.resnet_flow.config import ExperimentCfg  # noqa: E402
from src.resnet_lddmm.resnet_flow.io import load_shape  # noqa: E402
from src.resnet_lddmm.resnet_flow.runner import build_flow, load_cohort, set_default_dtype  # noqa: E402
from src.SSM.model.model_memory_io import load_model_file  # noqa: E402

PCA_K = (0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 163)
N_PCA_BASELINE = 500      # members the PCA-K baseline errors are averaged over
N_SPEC_VERTICES = 4000    # vertices the specificity distances are computed on

# Light-surface chart tokens (dataviz reference palette).
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")


def load_run(run, step, device):
    """(train_cfg, flow on device in eval mode, codes [M, n_z], cohort_norm, template_norm, transform)."""
    data = load_yaml(os.path.join(run, "config.json"))
    data.pop("mode", None)
    cfg = ExperimentCfg.from_dict(data)
    set_default_dtype(cfg)
    name = "final.pt" if step is None else f"step_{step}.pt"
    checkpoint = torch.load(os.path.join(run, "checkpoints", name), map_location="cpu", weights_only=False)
    flow = build_flow(cfg)
    flow.load_state_dict(checkpoint["flow"])
    flow = flow.to(device).eval()
    _, cohort_norm, template_norm, transform = load_cohort(cfg)
    return cfg, flow, checkpoint["codes"]["codes.weight"].float(), cohort_norm, template_norm, transform


@torch.no_grad()
def decode(flow, template_norm, transform, codes, device, batch):
    """Yield (start index, world points [b, N, 3] on device) of phi_z(template) for each code."""
    template = template_norm.points.to(device)
    center, scale = transform.center.to(device), transform.scale.to(device)
    for i in range(0, len(codes), batch):
        z = codes[i:i + batch].to(device)
        q = flow(template.expand(len(z), -1, -1), z).points[-1]
        yield i, q * scale + center


def r2_per_column(true, pred):
    """R^2 of each column of `pred` against `true` [M, D]."""
    resid = ((true - pred) ** 2).sum(0)
    total = ((true - true.mean(0)) ** 2).sum(0)
    return 1.0 - resid / total


def linear_fit(x, y):
    """Least-squares y ~ [x, 1] W; returns a predict function."""
    w, *_ = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=None)
    return lambda x_new: np.c_[x_new, np.ones(len(x_new))] @ w


def vertex_errors(pred, true):
    """Per-member (mean vertex distance, RMS vertex distance, max vertex distance) [b] each."""
    d = (pred - true).norm(dim=-1)
    return d.mean(1), d.pow(2).mean(1).sqrt(), d.max(1).values


def gaussian_mahalanobis(z, mean, cov_inv):
    """Squared Mahalanobis distance of each row of z."""
    d = z - mean
    return np.einsum("ij,jk,ik->i", d, cov_inv, d)


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="the training run folder")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step (default: final.pt)")
    parser.add_argument("--fit-run", required=True, help="the freeze_flow code-fit run on new samples")
    parser.add_argument("--train-data", required=True, help="sample_cohort.py output of the training set")
    parser.add_argument("--test-data", required=True, help="sample_cohort.py output of the new samples")
    parser.add_argument("--model", default="assets/shape_models/baby/baby_pca.npz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-generate", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # --- inputs -----------------------------------------------------------------
    model = load_model_file(args.model)
    n_modes = model.n_components
    mean = model.mean                                                    # [3N]
    sqrt_eig = np.sqrt(model.eigenvalues)
    basis = model.eigenfunctions * sqrt_eig                              # [3N, K]: x = mean + basis @ s
    proj = torch.as_tensor(np.linalg.pinv(basis), dtype=torch.float32, device=device)  # [K, 3N]
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    v = mean.reshape(-1, 3)
    diag = float(np.linalg.norm(v.max(0) - v.min(0)))
    pct = 100.0 / diag

    s_train = np.load(os.path.join(args.train_data, "coefficients.npz"))["sigmas"]
    s_test = np.load(os.path.join(args.test_data, "coefficients.npz"))["sigmas"]

    cfg, flow, z_train, cohort_norm, template_norm, transform = load_run(args.run, args.step, device)
    z_test = torch.load(os.path.join(args.fit_run, "checkpoints", "final.pt"), map_location="cpu",
                        weights_only=False)["codes"]["codes.weight"].float()
    if len(z_train) != len(s_train) or len(z_test) != len(s_test):
        raise ValueError(f"codes ({len(z_train)}, {len(z_test)}) and coefficients "
                         f"({len(s_train)}, {len(s_test)}) do not match")
    center, scale = transform.center.to(device), transform.scale.to(device)
    x_train = torch.stack([m.points[0] for m in cohort_norm])            # [M, N, 3] normalised, CPU
    test_paths = sorted(glob.glob(os.path.join(args.test_data, "members", "*.vtp")))
    x_test = torch.stack([load_shape(p).points[0] for p in test_paths])  # [M', N, 3] world, CPU
    summary = {"run": args.run, "step": args.step, "bbox_diagonal": diag, "n_train": len(z_train),
               "n_test": len(z_test), "n_z": int(z_train.shape[1]), "n_modes": n_modes}

    # --- 1. codes vs PCA coefficients ---------------------------------------------
    zt, ze = z_train.numpy().astype(np.float64), z_test.numpy().astype(np.float64)
    order = rng.permutation(len(zt))
    n_fit = int(0.8 * len(zt))
    fit_idx, hold_idx = order[:n_fit], order[n_fit:]
    z_to_s = linear_fit(zt[fit_idx], s_train[fit_idx])
    r2_hold = r2_per_column(s_train[hold_idx], z_to_s(zt[hold_idx]))
    r2_test = r2_per_column(s_test, z_to_s(ze))
    s_to_z = linear_fit(s_train[fit_idx], zt[fit_idx])
    z_hold_pred = s_to_z(s_train[hold_idx])
    z_var_explained = 1.0 - ((zt[hold_idx] - z_hold_pred) ** 2).sum() / ((zt[hold_idx] - zt[hold_idx].mean(0)) ** 2).sum()

    z_mean, z_cov = zt.mean(0), np.cov(zt, rowvar=False)
    code_eig = np.clip(np.linalg.eigvalsh(z_cov)[::-1], 0, None)
    cov_inv = np.linalg.inv(z_cov)
    maha_train = gaussian_mahalanobis(zt, z_mean, cov_inv)
    maha_test = gaussian_mahalanobis(ze, z_mean, cov_inv)
    eig_frac = model.eigenvalues / model.eigenvalues.sum()
    summary["codes"] = {
        "code_norm_mean": float(np.linalg.norm(zt, axis=1).mean()),
        "code_mean_norm": float(np.linalg.norm(z_mean)),
        "per_dim_std_min_max": [float(zt.std(0).min()), float(zt.std(0).max())],
        "effective_dim_codes": float(code_eig.sum() ** 2 / (code_eig ** 2).sum()),
        "effective_dim_pca": float(model.eigenvalues.sum() ** 2 / (model.eigenvalues ** 2).sum()),
        "code_dims_for_95pct_var": int(np.searchsorted(np.cumsum(code_eig) / code_eig.sum(), 0.95) + 1),
        "pca_modes_for_95pct_var": int(np.searchsorted(np.cumsum(eig_frac), 0.95) + 1),
        "z_to_s_r2_holdout_first10": r2_hold[:10].round(3).tolist(),
        "z_to_s_r2_holdout_variance_weighted": float((r2_hold * eig_frac).sum()),
        "z_to_s_r2_test_variance_weighted": float((r2_test * eig_frac).sum()),
        "z_to_s_r2_test_first10": r2_test[:10].round(3).tolist(),
        "s_to_z_variance_explained_holdout": float(z_var_explained),
        "mahalanobis2_train_mean_median": [float(maha_train.mean()), float(np.median(maha_train))],
        "mahalanobis2_test_mean_median": [float(maha_test.mean()), float(np.median(maha_test))],
        "test_codes_dist_to_mean_vs_train": [float(np.linalg.norm(ze - z_mean, axis=1).mean()),
                                             float(np.linalg.norm(zt - z_mean, axis=1).mean())],
    }
    print("[codes]", json.dumps(summary["codes"], indent=1))

    # --- 2. + 4. reconstructions: training members and new samples ----------------
    def reconstruct(codes, targets, targets_normalised):
        errs, s_hat = [], []
        for i, pred in decode(flow, template_norm, transform, codes, device, args.batch):
            true = targets[i:i + len(pred)].to(device)
            if targets_normalised:
                true = true * scale + center
            errs.append(torch.stack(vertex_errors(pred, true), 1).cpu())
            s_hat.append(((pred.reshape(len(pred), -1) - mean_t) @ proj.T).cpu())
        return torch.cat(errs).numpy(), torch.cat(s_hat).numpy().astype(np.float64)

    err_train, s_hat_train = reconstruct(z_train, x_train, True)
    err_test, s_hat_test = reconstruct(z_test, x_test, False)

    # PCA truncated to K modes (Euclidean least squares onto the first K basis vectors).
    sub = rng.choice(len(s_train), N_PCA_BASELINE, replace=False)
    x_sub = mean + s_train[sub] @ basis.T                                 # [b, 3N]
    pca_err = {}
    for k in PCA_K:
        d = x_sub - mean
        if k:
            bk = basis[:, :k]
            d = d - (d @ np.linalg.pinv(bk).T) @ bk.T
        pca_err[k] = float(np.linalg.norm(d.reshape(len(sub), -1, 3), axis=-1).mean())
    ks, es = np.array(PCA_K), np.array([pca_err[k] for k in PCA_K])

    def equivalent_k(err):
        """Number of PCA modes with the same mean vertex error (log-linear interpolation)."""
        return float(np.interp(-np.log(err), -np.log(es), ks))

    r2_recon_train = r2_per_column(s_train, s_hat_train)
    r2_recon_test = r2_per_column(s_test, s_hat_test)
    summary["reconstruction"] = {
        "pca_K_mean_vertex_error_pct": {str(k): pca_err[k] * pct for k in PCA_K},
        "train_mean_vertex_error_pct": float(err_train[:, 0].mean() * pct),
        "train_rms_vertex_error_pct": float(np.sqrt((err_train[:, 1] ** 2).mean()) * pct),
        "train_max_vertex_error_pct_median_p95": [float(np.median(err_train[:, 2]) * pct),
                                                  float(np.percentile(err_train[:, 2], 95) * pct)],
        "test_mean_vertex_error_pct": float(err_test[:, 0].mean() * pct),
        "test_rms_vertex_error_pct": float(np.sqrt((err_test[:, 1] ** 2).mean()) * pct),
        "test_max_vertex_error_pct_median_p95": [float(np.median(err_test[:, 2]) * pct),
                                                 float(np.percentile(err_test[:, 2], 95) * pct)],
        "train_equivalent_pca_modes": equivalent_k(err_train[:, 0].mean()),
        "test_equivalent_pca_modes": equivalent_k(err_test[:, 0].mean()),
        "recon_r2_per_mode_train_first10": r2_recon_train[:10].round(3).tolist(),
        "recon_r2_per_mode_test_first10": r2_recon_test[:10].round(3).tolist(),
        "recon_r2_variance_weighted_train": float((r2_recon_train * eig_frac).sum()),
        "recon_r2_variance_weighted_test": float((r2_recon_test * eig_frac).sum()),
    }
    print("[reconstruction]", json.dumps(summary["reconstruction"], indent=1))

    # --- 3. generation ------------------------------------------------------------------
    z_gen = torch.as_tensor(rng.multivariate_normal(z_mean, z_cov, size=args.n_generate), dtype=torch.float32)
    spec_idx = torch.as_tensor(np.sort(rng.choice(len(v), N_SPEC_VERTICES, replace=False)), device=device)
    train_spec = (x_train[:, spec_idx.cpu()].to(device) * scale + center).reshape(len(x_train), -1)

    def nearest_rms(shapes):
        """RMS vertex distance of each shape [b, N, 3] (device) to its nearest training member."""
        flat = shapes[:, spec_idx].reshape(len(shapes), -1)
        return (torch.cdist(flat, train_spec).min(1).values / np.sqrt(N_SPEC_VERTICES)).cpu()

    s_gen, off_model, spec_gen = [], [], []
    vert_sum = torch.zeros(len(v), 3, device=device, dtype=torch.float64)
    vert_sq = torch.zeros(len(v), 3, device=device, dtype=torch.float64)
    for _, pred in decode(flow, template_norm, transform, z_gen, device, args.batch):
        flat = pred.reshape(len(pred), -1) - mean_t
        s = flat @ proj.T
        resid = (flat - s @ torch.as_tensor(basis.T, dtype=torch.float32, device=device)).reshape(len(pred), -1, 3)
        s_gen.append(s.cpu())
        off_model.append(resid.norm(dim=-1).mean(1).cpu())
        spec_gen.append(nearest_rms(pred))
        vert_sum += pred.double().sum(0)
        vert_sq += pred.double().pow(2).sum(0)
    s_gen = torch.cat(s_gen).numpy()
    off_model = torch.cat(off_model).numpy()
    spec_gen = torch.cat(spec_gen).numpy()
    vert_std_gen = (vert_sq / args.n_generate - (vert_sum / args.n_generate) ** 2).clamp_min(0).sum(1).sqrt().cpu().numpy()
    phi = model.eigenfunctions.reshape(len(v), 3, n_modes)
    vert_std_pca = np.sqrt(((phi ** 2).sum(1) * model.eigenvalues).sum(1))
    _, _, x_pca_new = draw_cohort(model, args.n_generate, seed=args.seed + 2)
    spec_pca = torch.cat([nearest_rms(torch.as_tensor(x_pca_new[i:i + 100], dtype=torch.float32, device=device))
                          for i in range(0, args.n_generate, 100)]).numpy()
    ratio = vert_std_gen / vert_std_pca
    summary["generation"] = {
        "n": args.n_generate,
        "mode_std_first10": s_gen.std(0)[:10].round(3).tolist(),
        "mode_std_variance_weighted": float((s_gen.std(0) * eig_frac).sum()),
        "mode_mean_abs_max": float(np.abs(s_gen.mean(0)).max()),
        "recon_mode_std_first10_train": s_hat_train.std(0)[:10].round(3).tolist(),
        "off_model_mean_vertex_pct": float(off_model.mean() * pct),
        "vertex_std_ratio_gen_over_pca_median_p5_p95": [float(np.median(ratio)), float(np.percentile(ratio, 5)),
                                                        float(np.percentile(ratio, 95))],
        "specificity_rms_pct_generated_mean": float(spec_gen.mean() * pct),
        "specificity_rms_pct_pca_samples_mean": float(spec_pca.mean() * pct),
    }
    print("[generation]", json.dumps(summary["generation"], indent=1))

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(os.path.join(args.output_dir, "arrays.npz"), r2_hold=r2_hold, r2_test=r2_test,
             r2_recon_train=r2_recon_train, r2_recon_test=r2_recon_test, err_train=err_train,
             err_test=err_test, s_gen=s_gen, maha_train=maha_train, maha_test=maha_test,
             code_eig=code_eig, vert_std_gen=vert_std_gen, vert_std_pca=vert_std_pca,
             spec_gen=spec_gen, spec_pca=spec_pca, pca_k=ks, pca_err=es)

    # --- plots ----------------------------------------------------------------------
    modes = np.arange(1, n_modes + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(modes, r2_hold, color=SERIES[0], lw=2, label=f"training codes (held-out {len(hold_idx)})")
    ax.plot(modes, r2_test, color=SERIES[1], lw=2, label="codes fitted to new samples")
    style(ax, "Linear map code -> PCA coefficient", "PCA mode", "R²")
    ax.set_ylim(-0.1, 1.02)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    ax = axes[1]
    ax.plot(np.arange(1, len(code_eig) + 1), np.cumsum(code_eig) / code_eig.sum(), color=SERIES[0], lw=2,
            label="learned codes (PCA of Z)")
    ax.plot(modes, np.cumsum(eig_frac), color=SERIES[1], lw=2, label="PCA model eigenvalues")
    style(ax, "Cumulative variance", "component", "fraction of variance")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    save(fig, os.path.join(args.output_dir, "codes_vs_pca.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    # K = all modes is exact (error ~1e-14) and is left out, so the axis can stay linear.
    ax.plot(ks[:-1], es[:-1] * pct, color=MUTED, lw=2, marker="o", ms=5, label="PCA truncated to K modes")
    ax.axhline(err_train[:, 0].mean() * pct, color=SERIES[0], lw=2, label="flow, training members")
    ax.axhline(err_test[:, 0].mean() * pct, color=SERIES[1], lw=2, ls="--", label="flow, new samples (code fit)")
    ax.set_ylim(0, None)
    style(ax, f"Mean vertex error (PCA with all {n_modes} modes: 0)", "K (PCA modes)", "% of bbox diagonal")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    ax = axes[1]
    ax.plot(modes, r2_recon_train, color=SERIES[0], lw=2, label="training members")
    ax.plot(modes, r2_recon_test, color=SERIES[1], lw=2, label="new samples (code fit)")
    style(ax, "PCA modes reproduced by the reconstructions", "PCA mode", "R² (reconstructed vs true coefficient)")
    ax.set_ylim(-0.1, 1.02)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    save(fig, os.path.join(args.output_dir, "reconstruction.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.axhline(1.0, color=MUTED, lw=2, label="PCA samples (1 by construction)")
    ax.plot(modes, s_hat_train.std(0), color=SERIES[0], lw=2, label="reconstructions of training members")
    ax.plot(modes, s_gen.std(0), color=SERIES[1], lw=2, label="generated (Gaussian codes)")
    style(ax, "Std of PCA coefficients", "PCA mode", "std (sigma units)")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
    ax = axes[1]
    bins = np.linspace(0, max(spec_gen.max(), spec_pca.max()) * pct, 40)
    ax.hist(spec_pca * pct, bins=bins, color=MUTED, alpha=0.7, label="new PCA samples")
    ax.hist(spec_gen * pct, bins=bins, color=SERIES[1], alpha=0.7, label="generated")
    style(ax, "Specificity: RMS to nearest training member", "% of bbox diagonal", "count")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    save(fig, os.path.join(args.output_dir, "generation.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    bins = np.linspace(0, np.percentile(np.r_[maha_train, maha_test], 99.5), 40)
    ax.hist(maha_train, bins=bins, color=SERIES[0], alpha=0.7, density=True, label="training codes")
    ax.hist(maha_test, bins=bins, color=SERIES[1], alpha=0.7, density=True, label="codes fitted to new samples")
    style(ax, "Squared Mahalanobis distance under the code Gaussian", "d²", "density")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    ax = axes[1]
    bins = np.linspace(0, max(err_train[:, 0].max(), err_test[:, 0].max()) * pct, 40)
    ax.hist(err_train[:, 0] * pct, bins=bins, color=SERIES[0], alpha=0.7, density=True, label="training members")
    ax.hist(err_test[:, 0] * pct, bins=bins, color=SERIES[1], alpha=0.7, density=True, label="new samples (code fit)")
    style(ax, "Per-member mean vertex error", "% of bbox diagonal", "density")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    save(fig, os.path.join(args.output_dir, "code_stats.png"))

    # The training codes on their own first two principal components; the fitted codes of
    # the new samples projected onto the same axes.
    _, sv, pcs = np.linalg.svd(zt - z_mean, full_matrices=False)
    var = sv ** 2 / (sv ** 2).sum()
    p_train, p_test = (zt - z_mean) @ pcs[:2].T, (ze - z_mean) @ pcs[:2].T
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(p_train[:, 0], p_train[:, 1], s=10, color=SERIES[0], alpha=0.5, linewidths=0,
               label=f"training codes ({len(zt)})")
    ax.scatter(p_test[:, 0], p_test[:, 1], s=16, color=SERIES[1], alpha=0.8, linewidths=0,
               label=f"codes fitted to new samples ({len(ze)})")
    style(ax, "Codes on the first two principal components of the training codes",
          f"PC1 ({var[0] * 100:.1f} % of code variance)", f"PC2 ({var[1] * 100:.1f} %)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    save(fig, os.path.join(args.output_dir, "code_pca_scatter.png"))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
