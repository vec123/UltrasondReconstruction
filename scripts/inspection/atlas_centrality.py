"""Score how good the template of a cohort auto-decoder run is as an atlas, and plot the
first two PCA components of its codes, each member coloured by the deformation energy
of its flow. The code mean and the zero code are marked and coloured on the same scale.

The energy is the flow's kinetic energy ½ ∫ mean_i ||v_i(t)||² dt over all template
points (the training's kinetic term for one member), in the run's normalised frame.
The template itself has energy 0: every flow starts there. A good atlas has a
low-energy code mean at the centre of the cloud, the energy growing with the distance
from it. The PCA plane shows only part of the code variance (see the axis labels), so
the title also gives the Spearman rank correlation (the correlation of the ranks) of
the energy with the distance to the code mean in the full code space.

The members are decoded from the run's checkpoint, so no apply/ is needed; with one,
the members get their names, and with each member's eval_reference_frame/ the bias is
also given over the vertices that the scans see (coverage). The other numbers come
from src/evaluation/atlas/metrics.py: bias, Fréchet test, mean-code difference,
compactness, specificity and triangle area ratios, in mm.

Writes <run>/atlas_centrality.png and <run>/atlas_centrality.json.

Run with:
    python scripts/inspection/atlas_centrality.py --run <run folder> \\
        --similarity /home/victor/Escritorio/Data/fused_targets_full313/similarity.json
"""

import argparse
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from vtk.util.numpy_support import vtk_to_numpy  # noqa: E402

from scripts.inspection.code_pca_views import load_codes, similarity_map  # noqa: E402
from src.evaluation.atlas import metrics  # noqa: E402
from src.models.lddmm.resnet_flow.apply.trained_run import load_run  # noqa: E402
from src.shared.vtk.io import load_vtp  # noqa: E402

# One blue hue, light (low energy) to dark (high energy).
ENERGY_CMAP = LinearSegmentedColormap.from_list(
    "energy", ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
INK = "#2b2b2b"
GRID = "#e4e3df"
COVERAGE_DISTANCE = "recon_dist_2_target"  # the eval's per-vertex distance array


@torch.no_grad()
def deform(flow, template_norm, transform, codes, device, batch):
    """(world points [M, N, 3], kinetic energy [M]) of phi_z(template) for each code [M, n_z]."""
    template = template_norm.points.to(device)
    center, scale = transform.center.to(device), transform.scale.to(device)
    points, energy = [], []
    for i in range(0, len(codes), batch):
        z = torch.as_tensor(codes[i:i + batch], dtype=template.dtype, device=device)
        trajectory = flow(template.expand(len(z), -1, -1), z)
        points.append((trajectory.points[-1] * scale + center).cpu().numpy())
        energy.append(metrics.kinetic_energy(trajectory.free_velocities.transpose(0, 1), trajectory.dt).cpu().numpy())
    return np.concatenate(points).astype(np.float64), np.concatenate(energy).astype(np.float64)


def coverage_distances(run, names):
    """[M, N] each member's eval distance per template vertex, or None if a member has no eval."""
    paths = [os.path.join(run, "apply", name or "", "eval_reference_frame", "recon_w_distance.vtp") for name in names]
    if not all(name and os.path.isfile(path) for name, path in zip(names, paths)):
        return None
    return np.stack([vtk_to_numpy(load_vtp(path).GetPointData().GetArray(COVERAGE_DISTANCE)) for path in paths])


def ranks(values):
    return np.argsort(np.argsort(values))


def plot(scores, energy, member_energy, zero_xy, ratio, rho, run, out_path):
    """The PCA scatter of the codes, coloured by energy, with the code mean and code 0 marked."""
    fig, ax = plt.subplots(figsize=(8, 7))
    # Log scale: a few far members carry many times the median energy and would
    # otherwise leave the rest one colour.
    all_energy = [*member_energy, energy["mean"], energy["zero"]]
    norm = LogNorm(min(all_energy), max(all_energy))
    dots = ax.scatter(scores[:, 0], scores[:, 1], c=member_energy, cmap=ENERGY_CMAP, norm=norm,
                      s=40, edgecolors="white", linewidths=0.8, zorder=2)
    # The two often lie close together: labels on opposite sides, the star on top.
    for key, xy, marker, size, offset, align, zorder in (
            ("zero", zero_xy, "D", 90, (10, 10), "left", 3),
            ("mean", (0.0, 0.0), "*", 320, (-12, -26), "right", 4)):
        ax.scatter(*xy, c=[energy[key]], cmap=ENERGY_CMAP, norm=norm, marker=marker,
                   s=size, edgecolors=INK, linewidths=1.2, zorder=zorder)
        label = "code mean" if key == "mean" else "code 0"
        ax.annotate(f"{label}\nE = {energy[key]:.2e}", xy, xytext=offset, textcoords="offset points",
                    ha=align, fontsize=9, color=INK, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))
    fig.colorbar(dots, ax=ax, shrink=0.8, label="kinetic energy ½∫ mean ||v||² dt (normalised frame, log scale)")
    ax.set_xlabel(f"PC1 ({100 * ratio[0]:.1f} % of code variance)", color=INK)
    ax.set_ylabel(f"PC2 ({100 * ratio[1]:.1f} % of code variance)", color=INK)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    median = np.median(member_energy)
    ax.set_title(f"{textwrap.fill(os.path.basename(run), 90)}\n"
                 f"code mean E = {energy['mean'] / median:.2f} × member median ({median:.2e}); "
                 f"Spearman ρ(energy, full-code distance to mean) = {rho:.2f}", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="The cohort run folder (config.json, checkpoints/).")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: final.pt).")
    parser.add_argument("--similarity", default=None,
                        help="similarity.json mapping the run's world frame to mm; default: world is mm.")
    parser.add_argument("--output", default=None, help="PNG to write (default: <run>/atlas_centrality.png).")
    parser.add_argument("--batch", type=int, default=16, help="Codes decoded at once.")
    parser.add_argument("--samples", type=int, default=50, help="Random codes for the specificity.")
    parser.add_argument("--coverage-threshold", type=float, default=2.0,
                        help="Eval distance (mm) below which a scan sees a vertex.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    to_mm = similarity_map(args.similarity)
    mm = lambda points: to_mm(points.reshape(-1, 3)).reshape(points.shape)  # noqa: E731

    _, flow, codes, _, template_norm, transform = load_run(args.run, args.step, device)
    codes = codes.numpy().astype(np.float64)
    names, _ = load_codes(args.run)
    mean = codes.mean(axis=0)
    rng = np.random.default_rng(args.seed)
    sampled = rng.multivariate_normal(mean, np.cov(codes, rowvar=False), size=args.samples)
    extra = np.stack([mean, np.zeros_like(mean)])
    points, energies = deform(flow, template_norm, transform, np.concatenate([codes, extra, sampled]),
                              device, args.batch)
    points = mm(points)
    recons, (mean_recon, _), samples = np.split(points, [len(codes), len(codes) + 2])
    member_energy = energies[:len(codes)]
    energy = {"mean": float(energies[len(codes)]), "zero": float(energies[len(codes) + 1])}
    template = mm((template_norm.points[0] * transform.scale + transform.center).cpu().numpy().astype(np.float64))
    faces = np.asarray(template_norm.faces.cpu() if torch.is_tensor(template_norm.faces) else template_norm.faces)

    _, s, vt = np.linalg.svd(codes - mean, full_matrices=False)
    ratio = s[:2] ** 2 / np.sum(s ** 2)
    rho = float(np.corrcoef(ranks(member_energy), ranks(np.linalg.norm(codes - mean, axis=1)))[0, 1])
    out_path = args.output or os.path.join(args.run, "atlas_centrality.png")
    plot((codes - mean) @ vt[:2].T, energy, member_energy, -mean @ vt[:2].T, ratio, rho, args.run, out_path)

    _, _, bias_ratio = metrics.bias(template, recons)
    report = {
        "units": "mm, energies in the run's normalised frame",
        "energy": {**energy, "member_median": float(np.median(member_energy)),
                   "members": {names[i] or str(i): float(e) for i, e in enumerate(member_energy)}},
        "spearman_energy_vs_code_distance": rho,
        "bias_ratio": bias_ratio,
        "frechet_test": metrics.frechet_test(template, recons, seed=args.seed),
        "mean_code_difference": metrics.mean_code_difference(template, recons, mean_recon),
        "compactness": {"codes": metrics.compactness(codes)[:10].tolist(),
                        "shapes": metrics.compactness(recons)[:10].tolist()},
        "specificity": {"median": float(np.median(metrics.specificity(samples, recons))),
                        "samples": args.samples},
        "triangle_area_ratio": metrics.determinant_stats(metrics.triangle_area_ratios(template, recons, faces)),
    }
    distances = coverage_distances(args.run, names)
    if distances is not None:
        seen = metrics.coverage(distances, args.coverage_threshold)
        report["coverage"] = {"threshold": args.coverage_threshold, "mean": float(seen.mean()),
                              "bias_ratio_weighted": metrics.bias(template, recons, seen)[2]}
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
