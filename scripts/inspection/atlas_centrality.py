"""Plot the first two PCA components of a cohort auto-decoder's codes, each member
coloured by the deformation energy of its flow, to see whether the code mean is a
good atlas. The code mean and the zero code are marked and coloured on the same scale.

The energy is the flow's kinetic energy ½ ∫ mean_i ||v_i(t)||² dt over all template
points (Trajectory.kinetic_energy: the training's kinetic term for one member), in the
run's normalised frame. The template itself has energy 0: every flow starts there.
A good atlas has a low-energy code mean at the centre of the cloud, the energy growing
with the distance from it. The PCA plane shows only part of the code variance (see the
axis labels), so the title also gives the Spearman rank correlation (the correlation
of the ranks) of the energy with the distance to the code mean in the full code space.

Writes <run>/atlas_centrality.png and <run>/atlas_centrality.json (the energies).

Run with:
    python scripts/inspection/atlas_centrality.py --apply-config <the run's cohort apply .yaml>
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

from scripts.inspection.code_pca_modes import load_run  # noqa: E402
from scripts.inspection.code_pca_views import load_codes  # noqa: E402
from src.models.lddmm.resnet_flow.io import load_shape  # noqa: E402

# One blue hue, light (low energy) to dark (high energy).
ENERGY_CMAP = LinearSegmentedColormap.from_list(
    "energy", ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
INK = "#2b2b2b"
GRID = "#e4e3df"


def kinetic_energies(apply_config, extra):
    """({row index or extra key: energy}, [M, n_z] codes, run folder): the kinetic energy
    of every member's code and of each extra code (key -> function of the codes)."""
    _, cfg, _, codes, run, client_at = load_run(apply_config)
    template = load_shape(cfg.template)
    energy = {}
    for key, code in {**dict(enumerate(codes)), **{k: make(codes) for k, make in extra.items()}}.items():
        energy[key] = client_at(code).deform(template).trajectory.kinetic_energy().item()
    return energy, codes, run


def ranks(values):
    return np.argsort(np.argsort(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply-config", required=True, help="The run's cohort apply .yaml.")
    parser.add_argument("--output", default=None, help="PNG to write (default: <run>/atlas_centrality.png).")
    args = parser.parse_args()

    energy, codes, run = kinetic_energies(args.apply_config, {
        "mean": lambda codes: codes.mean(axis=0), "zero": lambda codes: np.zeros(codes.shape[1])})
    names, _ = load_codes(run)
    member_energy = np.array([energy[i] for i in range(len(codes))])
    mean = codes.mean(axis=0)
    _, s, vt = np.linalg.svd(codes - mean, full_matrices=False)
    ratio = s[:2] ** 2 / np.sum(s ** 2)
    scores = (codes - mean) @ vt[:2].T
    zero_xy = -mean @ vt[:2].T
    distance = np.linalg.norm(codes - mean, axis=1)
    rho = np.corrcoef(ranks(member_energy), ranks(distance))[0, 1]

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

    out_path = args.output or os.path.join(run, "atlas_centrality.png")
    fig.savefig(out_path, dpi=150)
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump({"units": "normalised frame", "code_mean": energy["mean"], "code_zero": energy["zero"],
                   "spearman_energy_vs_code_distance": rho,
                   "members": {names[i]: e for i, e in enumerate(member_energy)}}, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
        main()
