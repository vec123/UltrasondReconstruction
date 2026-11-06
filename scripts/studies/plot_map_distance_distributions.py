"""Plot map_1x point-distance distributions for one subject."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def read_distance_field(path, field_name):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    array = reader.GetOutput().GetPointData().GetArray(field_name)
    if array is None:
        raise ValueError(f"'{field_name}' not found in {path}")
    return vtk_to_numpy(array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="1")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = os.path.join("outputs", "baby_ssm", args.subject, "end_to_end")
    models = {
        "PCA": "pca",
        "PCA x Gaussian": "pca_times_gaussian",
        "PCA x Wendland": "pca_times_wendland",
    }
    distances = {
        label: {
            "recon_to_target": read_distance_field(
                os.path.join(root, folder, "eval_map", "recon_w_distance.vtp"),
                "recon_dist_2_target",
            ),
            "target_to_recon": read_distance_field(
                os.path.join(root, folder, "eval_map", "target_w_distance.vtp"),
                "target_dist_2_recon",
            ),
        }
        for label, folder in models.items()
    }
    bins = np.histogram_bin_edges(
        np.concatenate(
            [values[direction] for values in distances.values() for direction in values]
        ),
        bins=100,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    colors = ("#1f77b4", "#d95f02", "#2ca02c")
    for (label, values), color in zip(distances.items(), colors):
        axes[0].hist(
            values["recon_to_target"],
            bins=bins,
            histtype="step",
            linewidth=2.0,
            label=label,
            color=color,
        )
        axes[1].hist(
            values["target_to_recon"],
            bins=bins,
            histtype="step",
            linewidth=1.6,
            label=label,
            color=color,
        )
    axes[0].set_title("Reconstruction to target")
    axes[1].set_title("Target to reconstruction")
    for axis in axes:
        axis.set_xlabel("Distance")
        axis.set_ylabel("Number of points")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle(f"Map 1x distance distributions, subject {args.subject}")

    output = args.output or os.path.join(root, "map_distance_distributions.png")
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
