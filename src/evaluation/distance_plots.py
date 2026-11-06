"""The distance figures of the evaluation stages (distance_distribution_overlay, compare).

Several sources' point distances (the fields recon_target_distance writes) as
histogram lines over shared bins, as coverage curves (the share of points within
a distance), and their landmark errors as one row per landmark. Also the readers
and summaries those figures and the stages' JSON files share.
"""

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from src.shared.vtk.io import load_vtp

# One definition of the curve, shared with the stage that writes the meshes
# these figures read back.
from src.evaluation.metrics import coverage_curve, fraction_within


# Categorical slots 1-3 validate all-pairs for both normal and CVD vision;
# past three, series also differ by line style (see LINE_STYLES) so identity
# never rests on hue alone.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
LINE_STYLES = ("-", "--", "-.", ":")
# Scatter equivalent of LINE_STYLES, for the landmark figure: line style does
# not apply to a marker, so identity there rests on shape plus hue.
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

# Which mesh file carries which point field, per direction.
DIRECTIONS = {
    "recon_to_target": ("recon_w_distance.vtp", "recon_dist_2_target"),
    "target_to_recon": ("target_w_distance.vtp", "target_dist_2_recon"),
}


def read_distance_field(path, field_name):
    """The named point field of a .vtp, as a 1-D float array."""
    array = load_vtp(path).GetPointData().GetArray(field_name)
    if array is None:
        raise ValueError(f"Point field '{field_name}' not found in '{path}'")
    return vtk_to_numpy(array).astype(float).ravel()


def histogram_line(values, bin_edges, normalize):
    """Histogram heights for `values` on `bin_edges`, in the requested units.

    Points past the last edge are left out of the drawn line - folding them
    into the last bin would raise a spike there that reads as a real mode -
    but they still count towards the normalization, so the heights stay honest
    shares of the whole mesh and the missing mass shows up as a curve that
    doesn't integrate to 1. `run` reports that mass in the figure's footnote.
    """
    counts, _ = np.histogram(values, bins=bin_edges)
    if normalize == "count":
        return counts.astype(float)
    if normalize == "density":
        return counts / (values.size * np.diff(bin_edges))
    if normalize == "fraction":
        return counts / values.size
    raise ValueError(f"Unknown normalize mode '{normalize}' (use 'fraction', 'density' or 'count').")


def _y_label(normalize):
    return {
        "fraction": "Share of points",
        "density": "Probability density",
        "count": "Number of points",
    }[normalize]


def bin_edges(series, bins, clip_percentile):
    """Bin edges shared by every series, cut at clip_percentile of the pooled distances.

    Returns (edges, beyond, note): beyond is, per series label, the share of its
    points past the cut, and note the footnote that names them ("" if none).
    """
    pooled = np.concatenate([entry["values"] for entry in series])
    lower = float(pooled.min())
    upper = (
        float(pooled.max())
        if clip_percentile >= 100
        else float(np.percentile(pooled, clip_percentile))
    )
    if upper <= lower:
        upper = float(pooled.max())

    # Per source, the share of points past the axis cut - drawn nowhere, so
    # the figure says so rather than letting the reader assume it saw all of it.
    beyond = {
        entry["label"]: float((entry["values"] > upper).mean()) for entry in series
    }
    note = ""
    if max(beyond.values()) > 0:
        worst = ", ".join(f"{label} {share:.1%}" for label, share in beyond.items() if share > 0)
        note = (
            f"x-axis cut at the {clip_percentile:g}th percentile of the pooled "
            f"distances ({upper:.2f}); points beyond it are not drawn: {worst}"
        )
    return np.linspace(lower, upper, bins + 1), beyond, note


def draw_overlay(series, bin_edges, cfg, title, footnote, output_path):
    """One figure: every source's histogram as a line over shared bins.

    Every curve's own median and 95th percentile ride in its legend entry, so
    the figure carries the table that separates lines which overlap almost
    everywhere - and so identity never rests on a legend swatch alone, which
    the lighter palette slots are too low-contrast to carry by themselves.
    """
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Wrap the footnote first: it is drawn with fig.text, which constrained
    # layout does not measure, so the space it needs has to be reserved up
    # front or the engine collapses the axes into it.
    note_lines = [
        wrapped
        for line in (footnote.splitlines() if footnote else [])
        for wrapped in textwrap.wrap(line, width=118) or [""]
    ]
    note_height = 0.20 * len(note_lines)

    fig, axis = plt.subplots(figsize=(9.5, 5.5 + note_height), constrained_layout=True)
    if note_lines:
        reserved = note_height / (5.5 + note_height)
        fig.get_layout_engine().set(rect=(0, reserved, 1, 1 - reserved))
    for index, entry in enumerate(series):
        values = entry["values"]
        color = entry["color"] or PALETTE[index % len(PALETTE)]
        heights = histogram_line(values, bin_edges, cfg.normalize)
        label = (
            f"{entry['label']}   median {np.median(values):.2f}"
            f"   p95 {np.percentile(values, 95):.2f}"
        )
        axis.plot(
            centers,
            heights,
            color=color,
            linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            linewidth=2.0,
            label=label,
        )
        axis.fill_between(centers, heights, color=color, alpha=0.07, linewidth=0)

    axis.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=13)
    axis.set_xlabel(f"Distance ({cfg.units})", color=TEXT_SECONDARY)
    axis.set_ylabel(_y_label(cfg.normalize), color=TEXT_SECONDARY)
    axis.set_xlim(bin_edges[0], bin_edges[-1])
    if cfg.log_y:
        axis.set_yscale("log")
    else:
        axis.set_ylim(bottom=0)
    axis.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)
    axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#d5d4cf")
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for i, line in enumerate(note_lines):
        fig.text(
            0.012,
            (note_height - 0.16 * (i + 0.8)) / (5.5 + note_height),
            line,
            fontsize=8,
            color=TEXT_SECONDARY,
        )

    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


def draw_coverage_overlay(series, cfg, title, footnote, output_path):
    """One figure: distance on x, the share of each source's points already
    within it on y. Same series, colors and line styles as the histogram
    figure beside it, so the two read as one pair."""
    note_lines = [
        wrapped
        for line in (footnote.splitlines() if footnote else [])
        for wrapped in textwrap.wrap(line, width=118) or [""]
    ]
    note_height = 0.20 * len(note_lines)

    fig, axis = plt.subplots(figsize=(9.5, 5.5 + note_height), constrained_layout=True)
    if note_lines:
        reserved = note_height / (5.5 + note_height)
        fig.get_layout_engine().set(rect=(0, reserved, 1, 1 - reserved))

    upper = 0.0
    for index, entry in enumerate(series):
        values = entry["values"]
        color = entry["color"] or PALETTE[index % len(PALETTE)]
        percent, distance = coverage_curve(values)
        label = entry["label"] + "".join(
            f"   {(values < threshold).mean():.0%} < {threshold:g}"
            for threshold in cfg.coverage_thresholds[:2]
        )
        axis.plot(
            distance,
            percent,
            color=color,
            linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            linewidth=2.0,
            label=label,
        )
        # The x range follows the 90th percentile rather than the clip
        # percentile the histograms use: past that these curves are all but
        # flat, and letting the tail in compresses the stretch where they
        # actually separate into the first inch of the axis.
        upper = max(upper, float(np.percentile(values, 90)))

    axis.set_xlim(0, upper)
    axis.set_ylim(0, 100)
    for threshold in cfg.coverage_thresholds:
        if threshold <= upper:
            axis.axvline(threshold, color="#9a9a94", linewidth=0.8, linestyle=":")
    axis.set_title(
        f"{title} - share within a given distance",
        color=TEXT_PRIMARY, loc="left", fontsize=13,
    )
    axis.set_xlabel(f"Distance in {cfg.units}", color=TEXT_SECONDARY)
    axis.set_ylabel("Points within that distance (%)", color=TEXT_SECONDARY)
    axis.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, loc="lower right")
    axis.grid(alpha=0.2, linewidth=0.6)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#d5d4cf")
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for i, line in enumerate(note_lines):
        fig.text(
            0.012,
            (note_height - 0.16 * (i + 0.8)) / (5.5 + note_height),
            line,
            fontsize=8,
            color=TEXT_SECONDARY,
        )

    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


def coverage_entry(values, coverage_thresholds):
    """One source's sampled curve, for distance_coverage_overlay.json."""
    percent, distance = coverage_curve(values)
    return {
        "n_points": int(values.size),
        "percent": [round(float(value), 4) for value in percent],
        "distance": [round(float(value), 6) for value in distance],
        "fraction_within": fraction_within(values, coverage_thresholds),
    }


def summarize(values):
    return {
        "n_points": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "rms": float(np.sqrt(np.mean(values ** 2))),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def landmark_stats(values):
    """mean / median / max / rms over one set of landmark distances."""
    values = np.asarray(list(values), dtype=float)
    if not values.size:
        return {"n_landmarks": 0}
    return {
        "n_landmarks": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "rms": float(np.sqrt(np.mean(values ** 2))),
    }


def draw_landmark_dots(entries, labels, common, cfg, footnote, output_path):
    """One row per landmark, one marker per source.

    A row-per-landmark dot plot rather than grouped bars: the question is which
    landmark each model misses and by how much, and bars spend their length on
    a baseline every reader already knows is zero. Rows are ordered worst-first
    by the mean across sources, so the landmarks that decide the comparison are
    the ones at the top.
    """
    note_lines = [
        wrapped
        for line in (footnote.splitlines() if footnote else [])
        for wrapped in textwrap.wrap(line, width=118) or [""]
    ]
    note_height = 0.20 * len(note_lines)
    height = 2.6 + 0.32 * len(labels) + note_height

    fig, axis = plt.subplots(figsize=(9.5, height), constrained_layout=True)
    if note_lines:
        reserved = note_height / height
        fig.get_layout_engine().set(rect=(0, reserved, 1, 1 - reserved))

    positions = {label: len(labels) - 1 - index for index, label in enumerate(labels)}
    for index, label in enumerate(labels):
        if index % 2 == 0:
            axis.axhspan(positions[label] - 0.5, positions[label] + 0.5,
                         color="#f4f3f0", linewidth=0, zorder=0)
        if label not in common:
            # Not every model carries every landmark - the fetus and baby
            # templates define different sets - so a row that is not shared by
            # all of them is marked rather than silently read as a comparison.
            axis.text(0.0, positions[label] + 0.34, "not in every model",
                      fontsize=6.5, color=TEXT_SECONDARY, va="center", zorder=4)

    span = 0.66
    for index, entry in enumerate(entries):
        color = entry["color"] or PALETTE[index % len(PALETTE)]
        offset = span * ((index - (len(entries) - 1) / 2) / max(len(entries), 2))
        shared = [entry["distances"][label] for label in common if label in entry["distances"]]
        stats = landmark_stats(shared)
        axis.scatter(
            [entry["distances"][label] for label in labels if label in entry["distances"]],
            [positions[label] + offset for label in labels if label in entry["distances"]],
            s=26,
            color=color,
            marker=MARKERS[index % len(MARKERS)],
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
            label=f"{entry['label']}   mean {stats.get('mean', float('nan')):.2f}"
                  f"   rms {stats.get('rms', float('nan')):.2f}",
        )

    axis.set_yticks([positions[label] for label in labels])
    axis.set_yticklabels(labels, fontsize=9)
    axis.set_ylim(-0.6, len(labels) - 0.4)
    axis.set_xlim(left=0)
    axis.set_title("Landmark error per landmark", color=TEXT_PRIMARY, loc="left", fontsize=13)
    axis.set_xlabel(f"Distance to the subject's landmark, in {cfg.units}", color=TEXT_SECONDARY)
    axis.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_PRIMARY,
                loc="lower right", title="mean / rms over shared landmarks",
                title_fontsize=8)
    axis.grid(axis="x", alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#d5d4cf")
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    for i, line in enumerate(note_lines):
        fig.text(0.012, (note_height - 0.16 * (i + 0.8)) / height, line,
                 fontsize=8, color=TEXT_SECONDARY)

    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")

