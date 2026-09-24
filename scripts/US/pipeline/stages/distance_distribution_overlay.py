"""Stage: overlay several reconstructions' point-distance distributions.

`recon_target_distance` writes, per reconstruction, a recon_w_distance.vtp
carrying 'recon_dist_2_target' and a target_w_distance.vtp carrying
'target_dist_2_recon'. This stage reads those fields back from any number of
such runs and draws them together as histogram *lines*, one figure per
direction:

- <output_dir>/recon_to_target_distribution.png
- <output_dir>/target_to_recon_distribution.png

Both figures share one set of bin edges per direction (computed over the pooled
distances of every source), so the lines are directly comparable. Counts are
normalized to the share of each mesh's points by default - reconstructions from
different models rarely have the same vertex count, and raw counts would make
the denser mesh look worse than it is.

Each direction also gets a distance-over-percentage figure - the share of that
mesh's points already within a given distance, read left to right:

- <output_dir>/recon_to_target_coverage.png
- <output_dir>/target_to_recon_coverage.png

That curve is what separates fits the mean cannot. Pooled target->recon means
sit within a few tenths of each other across very different models, because
they are dominated by the stretch of scan no reconstruction covers; the share
of the scan explained to within a millimetre is not, and the curves separate
cleanly there while their tails stay tangled.

Landmarks get their own comparison, off the landmark_distances.json beside
those meshes: <output_dir>/landmark_distances.png places every source's error
for every landmark on one row per landmark, and
<output_dir>/landmark_distance_overlay.json carries the numbers. Surface
distance and landmark distance do not have to agree - a model can lie close to
the scan everywhere and still put the eye corners in the wrong place - so this
answers a question the two histogram figures cannot.

A <output_dir>/distance_distribution_overlay.json records mean / median / p95 /
max per source, which is also the table view the plots' direct labels lean on,
and <output_dir>/distance_coverage_overlay.json carries the sampled curves
themselves, so a figure can be redrawn without re-reading the meshes.
"""

import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from src.config.US.schema import DistanceDistributionOverlayConfig as Config
from src.vtk.io import load_vtp

# One definition of the curve, shared with the stage that writes the meshes
# this one reads back.
from scripts.US.pipeline.stages.recon_target_distance import (
    COVERAGE_PERCENT_STEP,
    coverage_curve,
)


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


def _source_path(source, direction):
    """Where this source's mesh for `direction` lives.

    An explicit recon_path / target_path wins; otherwise the file sits under
    eval_dir under the name recon_target_distance gave it.
    """
    filename, field_name = DIRECTIONS[direction]
    override = source.recon_path if direction == "recon_to_target" else source.target_path
    if override:
        return override, field_name
    if not source.eval_dir:
        missing = "recon_path" if direction == "recon_to_target" else "target_path"
        raise ValueError(
            f"Source '{source.label}' sets neither eval_dir nor an explicit {missing}."
        )
    return os.path.join(source.eval_dir, filename), field_name


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
        "fraction_within": {
            f"{float(threshold):g}": float((values < threshold).mean())
            for threshold in coverage_thresholds
        },
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


def source_target(eval_dir):
    """Which scan recon_target_distance scored this eval against, as
    (display name, point count).

    It records the target's path in distance_summary.json beside the meshes,
    but the path alone does not identify the scan: every SSM fit gets its own
    rigid_alignment.vtp, all of them the same scan carried into that model's
    frame. The point count does identify it - the fused scan and the merged
    high-confidence scan differ by a factor of 32 - so that is what groups
    sources here. Returns (None, None) when the summary or mesh is unreadable
    (an eval_dir assembled by hand, or an older run).
    """
    try:
        with open(os.path.join(eval_dir, "distance_summary.json")) as f:
            target = json.load(f)["target_path"]
        n_points = len(vtk_to_numpy(load_vtp(str(target)).GetPoints().GetData()))
        return os.path.basename(str(target)), n_points
    except Exception:
        return None, None


def target_mismatch_note(sources):
    """A warning for the figures when the sources were not all scored against
    the same scan.

    Curves from different targets share an x-axis but not a question: a target
    containing geometry the other one doesn't (an unfused scan's outliers, say)
    inflates target->recon for reasons that have nothing to do with the
    reconstruction. Silently overlaying them produces a plot that looks like a
    comparison and isn't, so the figure says so rather than relying on whoever
    reads it to remember which eval used which target.
    """
    groups = {}
    for source in sources:
        name, n_points = source_target(source.eval_dir) if source.eval_dir else (None, None)
        groups.setdefault(n_points, (name, []))[1].append(source.label or "?")
    if len(groups) < 2:
        return ""
    described = "; ".join(
        f"{name or 'unknown'}"
        + (f" ({n:,} pts)" if n else "")
        + f": {', '.join(labels)}"
        for n, (name, labels) in groups.items()
    )
    return ("NOT ONE COMPARISON - these curves were scored against different targets. "
            f"{described}. Target-to-reconstruction is not comparable across them; "
            "reconstruction-to-target roughly is.")


def read_landmark_distances(eval_dir):
    """{label: distance} from the landmark_distances.json recon_target_distance
    wrote beside the meshes, or None when there isn't one (an eval_dir put
    together by hand, or a run from before that file existed)."""
    if not eval_dir:
        return None
    try:
        with open(os.path.join(eval_dir, "landmark_distances.json")) as f:
            payload = json.load(f)
        return {
            label: float(entry["distance"])
            for label, entry in payload["landmarks"].items()
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


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


def run_landmark_comparison(cfg):
    """The landmark figure and its JSON, if any source has the file.

    Returns the summary dict, or None when no source does - the meshes this
    stage reads can be there without it (recon_target_distance only started
    writing landmark_distances.json later), and the surface figures should not
    fail over a missing landmark file.
    """
    entries, missing = [], []
    for source in cfg.sources:
        label = source.label or os.path.basename(os.path.normpath(source.eval_dir))
        distances = read_landmark_distances(source.eval_dir)
        if distances:
            entries.append({"label": label, "color": source.color, "distances": distances})
        else:
            missing.append(label)

    if not entries:
        print("No source has a landmark_distances.json - skipping the landmark comparison.")
        return None

    # Sources built on different templates carry different landmark sets, and a
    # mean over a model's own set is not comparable with a mean over another's.
    # Every stat that names a model therefore runs over the shared subset, and
    # the per-label block keeps whatever that model actually has.
    common = sorted(set.intersection(*(set(entry["distances"]) for entry in entries)))
    union = set().union(*(set(entry["distances"]) for entry in entries))
    ordering = sorted(
        union,
        key=lambda label: -np.mean([
            entry["distances"][label] for entry in entries if label in entry["distances"]
        ]),
    )

    notes = []
    if len(common) < len(union):
        absent = ", ".join(sorted(union - set(common)))
        notes.append(
            f"Sources do not share one landmark set - {len(common)} of {len(union)} labels are "
            f"common; mean and rms per source are over those {len(common)} only. "
            f"Not in every source: {absent}."
        )
    if missing:
        notes.append("No landmark_distances.json, left out: " + ", ".join(missing) + ".")

    for note in notes:
        print("NOTE: " + note)

    draw_landmark_dots(
        entries, ordering, set(common), cfg, chr(10).join(notes),
        os.path.join(cfg.output_dir, cfg.landmark_plot_name),
    )

    summary = {
        "units": cfg.units,
        "common_labels": common,
        "sources_without_landmarks": missing,
        "note": chr(10).join(notes) or None,
        "sources": {
            entry["label"]: {
                "all_labels": landmark_stats(entry["distances"].values()),
                "shared_labels": landmark_stats(
                    [entry["distances"][label] for label in common]
                ),
                "landmarks": {
                    label: entry["distances"][label] for label in sorted(entry["distances"])
                },
            }
            for entry in entries
        },
    }
    path = os.path.join(cfg.output_dir, cfg.landmark_summary_name)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {path}")

    print(f"{chr(10)}{'source':<34}{'landmarks':>10}{'mean':>9}{'median':>9}{'max':>9}{'rms':>9}"
          f"   (over the {len(common)} shared landmarks)")
    for label, stats in summary["sources"].items():
        shared = stats["shared_labels"]
        print(f"{label:<34}{shared['n_landmarks']:>10}{shared['mean']:>9.3f}"
              f"{shared['median']:>9.3f}{shared['max']:>9.3f}{shared['rms']:>9.3f}")
    return summary


def run(cfg: Config):
    if not cfg.sources:
        raise ValueError("distance_distribution_overlay needs at least one entry under 'sources'.")
    if not cfg.output_dir:
        raise ValueError("distance_distribution_overlay requires output_dir.")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # {direction: [{label, color, values, path}, ...]}, read once and reused for
    # both the shared bin edges and the plots.
    loaded = {direction: [] for direction in DIRECTIONS}
    for source in cfg.sources:
        label = source.label or os.path.basename(os.path.normpath(source.eval_dir))
        for direction in DIRECTIONS:
            path, field_name = _source_path(source, direction)
            loaded[direction].append({
                "label": label,
                "color": source.color,
                "values": read_distance_field(path, field_name),
                "path": path,
            })

    plot_names = {
        "recon_to_target": cfg.recon_to_target_plot_name,
        "target_to_recon": cfg.target_to_recon_plot_name,
    }
    coverage_plot_names = {
        "recon_to_target": cfg.recon_to_target_coverage_plot_name,
        "target_to_recon": cfg.target_to_recon_coverage_plot_name,
    }
    titles = {
        "recon_to_target": cfg.recon_to_target_title,
        "target_to_recon": cfg.target_to_recon_title,
    }

    mismatch = target_mismatch_note(cfg.sources)
    if mismatch:
        print(os.linesep + "WARNING: " + mismatch + os.linesep)

    coverage = {
        "units": cfg.units,
        "percent_step": COVERAGE_PERCENT_STEP,
        "coverage_thresholds": list(cfg.coverage_thresholds),
        "target_mismatch": mismatch or None,
        "sources": {},
    }

    summary = {
        "units": cfg.units,
        "normalize": cfg.normalize,
        "bins": cfg.bins,
        "clip_percentile": cfg.clip_percentile,
        "target_mismatch": mismatch or None,
        "sources": {},
    }

    for direction, series in loaded.items():
        pooled = np.concatenate([entry["values"] for entry in series])
        lower = float(pooled.min())
        upper = (
            float(pooled.max())
            if cfg.clip_percentile >= 100
            else float(np.percentile(pooled, cfg.clip_percentile))
        )
        if upper <= lower:
            upper = float(pooled.max())
        bin_edges = np.linspace(lower, upper, cfg.bins + 1)

        # Per source, the share of points past the axis cut - drawn nowhere, so
        # the figure says so rather than letting the reader assume it saw all of it.
        beyond = {
            entry["label"]: float((entry["values"] > upper).mean()) for entry in series
        }
        notes = []
        if max(beyond.values()) > 0:
            worst = ", ".join(f"{label} {share:.1%}" for label, share in beyond.items() if share > 0)
            notes.append(
                f"x-axis cut at the {cfg.clip_percentile:g}th percentile of the pooled "
                f"distances ({upper:.2f}); points beyond it are not drawn: {worst}"
            )
        if mismatch:
            notes.append(mismatch)
        footnote = chr(10).join(notes)

        draw_overlay(
            series, bin_edges, cfg, titles[direction], footnote,
            os.path.join(cfg.output_dir, plot_names[direction]),
        )
        # The coverage figure needs no axis cut of its own - its y is a share
        # of the whole mesh, so nothing goes undrawn - hence the mismatch note
        # alone, without the histogram's "points beyond the axis" line.
        draw_coverage_overlay(
            series, cfg, titles[direction], mismatch,
            os.path.join(cfg.output_dir, coverage_plot_names[direction]),
        )

        for entry in series:
            curve = coverage_entry(entry["values"], cfg.coverage_thresholds)
            summary["sources"].setdefault(entry["label"], {})[direction] = {
                "path": entry["path"],
                "fraction_beyond_axis": beyond[entry["label"]],
                **summarize(entry["values"]),
                "fraction_within": curve["fraction_within"],
            }
            coverage["sources"].setdefault(entry["label"], {})[direction] = curve

    summary_path = os.path.join(cfg.output_dir, cfg.summary_name)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary_path}")

    coverage_path = os.path.join(cfg.output_dir, cfg.coverage_name)
    with open(coverage_path, "w") as f:
        json.dump(coverage, f, indent=2)
    print(f"Saved {coverage_path}")

    run_landmark_comparison(cfg)

    # The table view the plots' contrast relief leans on.
    columns = [f"{float(threshold):g}" for threshold in cfg.coverage_thresholds]
    header = "".join(f"{'<' + column:>9}" for column in columns)
    print(f"\n{'source':<34}{'direction':<18}{'mean':>10}{'median':>10}{'p95':>10}{header}")
    for label, directions in summary["sources"].items():
        for direction, stats in directions.items():
            shares = "".join(f"{stats['fraction_within'][column]:>9.1%}" for column in columns)
            print(
                f"{label:<34}{direction:<18}{stats['mean']:>10.4f}"
                f"{stats['median']:>10.4f}{stats['p95']:>10.4f}{shares}"
            )
