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

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from src.evaluation.config import DistanceDistributionOverlayConfig as Config
from src.evaluation.distance_plots import (
    DIRECTIONS,
    bin_edges,
    coverage_entry,
    draw_coverage_overlay,
    draw_landmark_dots,
    draw_overlay,
    landmark_stats,
    read_distance_field,
    summarize,
)
from src.evaluation.metrics import COVERAGE_PERCENT_STEP
from src.shared.vtk.io import load_vtp


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
        edges, beyond, cut_note = bin_edges(series, cfg.bins, cfg.clip_percentile)
        footnote = chr(10).join(note for note in (cut_note, mismatch) if note)

        draw_overlay(
            series, edges, cfg, titles[direction], footnote,
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
