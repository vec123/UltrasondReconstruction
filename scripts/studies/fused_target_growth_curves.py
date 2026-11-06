"""Overlay every subject's align_to_template scale factor against gestational age:
a line for a subject with more than one fused target, a lone point for a subject
with only one - a growth-curve QC plot.

The scale plotted is session_scale (fused_target_ranking.py): the median
align_to_template fit per session, composed with align_to_reference's own scale.
It maps a session's own real mm landmarks onto ONE FIXED template
(assets/templates/fetus_mean.vtp), so it is inversely related to the subject's true size
at that week - a SMALLER scale means a session needed MORE shrinking to reach the
template's fixed size, i.e. a bigger true subject. A session with no saved
transform is skipped (not interpolated or dropped silently - see the printed count).

Casos_Maternitat has no week folders at all (no gestational age anywhere in its
source layout - checked, there is none), so it has no x position on this chart
and is left out entirely, not placed at a guessed location; every Casos_Maternitat
session is a "no week" skip in the printed count.

Two categories only, so they are colour-coded by dataset in the fixed categorical
order (slot 1 Mar, slot 2 Dex - dataviz skill palette), with a legend; each
subject's own line/point carries no separate colour or label (hundreds of subjects
is too many to colour individually - see dataviz anti-patterns on categorical
series count).

Run with:
    python scripts/studies/fused_target_growth_curves.py \
        --root /home/victor/Escritorio/Data/fused_scan_processing \
        --output /home/victor/Escritorio/Data/fused_scan_processing/fused_target_growth_curves.png
"""

import argparse
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.studies.fused_target_ranking import find_targets, session_scale  # noqa: E402
from scripts.studies.fused_target_trajectories import parse_week  # noqa: E402

# dataviz skill reference palette, categorical slots 1 and 2 (fixed order, not cycled).
DATASET_COLOR = {"Casos_Mar": "#2a78d6", "Estudio_Dexeus": "#eb6834"}
DATASET_LABEL = {"Casos_Mar": "Casos Mar", "Estudio_Dexeus": "Estudio Dexeus"}
TEXT_PRIMARY, TEXT_SECONDARY, GRID_COLOR, SURFACE = "#0b0b0b", "#52514e", "#e3e2dc", "#fcfcfb"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", nargs="+", required=True,
                        help="One or more fused_scan_processing dataset roots to search (recursively).")
    parser.add_argument("--output", required=True, help="PNG to write.")
    args = parser.parse_args()

    targets = find_targets(args.root)
    all_subjects = defaultdict(list)  # (dataset, case) -> [(week_num, scale), ...]
    missing_scale, no_week = 0, 0
    for label, dataset, case, week, fused_path, ref_path in targets:
        scale = session_scale(fused_path)
        if scale is None:
            missing_scale += 1
            continue
        week_num = parse_week(week)
        if math.isinf(week_num):
            no_week += 1
            continue
        all_subjects[(dataset, case)].append((week_num, scale))
    subjects = {key: sorted(sessions) for key, sessions in all_subjects.items()}
    n_lines = sum(1 for s in subjects.values() if len(s) > 1)
    n_points = sum(1 for s in subjects.values() if len(s) == 1)
    print(f"{len(subjects)} subjects plotted ({n_lines} as a line, {n_points} as a lone point); "
         f"{missing_scale} sessions skipped (no saved align_to_template transform), "
         f"{no_week} skipped (no gestational week - Casos_Maternitat)")

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    seen_dataset = set()
    for (dataset, case), sessions in subjects.items():
        weeks, scales = zip(*sessions)
        color = DATASET_COLOR.get(dataset, TEXT_SECONDARY)
        label = DATASET_LABEL.get(dataset, dataset) if dataset not in seen_dataset else None
        seen_dataset.add(dataset)
        ax.plot(weeks, scales, color=color, linewidth=1.2, alpha=0.55, marker="o", markersize=3.5,
               markeredgewidth=0, label=label, zorder=2)

    ax.set_xlabel("gestational age (weeks)", color=TEXT_SECONDARY, fontsize=11)
    ax.set_ylabel("align_to_template scale factor (smaller = bigger true subject)",
                 color=TEXT_SECONDARY, fontsize=11)
    ax.set_title(f"{len(subjects)} subjects ({n_lines} lines, {n_points} single points): "
                f"align_to_template scale vs. gestational age",
                color=TEXT_PRIMARY, fontsize=13, pad=12)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    legend = ax.legend(frameon=False, loc="upper right", fontsize=10)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(args.output, facecolor=SURFACE)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
