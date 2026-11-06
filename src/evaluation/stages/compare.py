"""Stage: several models, compared over the cases that all of them have.

Each entry of `models` is the pattern of one model's evaluation folder of a
case - the output_dir of recon_target_distance:

    models:
      gpmm_map: ${OUT}/fetus_ssm/${SESSION}/end_to_end/init_pca/eval_map
      flow_g8:  ${OUT}/resnet_lddmm_cohort_autodecoder_fieldweighted_g8/apply/${CASE_KEY}/eval

The variables without a value (SUBJECT_ID and WEEK, inside SESSION and
CASE_KEY) are the case variables. The engine leaves them in this field
(StageSpec.patterns), and find_paths finds their values on the disk, as for
`targets: match:`. Only the cases that every model has are compared; the
others are listed. A variable with a value selects: `--case '{"SUBJECT_ID": "1"}'`
compares the cases of subject 1 only.

Writes into output_dir:
- compare_table.csv: one row per case; per model the mean, median and p95 of
  both directions, the share within each coverage threshold, and the mean
  landmark distance (empty for a folder without landmark_distances.json).
- compare_summary.json: the cases, the cases each model does not have, the
  evaluation folders, and per model and column the mean and median over the
  cases.
- recon_to_target_distribution.png, target_to_recon_distribution.png and the
  two *_coverage.png: per model the points of all cases together, drawn by
  src/evaluation/distance_plots.py, as distance_distribution_overlay draws them.
"""

import csv
import dataclasses
import json
import os

import numpy as np

from src.evaluation.config import CompareConfig as Config
from src.evaluation.distance_plots import (
    DIRECTIONS,
    bin_edges,
    draw_coverage_overlay,
    draw_overlay,
    read_distance_field,
    summarize,
)
from src.evaluation.metrics import fraction_within
from src.shared.core.pipeline.targets import find_case_paths

MARKER = "distance_summary.json"  # recon_target_distance writes it last
LANDMARK_FILE = "landmark_distances.json"
TABLE_FILE = "compare_table.csv"
SUMMARY_FILE = "compare_summary.json"
STATS = ("mean", "median", "p95")
TITLES = {"recon_to_target": "Reconstruction to target", "target_to_recon": "Target to reconstruction"}


def find_evals(model, pattern):
    """{case name: evaluation folder} of every folder that fits `pattern`, in path order."""
    evals = {name: os.path.dirname(path)
             for name, (_, path) in find_case_paths(os.path.join(pattern, MARKER), f"Model '{model}'").items()}
    if not evals:
        raise ValueError(f"No evaluation folder ({MARKER}) fits the pattern of model '{model}': {pattern}")
    return evals


def case_columns(eval_dir, thresholds):
    """The columns of one evaluation folder, and its distances per direction."""
    columns, distances = {}, {}
    for direction, (filename, field_name) in DIRECTIONS.items():
        values = read_distance_field(os.path.join(eval_dir, filename), field_name)
        stats = summarize(values)
        columns.update({f"{direction}.{stat}": stats[stat] for stat in STATS})
        columns.update({f"{direction}.within_{threshold}": share
                        for threshold, share in fraction_within(values, thresholds).items()})
        distances[direction] = values
    columns["landmarks.mean"] = None
    landmarks = os.path.join(eval_dir, LANDMARK_FILE)
    if os.path.isfile(landmarks):
        with open(landmarks) as f:
            columns["landmarks.mean"] = json.load(f)["summary"]["mean"]
    return columns, distances


def folder_units(eval_dirs):
    """The one `units` of the evaluation folders; an error when they differ."""
    units = set()
    for eval_dir in eval_dirs:
        with open(os.path.join(eval_dir, MARKER)) as f:
            units.add(json.load(f).get("units"))
    if len(units) > 1:
        raise ValueError(f"The evaluation folders have different units {sorted(map(str, units))}; "
                         f"compare only folders of one frame.")
    return units.pop()


def over_cases(rows):
    """{column: {mean, median, n_cases}} over the cases that have a value."""
    result = {}
    for column in rows[0]:
        values = [row[column] for row in rows if row[column] is not None]
        result[column] = (None if not values else
                          {"mean": float(np.mean(values)), "median": float(np.median(values)), "n_cases": len(values)})
    return result


def run(cfg: Config):
    if not cfg.models:
        raise ValueError("compare needs at least one entry under 'models'.")
    if not cfg.output_dir:
        raise ValueError("compare requires output_dir.")

    evals = {model: find_evals(model, pattern) for model, pattern in cfg.models.items()}
    every = set().union(*evals.values())
    cases = [case for case in next(iter(evals.values())) if all(case in found for found in evals.values())]
    missing = {model: sorted(every - set(found)) for model, found in evals.items()}
    if not cases:
        raise ValueError(f"The models have no case in common: "
                         f"{ {model: len(found) for model, found in evals.items()} } cases.")
    for model, names in missing.items():
        if names:
            print(f"WARNING: model '{model}' has no evaluation of {len(names)} case(s), left out: {names}")
    units = cfg.units or folder_units(evals[model][case] for model in evals for case in cases)
    print(f"Comparing {len(evals)} models on {len(cases)} cases ({units}).")

    rows = {model: [] for model in evals}  # model -> the columns of each case
    pooled = {direction: {model: [] for model in evals} for direction in DIRECTIONS}
    for case in cases:
        for model in evals:
            columns, distances = case_columns(evals[model][case], cfg.coverage_thresholds)
            rows[model].append(columns)
            for direction, values in distances.items():
                pooled[direction][model].append(values)

    os.makedirs(cfg.output_dir, exist_ok=True)
    table_path = os.path.join(cfg.output_dir, TABLE_FILE)
    header = ["case"] + [f"{model}.{column}" for model in evals for column in rows[model][0]]
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for index, case in enumerate(cases):
            writer.writerow([case] + [rows[model][index][column] for model in evals for column in rows[model][0]])
    print(f"Saved {table_path}")

    summary = {
        "units": units,
        "models": dict(cfg.models),
        "cases": cases,
        "missing": missing,
        "eval_dirs": {model: {case: evals[model][case] for case in cases} for model in evals},
        "over_cases": {model: over_cases(rows[model]) for model in evals},
    }
    summary_path = os.path.join(cfg.output_dir, SUMMARY_FILE)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary_path}")

    plot_cfg = dataclasses.replace(cfg, units=units)
    for direction, by_model in pooled.items():
        series = [{"label": model, "color": "", "values": np.concatenate(parts)} for model, parts in by_model.items()]
        edges, _, note = bin_edges(series, cfg.bins, cfg.clip_percentile)
        title = f"{TITLES[direction]}, {len(cases)} cases"
        draw_overlay(series, edges, plot_cfg, title, note, os.path.join(cfg.output_dir, f"{direction}_distribution.png"))
        draw_coverage_overlay(series, plot_cfg, title, "", os.path.join(cfg.output_dir, f"{direction}_coverage.png"))

    shown = ["recon_to_target.median", "target_to_recon.median", "landmarks.mean"]
    print(f"\nmean over cases ({units})\n{'model':<34}" + "".join(f"{column:>26}" for column in shown))
    for model, stats in summary["over_cases"].items():
        print(f"{model:<34}" + "".join(
            f"{'-' if stats[column] is None else format(stats[column]['mean'], '.4f'):>26}" for column in shown))
