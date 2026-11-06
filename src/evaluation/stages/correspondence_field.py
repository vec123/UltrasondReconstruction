"""Stage: the correspondence field of one case. Matches the recon to the target and writes it onto copies of both.

The case folder is read by src.evaluation.loader (any registered
layout). The recon is the query: each recon vertex looks for a target point.
The outputs go to `output_dir`, by default <case_dir>/<name>/, where <name>
spells out the method and every one of its settings:

    target_w_correspondence.vtp   the target + recon_vertex_id, correspondence_*, matching_normal
    recon_w_correspondence.vtp    the recon + target_point_id, correspondence_*, matching_normal
    matching_lines.vtp            one line per pair
    correspondence_summary.json   settings, coverage and distance statistics

``matching_normal`` is the normal the normal check used, which is not always
the ``normal`` array of the input file (see DirectFlowEvalLayout).
correspondence_views.py draws these folders.
"""

import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from src.evaluation.config import CorrespondenceFieldConfig as Config
from src.evaluation.loader import load_pair
from src.shared.correspondence import CORRESPONDENCE, build_correspondence
from src.shared.correspondence.vtk import save_correspondence_field, save_matching_lines
from src.shared.vtk.fields import add_point_field
from src.shared.vtk.io import load_vtp


def recon_frame_kwargs(pair, method, kwargs, frame):
    """``kwargs`` with every distance setting of ``method`` converted from ``frame`` to the recon frame."""
    distances = CORRESPONDENCE.get(method).distance_kwargs
    return {key: pair.to_recon_frame(value, frame) if key in distances and value is not None else value
            for key, value in kwargs.items()}


def _format(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def output_name(strategy, given_kwargs, frame):
    """``correspondence_<method>_<setting>-<value>_...`` over every setting, defaults included.

    Distance settings show the value as given, followed by its frame.
    """
    distances = type(strategy).distance_kwargs
    settings = {f.name: getattr(strategy, f.name) for f in fields(strategy)}
    settings.update({key: value for key, value in given_kwargs.items() if key in distances})
    parts = [f"{key}-{_format(value)}{frame if key in distances and value is not None else ''}"
             for key, value in sorted(settings.items())]
    return "_".join(["correspondence", strategy.name, *parts])


def distance_stats(values):
    if not values.size:
        return {"n": 0}
    return {"n": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def side_summary(matched_idx, n_points):
    matched = int(np.unique(matched_idx).size)
    return {"matched": matched, "total": int(n_points), "fraction": matched / n_points}


def run(cfg: Config):
    case_dir = Path(cfg.case_dir)
    pair = load_pair(case_dir, layout=cfg.layout, kwargs=cfg.layout_kwargs)
    given = dict(cfg.kwargs)
    strategy = build_correspondence(cfg.method, recon_frame_kwargs(pair, cfg.method, given, cfg.distance_frame))
    print(pair.summary())
    print(f"method: {strategy.describe()}")

    matches = strategy.estimate(pair.recon, pair.target)
    if matches is None or matches.is_empty:
        raise ValueError(f"{cfg.method} produced no pairs for {pair.case_id}")
    print(matches.summary())

    # Filled in here, so the manifest entry of the run goes into this folder.
    cfg.output_dir = cfg.output_dir or str(case_dir / output_name(strategy, given, cfg.distance_frame))
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    recon_poly = load_vtp(str(pair.recon_path))
    target_poly = load_vtp(str(pair.target_path))
    add_point_field(recon_poly, pair.recon.normals, "matching_normal")
    add_point_field(target_poly, pair.target.normals, "matching_normal")
    written = save_correspondence_field(
        recon_poly, target_poly, matches,
        target_path=out / "target_w_correspondence.vtp", recon_path=out / "recon_w_correspondence.vtp",
    )
    lines_path = out / "matching_lines.vtp"
    save_matching_lines(pair.recon.points, pair.target.points, matches, filename=str(lines_path))

    summary = {
        "case_id": pair.case_id,
        "layout": pair.layout,
        "recon_path": str(pair.recon_path),
        "target_path": str(pair.target_path),
        "method": strategy.name,
        "kwargs": given,
        "distance_frame": cfg.distance_frame,
        "kwargs_recon_frame": {f.name: getattr(strategy, f.name) for f in fields(strategy)},
        "recon_to_surface_scale": None if pair.recon_to_surface is None else pair.recon_to_surface.scale,
        "n_pairs": len(matches),
        "target": side_summary(matches.target_idx, len(pair.target)),
        "recon": side_summary(matches.source_idx, len(pair.recon)),
        "distance_recon_frame": distance_stats(matches.distance),
        "distance_surface_frame": (None if pair.recon_to_surface is None
                                   else distance_stats(matches.distance * pair.recon_to_surface.scale)),
        "outputs": [str(p) for p in (*written, lines_path)],
    }
    summary_path = out / "correspondence_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    t = summary["target"]
    print(f"target: {t['matched']} of {t['total']} points matched ({t['fraction']:.1%})")
    print(f"wrote {out}")
