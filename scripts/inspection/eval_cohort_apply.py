"""Score every member of a cohort run's apply against its fused scan, as the Casos_Mar cohort runs are scored.

For each <run>/apply/<case>/world/deformed.vtp:
  * with --similarity (a run in a shape model's frame, not mm): carried into the scans' mm
    frame, x' = s R x + t (the similarity.json of scripts/SSM/target_to_ssm_frame.py), and
    written to <case>/world_mm/deformed.vtp;
  * scored by the recon_target_distance stage into <case>/eval/ (distance fields, coverage and
    distribution plots, landmark distances), against the SAME target scan and reference
    landmarks that --reference-run's eval of that case used (read from its
    distance_summary.json and landmark_distances.json), so the numbers compare directly.

Writes <run>/eval_summary.json: per case and over all cases (mean and median), the scan ->
recon distance (mean, median, share within 1/2/3 mm), the recon -> scan mean and the
landmark distance mean, each next to --reference-run's value for the same case.

Run with:
    python scripts/inspection/eval_cohort_apply.py \\
        --run outputs/stage2_flow/<run> \\
        --similarity /home/victor/Escritorio/Data/fused_targets/similarity.json
"""

import argparse
import importlib.util
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.inspection.code_pca_views import similarity_map  # noqa: E402
from src.config.US.schema import ReconTargetDistanceConfig  # noqa: E402
from src.vtk.io import load_vtp, save_vtp  # noqa: E402
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy  # noqa: E402

REFERENCE_RUN = "outputs/Casos_Mar/resnet_lddmm_cohort_autodecoder_fieldweighted_direct_g4c8"


def load_stage():
    """The recon_target_distance stage module (scripts/US/pipeline/stages is not a package path)."""
    path = os.path.join(ROOT, "scripts", "US", "pipeline", "stages", "recon_target_distance.py")
    spec = importlib.util.spec_from_file_location("recon_target_distance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case_numbers(eval_dir):
    """The numbers eval_summary.json reports for one case's eval/ folder."""
    with open(os.path.join(eval_dir, "distance_summary.json")) as f:
        distance = json.load(f)
    with open(os.path.join(eval_dir, "landmark_distances.json")) as f:
        landmarks = json.load(f)
    scan = distance["target_dist_2_recon"]
    return {"scan_to_recon_mean": scan["mean"], "scan_to_recon_median": scan["median"],
            **{f"scan_within_{t}mm": scan["fraction_within"][t] for t in ("1", "2", "3")},
            "recon_to_scan_mean": distance["recon_dist_2_target"]["mean"],
            "landmark_mean": landmarks["summary"]["mean"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="the cohort run; its apply/ must exist")
    parser.add_argument("--similarity", default=None,
                        help="similarity.json mapping the run's world frame to mm; default: world is mm")
    parser.add_argument("--reference-run", default=REFERENCE_RUN,
                        help="a scored cohort run: each case's target scan and landmarks come from its eval")
    parser.add_argument("--cases", nargs="+", default=None, help="case folders to score (default: all)")
    args = parser.parse_args()

    stage = load_stage()
    to_mm = similarity_map(args.similarity)
    apply_dir = os.path.join(args.run, "apply")
    cases = args.cases or sorted(n for n in os.listdir(apply_dir) if os.path.isdir(os.path.join(apply_dir, n)))
    rows = {}
    for case in cases:
        reference_eval = os.path.join(args.reference_run, "apply", case, "eval")
        if not os.path.isfile(os.path.join(reference_eval, "distance_summary.json")):
            print(f"[eval] {case}: no eval in {args.reference_run}, skipped")
            continue
        with open(os.path.join(reference_eval, "distance_summary.json")) as f:
            target_path = json.load(f)["target_path"]
        with open(os.path.join(reference_eval, "landmark_distances.json")) as f:
            landmarks_path = json.load(f)["landmarks_path"]

        recon_path = os.path.join(apply_dir, case, "world", "deformed.vtp")
        if args.similarity is not None:
            recon = load_vtp(recon_path)
            points = to_mm(vtk_to_numpy(recon.GetPoints().GetData()))
            recon.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float32), deep=True))
            recon_path = os.path.join(apply_dir, case, "world_mm", "deformed.vtp")
            os.makedirs(os.path.dirname(recon_path), exist_ok=True)
            save_vtp(recon, recon_path)

        eval_dir = os.path.join(apply_dir, case, "eval")
        stage.run(ReconTargetDistanceConfig(target_path=target_path, recon_path=recon_path,
                                            landmarks_path=landmarks_path, units="mm (original scan frame)",
                                            output_dir=eval_dir))
        rows[case] = {"run": case_numbers(eval_dir), "reference": case_numbers(reference_eval)}

    keys = list(next(iter(rows.values()))["run"])
    overall = {who: {stat: {key: float(fn([r[who][key] for r in rows.values()])) for key in keys}
                     for stat, fn in (("mean", np.mean), ("median", np.median))}
               for who in ("run", "reference")}
    out = {"run": args.run, "reference_run": args.reference_run, "similarity": args.similarity,
           "cases": len(rows), "overall": overall, "per_case": rows}
    out_path = os.path.join(args.run, "eval_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] {len(rows)} case(s), means over cases (run | reference):")
    for key in keys:
        print(f"[eval]   {key:22s} {overall['run']['mean'][key]:8.3f} | {overall['reference']['mean'][key]:8.3f}")
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
