"""Run a small non-chamfer Baby SSM model sweep and evaluate map fits."""

from copy import deepcopy
from pathlib import Path
import json
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.SSM.client import ConfigClient
from src.config.SSM.schema import PipelineConfig
from src.SSM.runners.pipeline import PipelineRunner
from scripts.US.pipeline.stages import recon_target_distance


SUBJECT = "1"
TARGET = ROOT / "outputs/fetus_ssm/1/end_to_end/fused_scan_ssm/fused_scan.vtp"
LANDMARKS = ROOT / "outputs/1/20_semanas/aligned_to_template/landmarks_csv/1-20s-01.csv"
TEMPLATE = ROOT / "Templates/baby_mean.vtp"
MODEL_ROOT = ROOT / "src/SSM/saved_models/baby"
OUTPUT_ROOT = ROOT / "outputs/baby_ssm/1/model_sweep"

# Base weights are anchored to the existing calibrated normalized-mahalanobis
# Baby sweep: PCA=0.01536, PCAxGaussian=0.1967, PCAxWendland=0.382.
CANDIDATES = {
    "pca_sum_wendland": ("baby_pca_sum_wendland.npz", 0.01536),
    "pca_sum_matern32": ("baby_pca_sum_matern32.npz", 0.01536),
    "pca_times_matern32": ("baby_pca_times_matern32.npz", 0.30),
    "matern52": ("baby_matern52.npz", 0.01536),
}


def main():
    master = yaml.safe_load((ROOT / "configs/master/end_to_end.yaml").read_text())
    base_hyper = deepcopy(master["hyperparameters"]["baby_ssm_fit_pca"])
    results = []

    for name, (model_name, weight) in CANDIDATES.items():
        output_dir = OUTPUT_ROOT / name
        fit_spec = deepcopy(base_hyper["nicp"]["fits"][2])
        fit_spec["name"] = "map_1x"
        fit_spec["prior"]["kwargs"]["weight"] = weight
        fit_spec["solver"]["kwargs"]["iterations"] = 10
        fit_spec["solver"]["kwargs"]["learning_rate"] = 1.0

        raw = deepcopy(base_hyper)
        raw["inputs"] = {
            "shape_model_mat": str(MODEL_ROOT / model_name),
            "mesh_vtp": str(TARGET),
            "landmarks_json": str(LANDMARKS.parent),
            "template_vtp": str(TEMPLATE),
        }
        raw["outputs"] = {"dir": str(output_dir)}
        raw["nicp"]["fits"] = [fit_spec]

        cfg = ConfigClient().from_dict(raw, PipelineConfig)
        print(f"\n=== Fitting {name} ({model_name}, weight={weight}) ===")
        PipelineRunner(cfg).run()

        distance_cfg = recon_target_distance.Config(
            target_path=str(output_dir / "rigid_alignment.vtp"),
            recon_path=str(output_dir / "nicp_refined_vertices_map_1x.vtp"),
            landmarks_path=str(LANDMARKS),
            output_dir=str(output_dir / "eval_map"),
        )
        recon_target_distance.run(distance_cfg)
        summary_path = output_dir / "eval_map/distance_summary.json"
        summary = json.loads(summary_path.read_text())
        results.append({
            "model": name,
            "shape_model": model_name,
            "map_weight": weight,
            "recon_to_target_mean": summary["recon_dist_2_target"]["mean"],
            "target_to_recon_mean": summary["target_dist_2_recon"]["mean"],
            "summary": str(summary_path),
        })

    result_path = OUTPUT_ROOT / "sweep_summary.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved sweep summary to {result_path}")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
