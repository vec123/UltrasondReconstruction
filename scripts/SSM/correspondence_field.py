"""Correspondence field of one case: matches the recon to the target and writes it onto copies of both.

The command line of the stage correspondence_field
(src/evaluation/stages/correspondence_field.py), which does the work and
describes the outputs. From a YAML: python scripts/run.py
configs/evaluation/fields/correspondence_field.yaml

    python scripts/SSM/correspondence_field.py <case_dir> \
        --method constrained --kwargs '{"max_dist": 2, "normal_thresh_deg": 60, "one_to_one": true}' \
        --distance-frame surface
"""

import argparse
import json

from src.evaluation.config import CorrespondenceFieldConfig
from src.evaluation.loader import FRAMES
from src.evaluation.stages.correspondence_field import run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_dir")
    parser.add_argument("--layout", default="auto")
    parser.add_argument("--layout-kwargs", default="{}", help="JSON, e.g. '{\"fit\": \"gamma_1000\"}' for direct_ssm_fit")
    parser.add_argument("--method", default="constrained")
    parser.add_argument("--kwargs", default='{"max_dist": 2.0, "normal_thresh_deg": 60.0, "one_to_one": true}',
                        help="JSON settings of the method")
    parser.add_argument("--distance-frame", default="surface", choices=FRAMES,
                        help="frame of the distance settings: 'surface' is the scan in mm")
    args = parser.parse_args()

    run(CorrespondenceFieldConfig(case_dir=args.case_dir, layout=args.layout,
                                  layout_kwargs=json.loads(args.layout_kwargs), method=args.method,
                                  kwargs=json.loads(args.kwargs), distance_frame=args.distance_frame))


if __name__ == "__main__":
    main()
