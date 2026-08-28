"""Converts Slicer .fcsv markup files into the plain .csv format the pipeline's
align_to_reference / align_to_template stages read (landmarks_csv_dir expects
'x', 'y', 'z', 'label' columns; see scripts/US/pipeline/landmarks.py).

Slicer's .fcsv keeps the column names in a '# columns = ...' comment line
that only lists 14 names even though 5.8-era files write 16 fields per row
(2 undocumented trailing fields) - so columns are located by fixed position
rather than by splitting the comment line.
"""

import csv
import glob
import os

FCSV_COLUMNS = ["id", "x", "y", "z", "ow", "ox", "oy", "oz", "vis", "sel", "lock", "label", "desc", "associatedNodeID"]

INPUT_DIR = os.path.join(
    os.environ.get("PROJECT_ROOT", ""),
    "US_samples", "Casos_Anomalias", "Segmentations", "Frank-ter_Haar", "33_semanas", "Lmks_Ricardo",
)


def convert_fcsv_to_csv(fcsv_path, csv_path):
    rows = []
    with open(fcsv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(",")[: len(FCSV_COLUMNS)]
            rows.append(fields)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + FCSV_COLUMNS[1:])
        for i, fields in enumerate(rows):
            writer.writerow([i] + fields[1:])

    print(f"Wrote {len(rows)} landmarks: {os.path.basename(csv_path)}")


if __name__ == "__main__":
    for fcsv_path in glob.glob(os.path.join(INPUT_DIR, "*.fcsv")):
        csv_path = os.path.splitext(fcsv_path)[0] + ".csv"
        convert_fcsv_to_csv(fcsv_path, csv_path)
