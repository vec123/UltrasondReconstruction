import csv
import json
from pathlib import Path

import numpy as np


def load_landmarks(path):
    """Load a named landmark set from either a single JSON file or a folder
    of Slicer fiducial CSVs (one file per annotation pass, same label set),
    which are averaged per-label on the fly."""
    path = Path(path)

    if path.is_dir():
        return _load_landmarks_from_csv_dir(path)
    return _load_landmarks_from_json(path)


def _load_landmarks_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    landmarks = {}
    names = []
    positions = []

    # Case 1: dict format
    if isinstance(data, dict):
        for k, v in data.items():
            landmarks[k] = np.asarray(v, dtype=np.float32)
            names.append(k)
            positions.append(v)

    # Case 2: list format
    elif isinstance(data, list):
        for item in data:
            name = item.get("name")
            pos = item.get("position")

            landmarks[name] = np.asarray(pos, dtype=np.float32)
            names.append(name)
            positions.append(pos)

    else:
        raise ValueError("Unsupported landmark JSON format")

    positions = np.asarray(positions, dtype=np.float32)

    return landmarks, positions, names


def _load_landmarks_from_csv_dir(csv_dir):
    csv_paths = sorted(csv_dir.glob("*.csv"))
    if not csv_paths:
        raise ValueError(f"No .csv annotation files found in {csv_dir}")

    sums = {}
    counts = {}
    names = []

    for csv_path in csv_paths:
        with open(csv_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                name = row["label"]
                pos = np.array(
                    [float(row["x"]), float(row["y"]), float(row["z"])],
                    dtype=np.float64,
                )
                if name not in sums:
                    sums[name] = np.zeros(3, dtype=np.float64)
                    counts[name] = 0
                    names.append(name)
                sums[name] += pos
                counts[name] += 1

    landmarks = {name: (sums[name] / counts[name]).astype(np.float32) for name in names}
    positions = np.asarray([landmarks[name] for name in names], dtype=np.float32)

    return landmarks, positions, names
