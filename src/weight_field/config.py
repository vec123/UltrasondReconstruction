from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml_mapping(path) -> dict:
    """Read a YAML file that must contain a mapping at the top level.

    The one reader every config loader in the package goes through, so a missing
    file and a YAML list fail the same way whichever config is being read.

    Args:
        path: path to a .yaml file

    Returns:
        The parsed dict

    Raises:
        FileNotFoundError: if the file does not exist
        ValueError: if the document is not a mapping
        yaml.YAMLError: if parsing fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML dict, got {type(data).__name__}")

    return data
