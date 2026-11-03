"""Project-root resolution shared by every pipeline's config loading.

The root is read from the ``PROJECT_ROOT`` environment variable (set in the
repo's ``.env``, loaded here via python-dotenv) rather than derived from
``__file__``, so it is correct no matter which pipeline's script, notebook,
or test imports this module.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def get_project_root() -> Path:
    load_dotenv()
    root = os.environ.get("PROJECT_ROOT")
    if not root:
        raise RuntimeError("PROJECT_ROOT is not set. Define it in the repo's .env file.")
    return Path(root)
