"""Entry point for the deformation-mode editor: config -> DeformationModesApp (src/GUIs/deformation_modes)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from src.config.SSM.client import ConfigClient
from src.GUIs.deformation_modes import DeformationModesApp, DeformationModesConfig

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ssm_gui" / "deformation_modes.yaml"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", "-c", type=Path, default=DEFAULT_CONFIG,
        help=f"YAML config file (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args(argv)

    config = ConfigClient(project_root=PROJECT_ROOT).load(args.config, DeformationModesConfig)
    result = DeformationModesApp(config).run()

    print(f"\n----------Done: {result.n_axes} axes, {result.n_constraint_sets} constraint sets, "
          f"{result.n_modes} modes"
          + (f", saved to {result.path}" if result.saved else " (nothing changed, not saved)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
