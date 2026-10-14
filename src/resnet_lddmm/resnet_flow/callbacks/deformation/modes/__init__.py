"""Designed deformation modes (fields/modes.py): during training, and in a finished run.

    amplitudes.py      ModeAmplitudeLogger: each mode's amplitude per step (modes/<name>_mm)
    parts.py           the maths: the actual flow split exactly by part (decompose), and
                       flows with only some parts switched on (flow_with) - every step kept
    export.py          a trained flow -> .vtp files in <run>/mode_decomposition/ (end shapes
                       and full trajectories) and a summary: write_decomposition for a flow
                       in memory, export_decomposition for a finished run on disk
                       (scripts/LDDMM/decompose_modes.py is its CLI)
    decomposition.py   DecompositionExporter: write_decomposition at the end of training
                       (field.modes.decompose)
"""

from src.resnet_lddmm.resnet_flow.callbacks.deformation.modes.amplitudes import ModeAmplitudeLogger
from src.resnet_lddmm.resnet_flow.callbacks.deformation.modes.decomposition import DecompositionExporter
from src.resnet_lddmm.resnet_flow.callbacks.deformation.modes.export import (
    OUT_DIR, TRAJECTORY_DIR, check_frame, export_decomposition, load_run, summary_text,
    write_decomposition, write_trajectory,
)
from src.resnet_lddmm.resnet_flow.callbacks.deformation.modes.parts import (
    NETWORK, PartsTrajectory, decompose, flow_with, modal_field,
)

__all__ = ["ModeAmplitudeLogger", "DecompositionExporter",
           "NETWORK", "PartsTrajectory", "decompose", "flow_with", "modal_field",
           "OUT_DIR", "TRAJECTORY_DIR", "check_frame", "export_decomposition", "load_run",
           "summary_text", "write_decomposition", "write_trajectory"]
