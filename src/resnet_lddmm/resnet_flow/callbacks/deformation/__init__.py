"""The deformation itself: its path, how well-behaved it is, and (modes/) what designed modes did.

    trajectories.py   TrajectoryExporter: snapshots of the flow as .vtp (trajectories/)
    diagnostics.py    DiagnosticsCallback: Jacobians, triangle flips, Lipschitz bound,
                      inverse residual (diag/* metrics)
    modes/            designed deformation modes: amplitudes during training, and a
                      finished run split into network and modes
"""

from src.resnet_lddmm.resnet_flow.callbacks.deformation.diagnostics import DiagnosticsCallback
from src.resnet_lddmm.resnet_flow.callbacks.deformation.trajectories import TrajectoryExporter

__all__ = ["DiagnosticsCallback", "TrajectoryExporter"]
