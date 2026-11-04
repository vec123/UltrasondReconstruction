"""The encoder (code.kind: encoder): its input graph and the pose it predicts.

    graph.py          EncoderGraphLogger: the graphs the encoder reads, as .vtp
    pose.py           PoseLogger: predicted pose vs the augmentation's (pose_logs/)
    pose_shapes.py    PoseShapeExporter: the pose as geometry
    pose_metrics.py   EncoderPoseLogger and the rotation / translation error metrics
                      it uses (not wired into the runner)
"""

from src.models.lddmm.resnet_flow.callbacks.encoder.graph import EncoderGraphLogger
from src.models.lddmm.resnet_flow.callbacks.encoder.pose import PoseLogger
from src.models.lddmm.resnet_flow.callbacks.encoder.pose_metrics import EncoderPoseLogger
from src.models.lddmm.resnet_flow.callbacks.encoder.pose_shapes import PoseShapeExporter

__all__ = ["EncoderGraphLogger", "PoseLogger", "PoseShapeExporter", "EncoderPoseLogger"]
