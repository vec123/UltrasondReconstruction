"""What the optimizer sees: gradient norms, which parameters train, and the networks' layout.

    gradients.py       GradientMonitor (in memory), GradientLogger (grad_logs/)
    requires_grad.py   RequiresGradMonitor (requires_grad_logs/)
    structure.py       NetworkStructureInspector (grad_logs/network_structure.txt)
"""

from src.resnet_lddmm.resnet_flow.callbacks.gradient_logs.gradients import GradientLogger, GradientMonitor
from src.resnet_lddmm.resnet_flow.callbacks.gradient_logs.requires_grad import RequiresGradMonitor
from src.resnet_lddmm.resnet_flow.callbacks.gradient_logs.structure import NetworkStructureInspector

__all__ = ["GradientLogger", "GradientMonitor", "RequiresGradMonitor", "NetworkStructureInspector"]
