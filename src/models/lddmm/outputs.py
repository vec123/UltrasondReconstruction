"""The files a resnet_flow apply run writes: the output contract of src/models/lddmm.

Code outside the model (src/evaluation) reads an apply folder through these
names, so this module imports nothing but the standard library.
"""

from dataclasses import dataclass


@dataclass
class ApplyOutputFiles:
    manifest: str = "apply.json"
    # The files below are written once per coordinate space, into a folder
    # named after it ("world" or "normalized", ApplyCfg.export.spaces).
    deformed: str = "deformed.vtp"
    template_points: str = "template_points.vtp"
    sample_points: str = "sample_points.vtp"
