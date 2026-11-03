"""Loss terms for shape registration: data fitting and flow regularization."""

from src.resnet_lddmm.learning.losses.losses import chamfer_loss  # noqa: F401  (unused in the old module; kept so the import still happens)

from .base import DataTerm, FlowTerm, RegistrationLoss  # noqa: F401
from .basic import L2Data, NullData  # noqa: F401
from .chamfer import CDData, NCDData, PCDData, WeightedCDData  # noqa: F401
from .distance import LossDistanceHandle, ROBUST_KINDS  # noqa: F401
from .field_weighted import FieldWeightedCDData, FieldWeightedP2PlaneData  # noqa: F401
from .isometry import IsometryLoss  # noqa: F401
from .transport import EMDData, SinkhornData  # noqa: F401
from .two_sided import TwoSidedDataTerm  # noqa: F401
