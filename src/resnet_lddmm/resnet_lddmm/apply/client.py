"""DeformationClient: a trained flow, ready to deform shapes.

The inference counterpart of a stepper. A stepper owns an optimizer, a loss and a
subsample draw because it is TRAINING; none of that exists here. What is left is
the mapping itself -- points in, trajectory out -- so this class is small, has no
opinion about files or configs, and can be driven from a notebook as easily as
from the runner.

Everything config-shaped (which checkpoint, which template, where to write) is the
runner's business. Everything device- and frame-shaped is this class's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from src.resnet_lddmm.config import ExperimentCfg
from src.resnet_lddmm.io import FrameTransform, Shape
from src.resnet_lddmm.losses.mapping_error import apply_pose
from src.resnet_lddmm.runner import build_flow, set_default_dtype
from src.resnet_lddmm.trajectory import Trajectory


@dataclass(frozen=True)
class Deformation:
    """The result of deforming one shape: the whole path, not just the answer.

    Points are in the NORMALISED frame -- the one the model works in. ``transform``
    is carried along so the caller can go to world coordinates without having to
    remember which run produced this.

    Attributes:
        trajectory: [K+1, 1, N, 3] points and [K, 1, N, 3] velocities
        faces: the source shape's connectivity, or None for a point cloud
        transform: the run's FrameTransform, for invert() into world coordinates
    """

    trajectory: Trajectory
    faces: Optional[Tensor]
    transform: FrameTransform

    @property
    def points(self) -> Tensor:
        """[N, 3] terminal points, phi(T), in the normalised frame."""
        return self.trajectory.end[0]

    @property
    def num_points(self) -> int:
        return self.trajectory.points.shape[2]

    @property
    def num_steps(self) -> int:
        """K, the number of integration steps (the trajectory holds K+1 snapshots)."""
        return self.trajectory.points.shape[0] - 1

    @property
    def displacement(self) -> Tensor:
        """[N, 3] net motion of each point, phi(T) - T, in the normalised frame."""
        return self.trajectory.points[-1, 0] - self.trajectory.points[0, 0]


class DeformationClient:
    """A trained flow, frozen, applied to whole shapes.

    Deliberately NOT a subclass of the steppers: it shares no training state with
    them, and the only thing it needs from that side -- how the architecture is
    assembled -- comes from runner.build_flow, which both call.

    The flow runs under eval() and no_grad(): inference never wants dropout-style
    train behaviour, and the autograd graph over a full-resolution template is
    pure memory cost.
    """

    def __init__(self, flow, transform: FrameTransform, *,
                 code: Optional[Tensor] = None,
                 pose: Tuple[Optional[Tensor], Optional[Tensor]] = (None, None),
                 pose_application: str = "post",
                 device: str = "cpu",
                 chunk_size: int = 0):
        """Wire a ready-made flow into an inference client.

        Args:
            flow: NeuralODEFlow (or IdentityFlowWrapper) with weights already loaded
            transform: the run's FrameTransform, from runner.load_pair
            code: [1, n_z] conditioning code, or None for an unconditioned field
            pose: (rotation [1,3,3], translation [1,3]), either may be None
            pose_application: "post" (pred = phi(T) @ R) or "pre" (pred = phi(T @ R)),
                matching the training config -- see UnidirectionalMappingError
            device: torch device for the flow and the points
            chunk_size: integrate this many points at a time, 0 for all at once

        Raises:
            ValueError: on an unknown pose_application, or chunking a field whose
                velocity is not pointwise
        """
        if pose_application not in ("post", "pre"):
            raise ValueError(
                f"pose_application must be 'post' or 'pre', got {pose_application!r}"
            )

        self.device = torch.device(device)
        self.flow = flow.to(self.device).eval()
        self.transform = transform
        self.pose_application = pose_application
        self.chunk_size = int(chunk_size)

        rotation, translation = pose
        self.rotation = None if rotation is None else rotation.to(self.device)
        self.translation = None if translation is None else translation.to(self.device)
        self.code = None if code is None else code.to(self.device)

        if self.chunk_size and self._field_is_contextual():
            raise ValueError(
                f"chunk_size={self.chunk_size} is not valid for field "
                f"{type(self._field()).__name__}: its velocity at a point depends on "
                f"the rest of the cloud, so splitting the cloud changes the answer. "
                f"Set chunk_size: 0."
            )

    # -- construction ------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, train_cfg: ExperimentCfg, checkpoint: dict,
                        transform: FrameTransform, **kwargs) -> "DeformationClient":
        """Rebuild the training run's flow and load a checkpoint's weights into it.

        The architecture comes from the config and the weights from the file, and
        ``load_state_dict`` is strict, so a config that no longer describes the
        checkpoint fails here with the mismatched parameter names rather than
        producing a plausible-looking wrong deformation.

        Args:
            train_cfg: the ExperimentCfg the run was trained from
            checkpoint: dict from checkpoint.load_checkpoint
            transform: the run's FrameTransform, from runner.load_pair
            **kwargs: forwarded to __init__ (code, pose, device, chunk_size, ...)

        Returns:
            DeformationClient
        """
        # Before build_flow: parameters take the default dtype at __init__, and a
        # float32 model cannot hold a float64 checkpoint's tensors.
        set_default_dtype(train_cfg)

        flow = build_flow(train_cfg)
        flow.load_state_dict(checkpoint["flow"])

        kwargs.setdefault("pose_application", getattr(train_cfg, "pose_application", "post"))
        return cls(flow, transform, **kwargs)

    # -- application -------------------------------------------------------

    @torch.no_grad()
    def deform_points(self, points: Tensor) -> Trajectory:
        """Integrate the flow from a cloud in the NORMALISED frame.

        Args:
            points: [N, 3] or [1, N, 3] normalised points

        Returns:
            Trajectory with points [K+1, 1, N, 3], on the CPU

        Raises:
            ValueError: if points is not a single cloud of 3-vectors
        """
        points = self._as_batch(points).to(self.device)

        # "pre": the pose moves the template BEFORE the flow, so the trajectory
        # itself is a function of it. This has to match how the run was TRAINED --
        # the same weights under the other convention are a different mapping.
        if self.pose_application == "pre":
            points = apply_pose(points, self.rotation, self.translation)

        trajectory = self._integrate(points)

        # "post": the flow ran in the canonical frame, and the pose carries its
        # output into the sample's. Applied to every step rather than only to the
        # endpoint: the endpoint is then exactly the training prediction
        # phi(T) @ R, and the intermediate steps are the same path rigidly moved
        # alongside it, instead of a path that jumps frames at its last step.
        if self.pose_application == "post":
            trajectory = _map_points(trajectory,
                                     lambda p: apply_pose(p, self.rotation, self.translation))

        return _to_cpu(trajectory)

    def deform(self, shape: Shape) -> Deformation:
        """Deform a shape given in WORLD coordinates.

        The shape is carried into the run's normalised frame first, so this is the
        method to use for anything loaded off disk -- including a template at a
        different resolution than the one the run trained on, which lands in the
        same frame because the frame is a property of the RUN, not of the cloud.

        Args:
            shape: Shape with points [1, N, 3] in world coordinates

        Returns:
            Deformation, in the normalised frame, carrying the transform back
        """
        return Deformation(
            trajectory=self.deform_points(self.transform.apply(shape.points)),
            faces=shape.faces,
            transform=self.transform,
        )

    # -- internals ---------------------------------------------------------

    def _integrate(self, points: Tensor) -> Trajectory:
        """Run the flow, in one pass or in chunks of self.chunk_size points."""
        if not self.chunk_size or points.shape[1] <= self.chunk_size:
            return self.flow(points, self.code)

        chunks = [self.flow(part, self.code)
                  for part in torch.split(points, self.chunk_size, dim=1)]
        # Concatenate along the POINT axis. Exact, because the guard in __init__
        # has already established that each point's velocity is independent of
        # which chunk its neighbours ended up in.
        return Trajectory(
            points=torch.cat([c.points for c in chunks], dim=2),
            velocities=torch.cat([c.velocities for c in chunks], dim=2),
            dt=chunks[0].dt,
        )

    def _field(self):
        return self.flow.field

    def _field_is_contextual(self) -> bool:
        """True when the field reads the whole cloud, not just the point at hand.

        ``set_template`` is the marker the steppers already use to recognise these
        fields (see PairRegistration._values), so recognising them the same way
        here keeps one definition of "contextual".
        """
        return hasattr(self._field(), "set_template")

    @staticmethod
    def _as_batch(points: Tensor) -> Tensor:
        """[N,3] or [1,N,3] -> [1,N,3]; anything else is an error, not a guess."""
        if points.dim() == 2 and points.shape[-1] == 3:
            return points.unsqueeze(0)
        if points.dim() == 3 and points.shape[0] == 1 and points.shape[-1] == 3:
            return points
        raise ValueError(
            f"expected one cloud of shape [N, 3] or [1, N, 3], got {tuple(points.shape)}"
        )


def _map_points(trajectory: Trajectory, fn) -> Trajectory:
    """Apply fn to every step's points, leaving velocities and dt alone.

    Velocities are deliberately untouched: they are the field's output at each
    step, and a rigid move of the path does not change what the field produced.
    """
    return Trajectory(
        points=torch.stack([fn(step) for step in trajectory.points]),
        velocities=trajectory.velocities,
        dt=trajectory.dt,
    )


def _to_cpu(trajectory: Trajectory) -> Trajectory:
    """Bring a trajectory back to the CPU, where every exporter expects it."""
    return Trajectory(
        points=trajectory.points.cpu(),
        velocities=trajectory.velocities.cpu(),
        dt=trajectory.dt,
    )
