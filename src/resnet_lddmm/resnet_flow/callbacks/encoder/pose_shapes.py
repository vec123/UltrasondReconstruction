"""PoseShapeExporter: the encoder's pose as geometry - the template before and after it,
and the sample it should land on."""

import os
import numpy as np

from src.resnet_lddmm.learning.callbacks.base import Callback
from src.resnet_lddmm.resnet_flow.flow import IdentityFlowWrapper
from src.resnet_lddmm.resnet_flow.io import export_shape, FrameTransform


class PoseShapeExporter(Callback):
    """Export the shape before and after the predicted pose, as .vtp.

    PoseLogger records the pose as numbers; this writes the geometry, so the
    alignment can be judged by eye in ParaView rather than inferred from a matrix.

    Per step, per exported shape:

        before.vtp    phi(T)          the deformed template, in the CANONICAL frame,
                                      i.e. before the predicted pose is applied
        after.vtp     phi(T) @ R_hat  the same points with the pose applied -- exactly
                                      what the data term compares against the sample
        target.vtp    the augmented sample, i.e. what `after` should converge onto

    Load all three together and colour by ``point_id`` to follow individual vertices:
    `after` and `target` should coincide when the pose is right, while `before` shows
    how much of the discrepancy was pose rather than deformation.

    Those three are the SUBSAMPLED clouds the loss actually saw -- a fresh random
    ``loss.subsample_M`` points every step, so they carry no connectivity (the
    template's face indices address vertices the draw dropped). With
    ``export_full`` the same three are written again at FULL resolution WITH faces,
    as ``*_full.vtp``, which is what you want to look at rather than a point cloud
    that changes membership every export. See _full_endpoint for when that is
    available without recomputing anything.

    Coordinates are the NORMALIZED frame, not world -- the pose is defined there, and
    mapping back through FrameTransform would fold in a rotation the pose head never
    saw. That is deliberate and differs from TrajectoryExporter, which does export to
    world coordinates.

    Reads what mapping_error recorded for the step it just finished, so it re-does no
    computation and cannot perturb training: `last_fwd_traj` for phi(T) and
    `last_effective_pose` for the pose that was actually applied (post the
    requires_grad filter, so this shows the real transform, not the predicted one).
    """

    def __init__(self, every_n_steps=100, export_shapes=2, subdir="pose_shapes",
                 export_full=True):
        """Initialize the exporter.

        Args:
            every_n_steps: cadence
            export_shapes: how many shapes of the batch to write (0 = all)
            subdir: directory under ctx.log_dir to write into
            export_full: also write the full-resolution input mesh, with faces,
                before and after the pose. Skipped silently when it would require
                recomputing the flow -- see _full_endpoint.
        """
        super().__init__(every_n_steps)
        self.export_shapes = export_shapes
        self.subdir = subdir
        self.export_full = export_full
        self.transform = None  # Set by runner after build(), as for TrajectoryExporter
        self._warned_no_full = False

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Write before/after/target for this step, if due."""
        if not self._due(step):
            return

        stepper = ctx.stepper
        mapping_error = getattr(stepper, "mapping_error", None)
        if mapping_error is None or getattr(mapping_error, "last_fwd_traj", None) is None:
            return

        rotation, translation = getattr(mapping_error, "last_effective_pose", (None, None))
        if rotation is None and translation is None:
            return  # no pose was applied; before and after would be identical

        from src.resnet_lddmm.resnet_flow.losses.terms import apply_pose
        from src.vtk.create import create_polydata
        from src.vtk.fields import add_point_field
        from src.vtk.io import save_vtp

        before = mapping_error.last_fwd_traj.end.detach()
        after = apply_pose(before, rotation, translation).detach()
        target = self._target_points(stepper, mapping_error)

        count = before.shape[0] if self.export_shapes == 0 else min(self.export_shapes,
                                                                   before.shape[0])
        identity = FrameTransform.identity()
        world = self.transform if self.transform is not None else identity

        # Both spaces: `normalized` is where the pose is defined and is the one to
        # judge alignment in; `world` is comparable with the input meshes.
        for space, transform in (("normalized", identity), ("world", world)):
            out_dir = os.path.join(ctx.log_dir, self.subdir, f"step_{step}", space)
            os.makedirs(out_dir, exist_ok=True)
            for b in range(count):
                clouds = {"before": before[b], "after": after[b]}
                if target is not None and b < target.shape[0]:
                    clouds["target"] = target[b]
                for name, points in clouds.items():
                    polydata = create_polydata(transform.invert(points), faces=None)
                    ids = np.arange(points.shape[0], dtype=np.float32)
                    polydata = add_point_field(polydata, ids, field_name="point_id")
                    save_vtp(polydata, os.path.join(out_dir, f"shape_{b}_{name}.vtp"),
                             binary=True)

            if self.export_full:
                self._export_full_meshes(stepper, mapping_error, batch, out_dir,
                                         transform, rotation, translation)

        print(f"[PoseShapeExporter] step {step}: wrote before/after/target for "
              f"{count} shape(s) in normalized/ and world/")

    def _export_full_meshes(self, stepper, mapping_error, batch, out_dir, transform,
                            rotation, translation):
        """Write the full-resolution template and sample, with connectivity.

        The subsampled clouds above are a different random subset every export, so
        they cannot be compared across steps and cannot carry faces. These can be
        both: the vertex set is the whole input mesh, so its face indices are valid
        and the same vertex is the same vertex at every step.

        Only shape 0 -- pair mode has one template, and a cohort's full meshes would
        multiply the written volume by the batch for no extra information about the
        pose, which is shared geometry-wide.

        Args:
            out_dir: this space's directory
            transform: the space's FrameTransform, applied via invert()
            rotation, translation: the pose that was actually applied
        """
        from src.resnet_lddmm.resnet_flow.losses.terms import apply_pose

        template, sample = batch
        before = self._full_endpoint(stepper, mapping_error, template)
        if before is None:
            return

        after = apply_pose(before, rotation, translation).detach()
        meshes = {
            "before_full": (before[0], getattr(template, "faces", None)),
            "after_full": (after[0], getattr(template, "faces", None)),
            "target_full": (sample.points[0], getattr(sample, "faces", None)),
        }
        for name, (points, faces) in meshes.items():
            # export_shape drops connectivity that does not address these exact
            # vertices, so a template whose faces went missing degrades to a cloud
            # rather than writing a mesh out of the wrong indices.
            export_shape(points.detach(), faces, transform,
                         os.path.join(out_dir, f"shape_0_{name}.vtp"), label=name)

    def _full_endpoint(self, stepper, mapping_error, template):
        """phi(T) at FULL resolution, or None when it cannot be had for free.

        Three cases, in order:

        1. ``loss.save_full: true`` already computed it -- use it.
        2. The flow is frozen at the identity, so phi(T_full) IS T_full. Exact and
           costs nothing, which is the rigid-registration case: the pose is then
           the entire model and the full posed mesh is the actual deliverable.
        3. A live flow without save_full. Producing it would mean an extra
           integration of the whole mesh from inside a callback -- real compute,
           silently added to every export. Skipped, with one message saying which
           flag turns it on.

        Also skipped under ``pose_application='pre'`` with a live flow: there the
        pose is applied BEFORE the flow, so the caller's ``apply_pose(phi(T), R)``
        would be applying it a second time. Under a frozen flow the two orders
        coincide, so that case is still exported.
        """
        traj_full = getattr(mapping_error, "last_fwd_traj_full", None)
        frozen = isinstance(getattr(stepper, "flow", None), IdentityFlowWrapper)

        if getattr(mapping_error, "pose_application", "post") == "pre" and not frozen:
            return None

        if traj_full is not None:
            return traj_full.end.detach()
        if frozen:
            return template.points.detach()

        if not self._warned_no_full:
            self._warned_no_full = True
            print("[PoseShapeExporter] full-resolution meshes need the full "
                  "trajectory; set loss.save_full: true (or freeze the flow) to "
                  "get *_full.vtp alongside the subsampled clouds.")
        return None

    @staticmethod
    def _target_points(stepper, mapping_error):
        """The augmented sample the prediction is compared against, or None.

        Prefers what mapping_error actually used (subsampled, so it lines up with
        `after` point-for-point); falls back to the stepper's stored augmented batch,
        whose attribute name differs between the pair and cohort steppers.
        """
        points = getattr(mapping_error, "last_sample_points", None)
        if points is not None:
            return points.detach()
        for attribute in ("augmented_sample_batch", "augmented_sample"):
            sample = getattr(stepper, attribute, None)
            if sample is not None and getattr(sample, "points", None) is not None:
                return sample.points.detach()
        return None
