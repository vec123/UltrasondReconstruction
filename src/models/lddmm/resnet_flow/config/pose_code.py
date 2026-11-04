"""Pose, grid regularization and code configs, and the code config parser."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field as dataclass_field
from typing import Optional, Tuple
from src.shared.core.config import reject_unknown_keys


@dataclass
class PoseCfg:
    """A rigid pose learned alongside (or instead of) the flow — see poses/.

    Orthogonal to CodeCfg.kind: any code source can carry any pose. Rigid-only
    registration is ``code: {kind: none, pose: {kind: se3}}`` with
    ``freeze_flow_at_identity: true``; pose plus deformation is the same pose block
    on ``kind: auto_decoder``.

    Bounds confine the fit to the basin around the identity, which is what makes a
    rigid chamfer fit well posed when the two shapes are known to start roughly
    aligned -- chamfer on its own does not reliably prefer the correct alignment
    over a large flip. Both are enforced by a smooth radial squash that is the
    identity near zero and never produces a dead gradient (poses.free.bound_norm).
    """

    kind: str = "none"                      # none | so3 | se3
    max_angle_deg: Optional[float] = None   # largest rotation in degrees; None = unbounded
    # Largest translation, in NORMALISED units -- joint_normalize fits the pair into
    # a domain of width 1, so 0.1 is a tenth of its bounding box. "se3" only.
    max_translation: Optional[float] = None

    KINDS = ("none", "so3", "se3")

    @classmethod
    def from_dict(cls, d: dict) -> "PoseCfg":
        reject_unknown_keys(d, {"kind", "max_angle_deg", "max_translation"},
                            what="code.pose")
        kind = d.get("kind", "none")
        if kind not in cls.KINDS:
            raise ValueError(
                f"unknown pose kind {kind!r}; expected one of {list(cls.KINDS)}"
            )
        cfg = cls(kind=kind,
                  max_angle_deg=d.get("max_angle_deg"),
                  max_translation=d.get("max_translation"))
        # A translation bound under a rotation-only pose would be accepted and then
        # ignored, which reads as "bounded" in the config and is not.
        if cfg.max_translation is not None and kind != "se3":
            raise ValueError(
                f"max_translation is meaningless for pose kind {kind!r}, which has "
                f"no translation. Use kind: se3, or drop max_translation."
            )
        return cfg


@dataclass
class GridRegCfg:
    """Per-node weights of the code L2 penalty on a direct grid (code.grid_direct).

    Each node's code entries are penalised with a weight between ``low`` and
    ``high``, so some regions of the grid are held at the mean (zero code) harder
    than others. Only acts when loss.code_reg_weight > 0, and only on a direct grid,
    where each code entry belongs to one node -- see conditioning/grid_regularization.

    coverage: per member and node, from that member's own target. A node the
        target's points reach with at least ``full_support`` trilinear weight gets
        ``low``; a node they do not reach gets ``high``; linear in between. A
        partial target's empty regions are thereby pulled to the mean face.
    region: the same for every member. Nodes at the corners of the grid cells that
        hold the template landmarks ``labels`` get ``low``, all others ``high``.
    distance: the same for every member, from each node's distance to the template
        surface, in grid cells: ``low`` up to ``near``, ``high`` from ``far`` on,
        linear in between. Nodes near the face stay free and the ones towards the
        boundary, which hardly touch the face, are held at the mean.
    face_distance: the same for every member, from each node's distance to the
        nearest template landmark of ``labels``, in grid cells: ``low`` up to
        ``near``, ``high`` from ``far`` on, linear in between. With the central
        face landmarks as labels, the cheeks, temples and the outer region get
        ``high`` and the face centre ``low``.
    """

    kind: str = "coverage"                  # coverage | region | distance | face_distance
    low: float = 0.1                        # weight of a covered / in-region node
    high: float = 1.0                       # weight of an uncovered / out-of-region node
    full_support: float = 20.0              # coverage: point weight at which a node is fully covered
    labels: Tuple[str, ...] = ()            # region, face_distance: template landmark labels (heatmap_<label> fields)
    near: float = 0.5                       # distance, face_distance: nodes up to this many cells away get low
    far: float = 1.0                        # distance, face_distance: nodes from this many cells on get high

    KINDS = ("coverage", "region", "distance", "face_distance")

    @classmethod
    def from_dict(cls, d: dict) -> "GridRegCfg":
        reject_unknown_keys(d, {"kind", "low", "high", "full_support", "labels", "near", "far"},
                            what="code.grid_reg")
        cfg = cls(kind=d.get("kind", "coverage"), low=float(d.get("low", 0.1)),
                  high=float(d.get("high", 1.0)), full_support=float(d.get("full_support", 20.0)),
                  labels=tuple(d.get("labels", ())), near=float(d.get("near", 0.5)),
                  far=float(d.get("far", 1.0)))
        if cfg.kind not in cls.KINDS:
            raise ValueError(f"unknown code.grid_reg kind {cfg.kind!r}; expected one of {list(cls.KINDS)}")
        if cfg.kind in ("region", "face_distance") and not cfg.labels:
            raise ValueError(f"code.grid_reg kind {cfg.kind!r} needs landmark labels")
        if cfg.kind in ("distance", "face_distance") and not cfg.far > cfg.near >= 0:
            raise ValueError(f"code.grid_reg {cfg.kind} needs far > near >= 0, got near {cfg.near}, far {cfg.far}")
        if cfg.full_support <= 0:
            raise ValueError(f"code.grid_reg full_support must be > 0, got {cfg.full_support}")
        return cfg


@dataclass
class CodeCfg:
    kind: str = "none"                      # none | auto_decoder | encoder
    n_z: int = 256
    position_aware: bool = False            # use grid interpolation (True) or broadcast (False)?
    conditioning_method: str = "concat"     # concat | film — how to apply features to velocity
    grid: int = 2                           # g — grid resolution (only if position_aware=True)
    grid_channels: int = 32                 # C — grid-head output per corner (only if position_aware=True)
    grid_direct: bool = False               # the code IS the grid (n_z = g³·C), not decoded into it
    grid_reg: GridRegCfg = dataclass_field(default_factory=GridRegCfg)  # per-node penalty weights (direct grid)
    init_std: float = 0.01                  # auto_decoder: std of the codes at init

    # Encoder-specific config (only used if kind == "encoder")
    encoder_config: Optional[EncoderConfig] = dataclass_field(default=None)
    graph_spec: Optional[GraphSpec] = dataclass_field(default=None)

    # The rigid pose this code source carries, if any. Orthogonal to `kind`.
    pose: PoseCfg = dataclass_field(default_factory=PoseCfg)


def _parse_code_config(code_dict: dict) -> CodeCfg:
    """Parse code config, handling the nested pose and grid_reg configs.

    kind "encoder" is not yet supported: it is accepted, with a warning, and its
    encoder_config and graph_spec are dropped. Building that code source then fails.
    """
    # Extract nested configs before passing to CodeCfg
    code_dict.pop("encoder_config", None)
    code_dict.pop("graph_spec", None)
    pose_cfg = PoseCfg.from_dict(code_dict.pop("pose", {}))
    grid_reg_cfg = GridRegCfg.from_dict(code_dict.pop("grid_reg", {}))

    if code_dict.get("kind") == "encoder":
        warnings.warn(
            "code.kind 'encoder' is not yet supported: encoder_config and graph_spec are ignored, "
            "and building this code source will fail",
            stacklevel=2,
        )

    return CodeCfg(pose=pose_cfg, grid_reg=grid_reg_cfg, **code_dict)
