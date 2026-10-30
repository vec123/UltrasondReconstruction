"""Loss configs: the Chamfer sides, fshape, point weight fields and the loss itself."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, List, Optional
from src.core.config import reject_unknown_keys


# The flat robust_* keys a data term used to take, each now a key of ONE
# loss_distance_handle block. Still recognised, but only to fail with the new
# syntax instead of a bare unknown-key or unexpected-argument error.
RENAMED_LOSS_DISTANCE_KEYS = {"robust_kernel": "kind", "robust_breakpoint": "breakpoint",
                              "robust_slope": "slope", "robust_tau": "tau"}


def reject_renamed_loss_distance_keys(d: dict, where: str) -> None:
    """Fail on the no-longer-supported robust_* keys, naming their replacement.

    Only called where a data term is BUILT (runner.build, ChamferSideCfg) --
    never on loading an ExperimentCfg, since every finished run's config.json
    still carries the old keys and apply must keep reading those.

    Args:
        d: a data_kwargs mapping, or one direction's block inside it.
        where: its place in the config, for the message.

    Raises:
        ValueError: any robust_* key is present.
    """
    old = [key for key in RENAMED_LOSS_DISTANCE_KEYS if key in d]
    if old:
        block = ", ".join(f"{RENAMED_LOSS_DISTANCE_KEYS[key]}: "
                          f"{'null' if d[key] is None else d[key]}" for key in old)
        raise ValueError(
            f"{where}: {', '.join(old)} {'is' if len(old) == 1 else 'are'} no longer "
            f"supported -- the robust_* keys were replaced by one loss_distance_handle "
            f"block. Write  loss_distance_handle: {{{block}}}  instead."
        )


@dataclass
class ChamferSideCfg:
    """One direction of a field-weighted chamfer term (data_name: field_weighted_chamfer).

    Generalises "one shared source_to_target_weight/target_to_source_weight plus
    one shared loss.sigma" into one independent (weight, base_sigma) PER
    DIRECTION, optionally modulated by a per-vertex precision field.

    weight and base_sigma are read by the data term itself (FieldWeightedCDData):
    data weight for this direction = weight / (2 * base_sigma**2), exactly the
    same "data weight = 1/(2*sigma**2)" convention loss.sigma already uses
    elsewhere in this file, just local to one direction instead of shared.

    field_name/field_source are deliberately NOT read by the data term -- turning
    a field name into an actual per-vertex tensor needs the run's normalised
    shapes, which do not exist yet when Registry.create builds the data term (see
    LossCfg.data_kwargs). They are resolved once, later, at build time (see
    field_weights.resolve_point_weight_field and runner._add_field_weighted_chamfer),
    into a tensor that reaches the data term the same way any other per-point
    weight already does: as pred_w/tgt_w, supplied by the mapping error on every
    call.

    Attributes:
        weight: how much this direction counts in the combined data loss (same
            role as source_to_target_weight/target_to_source_weight elsewhere).
        base_sigma: this direction's own noise scale, in the same normalised
            length units as loss.sigma.
        field_name: name of a point field to read. None (the default) means a
            CONSTANT field: every point in this direction is weighted equally,
            which is today's plain chamfer with a local sigma. Otherwise
            interpreted according to field_transform below.
        field_source: path to the .vtp the field lives on. None (the default)
            reads it from the run's own source/target shape -- whichever one
            this direction is indexed by (source_to_target reads the SOURCE's
            fields, target_to_source reads the TARGET's). If given, the file
            must describe the SAME vertices, in the same order, as that shape.
        field_transform: how to turn the named field into a per-vertex
            precision MULTIPLIER (dimensionless -- it scales base_sigma's
            precision up/down, it does not replace it):
              None (default) -- the field IS the multiplier already, used as-is.
              "gaussian_distance" -- the field is a DISTANCE (e.g. to the
                nearest landmark: small near it, large far away); converted via
                weight = exp(-0.5 * (distance / field_scale)**2), turning a
                "close to X" measurement into a "how much X matters here"
                weight, which a raw distance is not -- used directly it would
                emphasise points FAR from X, exactly backwards.
        field_scale: required when field_transform is set. Same units as the
            named field's own values (e.g. a landmark-distance field's
            world-frame length) -- no FrameTransform conversion needed, since
            it only ever appears as a RATIO against the field's own values.
        loss_distance_handle: this direction's OWN ``{kind, breakpoint, slope,
            tau, power}`` block, overriding the data term's shared loss_distance_handle
            for this direction alone -- used whole, never mixed with the shared
            block. None (the default) inherits the shared one; ``{kind:
            quadratic}`` forces plain squared distance here even when the shared
            block does not. Kept as the raw block: it is validated where the
            handle is built (data_terms.LossDistanceHandle.from_dict), since
            ROBUST_KINDS lives with the torch code.
        inherit_source_field: target_to_source ONLY. Each target point takes,
            as its precision multiplier, the source_to_target field's value at
            its NEAREST deformed source point -- recomputed every step from the
            current matches. The target then needs no field of its own: "this
            target point is explained by the template's nose" makes it a nose
            point, with no annotation on the target. Multiplies this side's own
            field when both are set. Needs a source_to_target field_name, and is
            refused under direction: bidirectional (see FieldWeightedCDData).
        inherit_target_field: source_to_target ONLY, the mirror of
            inherit_source_field. Each source point takes, as a precision
            multiplier, a target field's value at its NEAREST target point --
            recomputed every step from the current matches. ``true`` inherits the
            target_to_source field; a field NAME (e.g. edge_weight) inherits that
            target field instead, which then acts on this direction only (cohort
            mode). With a target field that is 0 on the target's rim
            (scripts/write_edge_fields.py: edge_weight), a source point the target
            never saw, whose nearest target point lies on the rim, stops pulling
            toward the rim. Multiplies this side's own field when both are set.
            Refused under direction: bidirectional.
    """
    weight: float = 1.0
    base_sigma: float = 0.1
    field_name: Optional[str] = None
    field_source: Optional[str] = None
    field_transform: Optional[str] = None
    field_scale: Optional[float] = None
    loss_distance_handle: Optional[dict] = None
    inherit_source_field: bool = False
    inherit_target_field: Any = False       # False | True | a target field name

    TRANSFORMS = ("gaussian_distance",)

    @classmethod
    def from_dict(cls, d: dict) -> "ChamferSideCfg":
        reject_renamed_loss_distance_keys(d, "source_to_target/target_to_source block")
        reject_unknown_keys(d, {"weight", "base_sigma", "field_name", "field_source",
                                "field_transform", "field_scale", "loss_distance_handle",
                                "inherit_source_field", "inherit_target_field"},
                            what="chamfer side")
        weight = float(d.get("weight", 1.0))
        base_sigma = float(d.get("base_sigma", 0.1))
        if weight < 0:
            raise ValueError(f"chamfer side weight must be >= 0, got {weight}")
        if base_sigma <= 0:
            raise ValueError(f"chamfer side base_sigma must be > 0, got {base_sigma}")

        field_transform = d.get("field_transform")
        field_scale = d.get("field_scale")
        if field_transform is not None:
            if field_transform not in cls.TRANSFORMS:
                raise ValueError(f"unknown field_transform {field_transform!r}; "
                                 f"expected one of {cls.TRANSFORMS} or null")
            if field_scale is None:
                raise ValueError(
                    f"field_transform={field_transform!r} needs a field_scale "
                    f"(same units as the named field's own values); pass one, "
                    f"or drop field_transform to use the field as-is."
                )
            field_scale = float(field_scale)
            if field_scale <= 0:
                raise ValueError(f"field_scale must be > 0, got {field_scale}")

        return cls(weight=weight, base_sigma=base_sigma,
                   field_name=d.get("field_name"), field_source=d.get("field_source"),
                   field_transform=field_transform, field_scale=field_scale,
                   loss_distance_handle=d.get("loss_distance_handle"),
                   inherit_source_field=bool(d.get("inherit_source_field", False)),
                   inherit_target_field=_parse_inherit_target_field(d.get("inherit_target_field", False)))


def _parse_inherit_target_field(value):
    """source_to_target.inherit_target_field: a bool, or a non-empty target field name."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"inherit_target_field must be true/false or a target field name "
                     f"(e.g. edge_weight), got {value!r}")


@dataclass
class FshapeCfg:
    """loss.fshape_kwargs: the windowed functional-varifold term (losses/fshape.py).

    Attributes:
        landmarks: template labels whose mean position centres the window.
        radius_mm: full-strength window radius (world units, mm here).
        falloff_mm: smooth falloff to zero beyond radius_mm.
        margin_mm: extra template crop beyond the window, so tissue moving
            INTO the window during the fit is represented.
        sigma_mm: position kernel widths (one or several scales, averaged).
        signal: "curvedness" (unsigned, orientation-free), "mean_curvature"
            (signed), or "none" for a plain varifold.
        signal_normalize: "rank" (default) compares percentiles within the
            window instead of raw curvature -- the scan is systematically
            curvier than the template (noise), which raw values would penalise
            everywhere. "none" compares raw 1/mm values.
        sigma_signal: signal kernel width (percentile units under rank, 1/mm
            under none); None = half the pooled standard deviation of both
            meshes' triangle signals.
        curvature_radius_mm: neighbourhood of the curvature fit, i.e. its
            smoothing scale.
        n_faces: triangles sampled per mesh (area-proportional, fixed).
        seed: for that sample.
        heatmap_prefix: template landmark fields' prefix.
    """
    landmarks: List[str] = dataclass_field(default_factory=lambda: ["aL", "aR", "acL", "acR", "prn", "sn"])
    radius_mm: float = 10.0
    falloff_mm: float = 8.0
    margin_mm: float = 4.0
    sigma_mm: List[float] = dataclass_field(default_factory=lambda: [4.0, 2.0])
    signal: str = "curvedness"
    signal_normalize: str = "rank"
    sigma_signal: Optional[float] = None
    curvature_radius_mm: float = 2.5
    n_faces: int = 1500
    seed: int = 0
    heatmap_prefix: str = "heatmap_"

    SIGNALS = ("curvedness", "mean_curvature", "none")

    @classmethod
    def from_dict(cls, d: dict) -> "FshapeCfg":
        reject_unknown_keys(d, {f.name for f in cls.__dataclass_fields__.values()},
                            what="loss.fshape_kwargs")
        cfg = cls(**d)
        if not cfg.landmarks or not all(isinstance(label, str) for label in cfg.landmarks):
            raise ValueError(f"loss.fshape_kwargs.landmarks must be a non-empty list of labels, got {cfg.landmarks!r}")
        if cfg.radius_mm < 0 or cfg.falloff_mm <= 0 or cfg.margin_mm < 0:
            raise ValueError("loss.fshape_kwargs needs radius_mm >= 0, falloff_mm > 0, margin_mm >= 0")
        sigmas = cfg.sigma_mm if isinstance(cfg.sigma_mm, (list, tuple)) else [cfg.sigma_mm]
        if not sigmas or any(float(s) <= 0 for s in sigmas):
            raise ValueError(f"loss.fshape_kwargs.sigma_mm must be positive, got {cfg.sigma_mm!r}")
        cfg.sigma_mm = [float(s) for s in sigmas]
        if cfg.signal not in cls.SIGNALS:
            raise ValueError(f"loss.fshape_kwargs.signal must be one of {cls.SIGNALS}, got {cfg.signal!r}")
        if cfg.signal_normalize not in ("rank", "none"):
            raise ValueError(f"loss.fshape_kwargs.signal_normalize must be 'rank' or 'none', "
                             f"got {cfg.signal_normalize!r}")
        if cfg.sigma_signal is not None and float(cfg.sigma_signal) <= 0:
            raise ValueError(f"loss.fshape_kwargs.sigma_signal must be > 0 or null, got {cfg.sigma_signal}")
        if cfg.curvature_radius_mm <= 0 or int(cfg.n_faces) < 100:
            raise ValueError("loss.fshape_kwargs needs curvature_radius_mm > 0 and n_faces >= 100")
        return cfg


@dataclass
class PointWeightFieldCfg:
    """A named per-vertex weight field on the SOURCE, for a term other than the data term.

    Reusable by any term that holds one weight per template vertex (today:
    loss.kinetic_field_kwargs, see losses/kinetic.py). Resolved once, at build
    time, by field_weights.resolve_point_weight_field -- the same resolver, and the
    same field_* keys with the same meaning, as ChamferSideCfg.

    Attributes:
        field_name: name of the point field to read. Required -- unlike
            ChamferSideCfg there is no "constant field" default, because a
            constant field is what the plain term already is.
        field_source: path to a .vtp carrying the field; None reads it from the
            run's own source shape. Must describe the same vertices, in the
            same order.
        field_transform: None uses the values as-is; "gaussian_distance" treats
            them as a distance, see ChamferSideCfg.
        field_scale: required with field_transform, rejected without it.
        normalize: rescale the resolved field to mean 1 over the full mesh
            (default). The term's weight then keeps the scale of the unweighted
            term it extends, and only the field's SHAPE -- where it is high,
            where low, by what ratio -- carries meaning. False uses the raw
            values as the multiplier.
    """
    field_name: str
    field_source: Optional[str] = None
    field_transform: Optional[str] = None
    field_scale: Optional[float] = None
    normalize: bool = True

    @classmethod
    def from_dict(cls, d: dict, what: str = "point weight field") -> "PointWeightFieldCfg":
        reject_unknown_keys(d, {"field_name", "field_source", "field_transform",
                                "field_scale", "normalize"}, what=what)
        field_name = d.get("field_name")
        if not field_name:
            raise ValueError(f"{what} needs a 'field_name': the point field to read "
                             f"from the source (or from field_source).")

        field_transform = d.get("field_transform")
        field_scale = d.get("field_scale")
        if field_transform is None:
            if field_scale is not None:
                raise ValueError(f"{what}: field_scale is only used with a "
                                 f"field_transform; set one, or drop field_scale.")
        else:
            if field_transform not in ChamferSideCfg.TRANSFORMS:
                raise ValueError(f"{what}: unknown field_transform {field_transform!r}; "
                                 f"expected one of {ChamferSideCfg.TRANSFORMS} or null")
            if field_scale is None:
                raise ValueError(f"{what}: field_transform={field_transform!r} needs a "
                                 f"field_scale (same units as the field's own values).")
            field_scale = float(field_scale)
            if field_scale <= 0:
                raise ValueError(f"{what}: field_scale must be > 0, got {field_scale}")

        return cls(field_name=field_name, field_source=d.get("field_source"),
                   field_transform=field_transform, field_scale=field_scale,
                   normalize=bool(d.get("normalize", True)))


@dataclass
class LossCfg:
    data_name: str = "chamfer"              # chamfer | weighted_chamfer | field_weighted_chamfer | field_weighted_point2plane_chamfer | pcd | ncd | l2 | sinkhorn | emd | none
    # Constructor kwargs for the data term, passed straight to Registry.create.
    # The chamfer family (chamfer/weighted_chamfer/pcd/ncd) takes per-direction
    # weights -- source_to_target_weight, target_to_source_weight, both 1.0 by
    # default, named for the `source:`/`target:` shapes above. Setting
    # source_to_target_weight to 0.0 makes the term ONE-SIDED, which is what a
    # PARTIAL target needs: "every source point lies on the target surface" is
    # false when most of the source was never scanned, and minimising it
    # collapses the source into the retained patches. See TwoSidedDataTerm.
    #
    # The same chamfer family also takes an optional loss_distance_handle block,
    # quadratic below its breakpoint and one of {linear, truncated, exponential}
    # above it -- e.g. loss_distance_handle: {kind: linear, breakpoint: 0.05,
    # slope: 0.0} caps a far correspondence's pull instead of letting it grow
    # without bound. breakpoint is in the same normalised length units as sigma
    # below. Its optional power (default 2) replaces the quadratic below the
    # breakpoint with d**power -- power: 1 is L1, pulling small residuals as hard
    # as large ones. See data_terms.LossDistanceHandle / _robust_penalty for
    # exactly what each kind does.
    #
    # data_name: field_weighted_chamfer and field_weighted_point2plane_chamfer take
    # a DIFFERENT shape: two ChamferSideCfg
    # dicts, {source_to_target: {...}, target_to_source: {...}}, each with its own
    # weight/base_sigma/field_name/field_source instead of one shared
    # source_to_target_weight/target_to_source_weight plus the one shared sigma
    # below (which this data term ignores -- see ChamferSideCfg and
    # FieldWeightedCDData for what each key does, and runner.build for how
    # field_name/field_source get resolved once the shapes are loaded). A side
    # may also carry its own loss_distance_handle block, overriding the shared
    # one for that direction only -- e.g. truncating source_to_target alone, so
    # the part of the source a partial target never saw stops pulling. Under
    # direction: bidirectional each side's settings stay with the same claim in
    # the backward half too (FieldWeightedCDData.mirrored), and point-to-plane
    # projects that half onto the SOURCE's normals, derived from its faces when
    # the file has none.
    #
    # data_name: none takes NO kwargs -- it is the explicit "no chamfer term at
    # all" opt-out, always zero/no-gradient, for runs where loss.landmark_weight
    # (or another non-chamfer term) is meant to be the only positional
    # supervision. See data_terms.NullData.
    data_kwargs: dict = dataclass_field(default_factory=dict)
    direction: str = "forward"              # forward | bidirectional
    sigma: float = 0.1                      # data weight = 1/(2σ²), named per D4
    kinetic_weight: float = 1.0
    # Extra kinetic energy weighted per TEMPLATE vertex, ON TOP of kinetic_weight's
    # uniform one (see losses/kinetic.py). 0 = not built. kinetic_field_kwargs is a
    # PointWeightFieldCfg block: {field_name, field_source, field_transform,
    # field_scale, normalize}. Resolved in runner._add_kinetic_field_term.
    kinetic_field_weight: float = 0.0
    kinetic_field_kwargs: dict = dataclass_field(default_factory=dict)
    # Windowed functional-varifold data term (losses/fshape.py), added to the main
    # data term. 0 = not built. fshape_kwargs: FshapeCfg keys.
    fshape_weight: float = 0.0
    fshape_kwargs: dict = dataclass_field(default_factory=dict)
    code_reg_weight: float = 0.0
    # {start, end}: code_reg_weight is 0 before step start and ramps linearly to its value at
    # end, so the codes can take on the data before the penalty pulls them to zero. None = off.
    code_reg_warmup: Optional[dict] = None
    # {every, nu, weighting, rel_eps}: learn a Gaussian prior on the codes a frozen field.prior
    # reads by EM over their Laplace posteriors (callbacks/schedules/code_prior_em.py). None = off.
    code_prior_em: Optional[dict] = None
    weight_decay: float = 0.0               # Θ only — see Phase 6
    isometry_weight: float = 0.0            # disabled by default
    equivariant_deformation_weight: float = 0.0   # ||phi(g.T) - g.phi(T)||^2; 0 = not computed
    equivariant_deformation_kwargs: dict = dataclass_field(default_factory=dict)  # field_only, translation
    pose_supervision_weight: float = 0.0          # ||R_hat - R_aug||_F^2 against the drawn element; 0 = not computed
    pose_supervision_kwargs: dict = dataclass_field(default_factory=dict)        # translation
    flow_rotation_penalty_weight: float = 0.0     # ||R_flow - I||_F^2; fixes the pose/deformation gauge
    flow_rotation_penalty_kwargs: dict = dataclass_field(default_factory=dict)
    landmark_weight: float = 0.0            # 0 = disabled, matches isometry_weight
    # Kwargs for landmarks.build_landmark_term_kwargs (sigma_floor, heatmap_prefix,
    # label_weights, hard_precision) -- NOT constructor kwargs for the module
    # itself (unlike the other *_kwargs fields above), because LandmarkGuidanceLoss's
    # constructor takes already resolved tensors, not raw floats. See runner._build_pair.
    # label_weights: {label: weight} is BOTH the inclusion filter and a SOFT,
    # Mahalanobis-style multiplier on top of that label's annotator-agreement
    # -derived precision -- e.g. {n: 3.0, sn: 0.5} leans harder on the
    # nasion, eases off subnasale. A label the target annotated but that is
    # missing here, or given weight <= 0, is dropped before the template is
    # ever asked to resolve it -- the escape hatch for a label some subjects
    # annotate that a given template has no heatmap_<label> for. Omitted
    # entirely (None): every annotated label is used, uniformly weighted 1.0.
    # hard_precision: {label: precision} REPLACES that label's precision
    # outright (std and label_weights both discarded for it) -- the strict
    # option, for when a landmark's fit must be enforced hard rather than
    # softly weighted. A label can be in label_weights or hard_precision, never
    # both. See build_landmark_term_kwargs for exactly what "hard" can and
    # can't mean for a quadratic penalty under gradient descent.
    landmark_kwargs: dict = dataclass_field(default_factory=dict)
    isometry_type: str = "strain"           # or "det" / "orthogonal"
    isometry_samples: int = 64              # points per step for isometry (64 = ~5x speedup)
    # True keeps the isometry term a pure FIELD regularizer (the trajectory it reads
    # is detached). Set False with pose_application="pre" so the term can also reach
    # the pose -- otherwise the trajectory depends on R but the gradient still cannot
    # get there. Costs backprop through the integration chain.
    isometry_detach: bool = True
    subsample_M: float = 2000               # subsample source to N points before flow. int (absolute) or 0<x<1 (fraction); 0 = disabled
    save_full: bool = False                 # when subsampling: also compute and export full trajectory
    # Open-ended extra terms, resolved through the "loss_term" Registry category.
    # Each entry: {kind: <registry name>, weight: float, name: <metric key>, kwargs: {}}
    # Adding a loss is an entry here plus a registry line -- never a stepper edit.
    terms: List[dict] = dataclass_field(default_factory=list)

    def resolve_subsample_count(self, n_source: int) -> int:
        """Resolve subsample_M to absolute point count given source size.

        Args:
            n_source: number of source points

        Returns:
            Absolute number of points to subsample to (or 0 if disabled)
        """
        if self.subsample_M == 0:
            return 0
        if self.subsample_M >= 1:
            return int(self.subsample_M)
        # Treat as fraction
        return max(1, int(self.subsample_M * n_source))
