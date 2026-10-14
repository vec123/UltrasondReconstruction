from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace as dataclass_replace
from typing import Optional, List, Any, Dict

from src.core.config import reject_unknown_keys

# Where joint_normalize places the shapes -- equivalently, where the ORIGIN sits
# relative to them. Not cosmetic: SO3Augmentation and the encoder pose are both
# applied as ``points @ R``, i.e. about the ORIGIN, so this decides whether a
# rotation is a spin or an orbit.
#
#   "box"    [0,1]^3, centroid at (0.5,0.5,0.5), 0.87 from the origin. A rotation
#            ORBITS the shape further than its own diameter, which gives chamfer a
#            strong pose signal (measured: 64x contrast vs 9x). But a rotation-only
#            pose can then only reach targets whose centroid stays on that sphere,
#            so samples that arrive already rotated in place are NOT reachable.
#   "origin" [-0.5,0.5]^3, centroid at the origin. A rotation is a spin in place, so
#            pre-rotated samples are reachable and the pose head's translation (an
#            absolute centroid) composes correctly. Weaker chamfer signal.
CENTERING_DOMAINS = {"box": (0.0, 1.0), "origin": (-0.5, 0.5)}

# Floating-point precision for the whole run. Named here, resolved to a torch dtype in
# the runner, so this module stays importable without torch.
#
# Nothing in the pipeline hardcodes precision -- every tensor derives its dtype from
# the data (``dtype=q0.dtype``, ``dtype=points.dtype``, ...) -- so this one key
# propagates everywhere by itself.
#
#   "float32"  the default, and what every result before this key was produced with.
#   "float64"  ~2x slower on CPU and considerably worse on GPU. Worth it as a
#              DIAGNOSTIC: the pose vectors are the residual of a first moment that
#              nearly cancels (measured ~1% of the per-node norms), and cancellation
#              is exactly where float32 loses relative precision -- 7 significant
#              digits in, ~5 left in the quantity actually used.
DTYPES = ("float32", "float64")


@dataclass
class FieldCfg:
    kind: str = "time_varying"              # time_varying | stationary | equivariant_stationary
    num_steps: int = 10                     # K — configured in ONE place
    mlp: tuple = (512,)                     # hidden widths per block, residual between them (ignored by stationary/equivariant_stationary)
    activation: str = "relu"
    fa: tuple = (64, 64, 64)                # stationary: FA-NN layer widths
    df: tuple = (256, 256, 256, 256, 256)   # DF-NN layer widths
    fourier_n_e: int = 3                    # Fourier bands; also time_varying_fourier's, and time_varying_local's COARSE field (0 = plain xyz there)
    # time_varying_fourier / time_varying_local (coarse field): {start, end} steps over
    # which the bands open, coarse to fine (callbacks/schedules/fourier.py FourierAnnealing).
    # None = all open from step 0.
    fourier_anneal: Optional[dict] = None
    # time_varying_local only: the fine field and its region -- see fields/local.py.
    # {landmarks, radius_mm, falloff_mm, mlp, fourier_n_e, fourier_anneal, heatmap_prefix}
    local: Optional[dict] = None
    # Designed deformation modes added to the field at a discounted kinetic cost --
    # see fields/modes.py. {spec, template, names, discount, lr, decompose}; None = no modes.
    modes: Optional[dict] = None
    # Equivariant stationary field (only if kind == "equivariant_stationary")
    hidden_irreps: str = "4x0e + 2x1o"      # intermediate irreps representation (single-layer fallback)
    gate_hidden_dim: int = 64               # hidden dim for gating network in SelfInteraction
    use_tensor_product_self: bool = True    # use SelfInteraction (V⊗V+gating) vs simple Linear
    layers: Optional[List[dict]] = dataclass_field(default=None)  # layer stack config for depth
    # Equivariant contextual field (only if kind == "equivariant_contextual")
    context_irreps: str = "32x0e + 16x1o"   # features after message passing aggregation
    sh_lmax: int = 2                        # spherical harmonics lmax for bipartite convolution
    # Simple equivariant contextual field (only if kind == "equivariant_contextual_simple")
    hidden_dim: int = 128                   # hidden dimension in MLP layers
    n_layers: int = 3                       # number of MLP layers

    @classmethod
    def from_dict(cls, d: dict) -> "FieldCfg":
        d = dict(d)
        # width: W is the old spelling of mlp: [W] -- the same block, parameter
        # names included. Translated rather than rejected, because every finished
        # run's config.json carries width (asdict dumps it for every kind) and
        # apply must keep rebuilding those models.
        if "width" in d:
            if "mlp" in d:
                raise ValueError(
                    f"field: width={d['width']} and mlp={d['mlp']} both given; "
                    f"width: W is the old spelling of mlp: [W]. Keep mlp only."
                )
            d["mlp"] = [d.pop("width")]
        if "mlp" in d:
            d["mlp"] = tuple(_parse_widths(d["mlp"], "field.mlp"))
        kind = d.get("kind", cls.kind)
        if d.get("fourier_anneal") is not None:
            if kind not in ("time_varying_fourier", "time_varying_local"):
                raise ValueError(f"field.fourier_anneal only applies to kind: time_varying_fourier "
                                 f"or time_varying_local, not {kind!r}")
            if kind == "time_varying_local" and int(d.get("fourier_n_e", cls.fourier_n_e)) < 1:
                raise ValueError("field.fourier_anneal on time_varying_local anneals the COARSE "
                                 "field's bands, but fourier_n_e is 0 (plain xyz): drop one of them")
            d["fourier_anneal"] = _parse_anneal(d["fourier_anneal"], "field.fourier_anneal")
        if kind == "time_varying_local":
            if int(d.get("fourier_n_e", cls.fourier_n_e)) < 0:
                raise ValueError("field.fourier_n_e must be >= 0 for time_varying_local (0 = plain coarse field)")
            d["local"] = _parse_local_field(d.get("local"))
        elif d.get("local") is not None:
            raise ValueError(f"field.local only applies to kind: time_varying_local, not {kind!r}")
        if d.get("modes") is not None:
            if kind not in MODE_FIELD_KINDS:
                raise ValueError(f"field.modes is supported for kinds {MODE_FIELD_KINDS}, not {kind!r}")
            d["modes"] = _parse_modes(d["modes"])
        return cls(**d)


# Field kinds a ModalField can wrap: pointwise ones. The contextual kinds read the
# whole template (set_template), which the wrapper does not pass through.
MODE_FIELD_KINDS = ("time_varying", "time_varying_fourier", "time_varying_local", "stationary")
# field.modes.decompose: what DecompositionExporter writes after training.
MODE_DECOMPOSITIONS = ("trajectories", "ends", "none")


def _parse_modes(modes) -> dict:
    """field.modes, validated, with defaults filled in -- a plain dict so config.json round-trips it.

    spec       the mode spec written by the deformation-mode editor (YAML)
    template   the mesh that spec was drawn on; the run's source must be a posed
               (similarity) copy of it, as similarity_transform writes
    names      which of the spec's modes to add, in order
    discount   the modes' kinetic price relative to the network's: their energy is
               charged at discount * kinetic_weight. 1 = no cheaper than any
               other motion, 0 = free. Default 0.1.
    lr         Adam learning rate of the amplitudes, which are in mm of peak
               speed: roughly how many mm per step a mode can open up. Default 0.05.
    decompose  after training, split the trained flow into network and modes and
               write it to <output_dir>/mode_decomposition/ (DecompositionExporter):
               trajectories = end shapes and every Euler step (~150 MB),
               ends = end shapes only, none = nothing (default; the
               scripts/LDDMM/decompose_modes.py CLI does it for a finished run).
    """
    if not isinstance(modes, dict):
        raise ValueError(f"field.modes must be a mapping {{spec, template, names, ...}}, got {modes!r}")
    reject_unknown_keys(modes, {"spec", "template", "names", "discount", "lr", "decompose"}, what="field.modes")
    for key in ("spec", "template"):
        if not isinstance(modes.get(key), str) or not modes[key]:
            raise ValueError(f"field.modes.{key} must be a file path, got {modes.get(key)!r}")
    names = modes.get("names")
    if not isinstance(names, (list, tuple)) or not names \
            or not all(isinstance(n, str) and n for n in names) or len(set(names)) != len(names):
        raise ValueError(f"field.modes.names must be a non-empty list of distinct mode names, "
                         f"got {names!r}")
    discount = float(modes.get("discount", 0.1))
    lr = float(modes.get("lr", 0.05))
    if discount < 0:
        raise ValueError(f"field.modes.discount must be >= 0, got {discount}")
    if lr <= 0:
        raise ValueError(f"field.modes.lr must be > 0, got {lr}")
    decompose = modes.get("decompose", "none")
    if decompose not in MODE_DECOMPOSITIONS:
        raise ValueError(f"field.modes.decompose must be one of {MODE_DECOMPOSITIONS}, got {decompose!r}")
    return {"spec": modes["spec"], "template": modes["template"], "names": list(names),
            "discount": discount, "lr": lr, "decompose": decompose}


def _parse_widths(mlp, what: str) -> list:
    """A non-empty list of positive int layer widths, as a list."""
    if not isinstance(mlp, (list, tuple)) or not mlp \
            or not all(isinstance(w, int) and w > 0 for w in mlp):
        raise ValueError(f"{what} must be a non-empty list of positive layer widths, "
                         f"e.g. [256, 256, 256]; got {mlp!r}")
    return list(mlp)


def _parse_anneal(anneal, what: str) -> dict:
    """A FourierAnnealing schedule, {start, end} with 0 <= start < end."""
    if not isinstance(anneal, dict):
        raise ValueError(f"{what} must be a {{start, end}} mapping, got {anneal!r}")
    reject_unknown_keys(anneal, {"start", "end"}, what=what)
    if "start" not in anneal or "end" not in anneal:
        raise ValueError(f"{what} needs both 'start' and 'end' steps, got {anneal}")
    start, end = int(anneal["start"]), int(anneal["end"])
    if start < 0 or end <= start:
        raise ValueError(f"{what} needs 0 <= start < end, got {anneal}")
    return {"start": start, "end": end}


def _parse_local_field(local) -> dict:
    """field.local for kind time_varying_local, validated, with defaults filled in.

    Returned as a plain dict so config.json round-trips it unchanged.
    """
    if not isinstance(local, dict):
        raise ValueError("kind: time_varying_local needs a field.local block: "
                         "{landmarks, radius_mm, falloff_mm, ...}")
    reject_unknown_keys(local, {"landmarks", "radius_mm", "falloff_mm", "mlp", "fourier_n_e",
                                "fourier_anneal", "heatmap_prefix"}, what="field.local")
    landmarks = local.get("landmarks")
    if not isinstance(landmarks, (list, tuple)) or not landmarks \
            or not all(isinstance(label, str) for label in landmarks):
        raise ValueError(f"field.local.landmarks must be a non-empty list of labels whose "
                         f"mean position is the region's centre; got {landmarks!r}")
    radius, falloff = local.get("radius_mm"), local.get("falloff_mm")
    if radius is None or falloff is None or float(radius) < 0 or float(falloff) <= 0:
        raise ValueError(f"field.local needs radius_mm >= 0 (full-strength ball) and "
                         f"falloff_mm > 0 (smooth edge); got {radius!r}, {falloff!r}")
    n_e = int(local.get("fourier_n_e", 5))
    if n_e < 1:
        raise ValueError(f"field.local.fourier_n_e must be >= 1, got {n_e}")
    anneal = local.get("fourier_anneal")
    return {
        "landmarks": list(landmarks),
        "radius_mm": float(radius),
        "falloff_mm": float(falloff),
        "mlp": _parse_widths(local.get("mlp", [128, 128]), "field.local.mlp"),
        "fourier_n_e": n_e,
        "fourier_anneal": None if anneal is None else _parse_anneal(anneal, "field.local.fourier_anneal"),
        "heatmap_prefix": str(local.get("heatmap_prefix", "heatmap_")),
    }


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
class CodeCfg:
    kind: str = "none"                      # none | auto_decoder | encoder
    n_z: int = 256
    position_aware: bool = False            # use grid interpolation (True) or broadcast (False)?
    conditioning_method: str = "concat"     # concat | film — how to apply features to velocity
    grid: int = 2                           # g — grid resolution (only if position_aware=True)
    grid_channels: int = 32                 # C — grid-head output per corner (only if position_aware=True)

    # Encoder-specific config (only used if kind == "encoder")
    encoder_config: Optional[EncoderConfig] = dataclass_field(default=None)
    graph_spec: Optional[GraphSpec] = dataclass_field(default=None)

    # The rigid pose this code source carries, if any. Orthogonal to `kind`.
    pose: PoseCfg = dataclass_field(default_factory=PoseCfg)


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
    """
    weight: float = 1.0
    base_sigma: float = 0.1
    field_name: Optional[str] = None
    field_source: Optional[str] = None
    field_transform: Optional[str] = None
    field_scale: Optional[float] = None
    loss_distance_handle: Optional[dict] = None
    inherit_source_field: bool = False

    TRANSFORMS = ("gaussian_distance",)

    @classmethod
    def from_dict(cls, d: dict) -> "ChamferSideCfg":
        reject_renamed_loss_distance_keys(d, "source_to_target/target_to_source block")
        reject_unknown_keys(d, {"weight", "base_sigma", "field_name", "field_source",
                                "field_transform", "field_scale", "loss_distance_handle",
                                "inherit_source_field"},
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
                   inherit_source_field=bool(d.get("inherit_source_field", False)))


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

@dataclass
class AugmentationCfg:
    """Configuration for source shape augmentation (random group transformations)."""
    kind: str = "none"                      # none | so3 | se3
    seed: Optional[int] = None              # Random seed for reproducibility
    translation_scale: float = 0.2          # Only for SE3: bounds on uniform translation


@dataclass
class InitCfg:
    """Start this run from a checkpoint instead of from fresh weights.

    Two orthogonal knobs rather than a resume/warm-start mode, because the two
    intents differ only in how they set them:

      * RESUME -- the same config, continued. Restore everything and keep
        counting: ``init: {checkpoint: outputs/prev/checkpoints}``, the defaults.
      * WARM START -- a DIFFERENT config (other loss weights, other target)
        initialised from a learned deformation. ``parts: [flow]`` leaves the
        optimizer fresh, and ``continue_step_count: false`` starts the new run's
        exports at step 0.

    Restoring is strict: ``load_state_dict`` fails naming the mismatched
    parameters when the config no longer describes the checkpoint, so a changed
    ``field.mlp`` is an error rather than a partially-loaded model.

    Attributes:
        checkpoint: a ``.pt`` file, or a run's ``checkpoints/`` directory -- see
            checkpoint.resolve_checkpoint for how each is resolved.
        step: pin an exact step within a checkpoints DIRECTORY. Meaningless
            alongside a ``.pt`` path, which already names one.
        parts: which pieces of the checkpoint to restore, out of PARTS.
        continue_step_count: carry on from the checkpoint's step rather than
            restarting at 0. On by default so a resumed run's checkpoints and
            exports do not overwrite the ones it resumed from. ``train.steps``
            means "run this many MORE steps" either way.
    """

    # The keys CheckpointWriter puts in the file, which are exactly the keys the
    # steppers' state_dict/load_state_dict speak. Validated here so a typo'd part
    # fails at config load rather than silently restoring nothing.
    PARTS = ("flow", "codes", "optimizer")

    checkpoint: str
    step: Optional[int] = None
    parts: tuple = PARTS
    continue_step_count: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "InitCfg":
        reject_unknown_keys(d, {"checkpoint", "step", "parts", "continue_step_count"},
                            what="init")
        if "checkpoint" not in d:
            raise ValueError(
                "init needs a 'checkpoint': a .pt file, or a run's checkpoints/ "
                "directory. Drop the whole init block to train from scratch."
            )

        parts = d.get("parts", cls.PARTS)
        if isinstance(parts, str):          # `parts: flow` -> ("flow",), not 4 chars
            parts = [parts]
        parts = tuple(parts)
        unknown = [p for p in parts if p not in cls.PARTS]
        if unknown:
            raise ValueError(
                f"unknown init.parts {unknown}; expected a subset of {list(cls.PARTS)}"
            )
        if not parts:
            raise ValueError(
                "init.parts is empty, so the checkpoint would be read and nothing "
                "restored. Drop the init block instead."
            )

        return cls(checkpoint=d["checkpoint"],
                   step=d.get("step"),
                   parts=parts,
                   continue_step_count=bool(d.get("continue_step_count", True)))


@dataclass
class TrainCfg:
    mode: str = "pair"                      # pair | cohort
    steps: int = 2000
    batch: int = 8
    lr: float = 1e-4
    seed: Optional[int] = None
    log_every: int = 50
    save_every: int = 500
    val_every: int = 0
    export_shapes: int = 0                  # 0 = all shapes (pair: always 1, cohort: min(batch_size, cohort_size))
    export_strategy: str = "sequential"     # sequential | random


@dataclass
class ExperimentCfg:
    source: str                             # pair: two paths; cohort: folder + template
    target: str
    output_dir: str
    # Folder of per-annotator landmark CSVs (Slicer fiducial export: x,y,z,label,
    # ...) in the same world frame as source/target. None = no landmark guidance;
    # only read when loss.landmark_weight > 0. See landmarks.py.
    landmarks: Optional[str] = None
    field: FieldCfg = dataclass_field(default_factory=FieldCfg)
    code: CodeCfg = dataclass_field(default_factory=CodeCfg)
    loss: LossCfg = dataclass_field(default_factory=LossCfg)
    augmentation: AugmentationCfg = dataclass_field(default_factory=AugmentationCfg)
    train: TrainCfg = dataclass_field(default_factory=TrainCfg)
    # Where the run's weights come from. None -- the default -- is a fresh model.
    init: Optional[InitCfg] = dataclass_field(default=None)
    centering: str = "box"                  # box | origin — see CENTERING_DOMAINS
    dtype: str = "float32"                  # float32 | float64 — see DTYPES
    # Apply the code source's learned pose to the flow output (unidirectional only).
    # Named `use_pose` in configs; `use_encoder_pose` is the original spelling and
    # stays valid, because the pose does not have to come from an encoder -- the
    # so3/se3 code sources carry one as free parameters. Both names set this field.
    use_encoder_pose: bool = False
    freeze_flow_at_identity: bool = False   # If True, skip flow computation; encoder learns pose only
    # Where the pose enters relative to the flow. "post" (pred = phi(T) @ R) leaves the
    # trajectory independent of R, so kinetic and isometry are EXACTLY constant in the
    # pose and only the data term can train it. "pre" (pred = phi(T @ R)) puts the
    # trajectory downstream of R, which is what makes "the correct pose needs the least
    # non-rigid flow" trainable. See UnidirectionalMappingError.
    pose_application: str = "post"          # post | pre

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentCfg":
        """Create ExperimentCfg from dict, erroring on unknown keys.

        Handles nested config for encoder_config and graph_spec.
        """
        reject_unknown_keys(d, {
            "source", "target", "output_dir", "landmarks", "field", "code", "loss",
            "augmentation", "train", "init", "centering", "dtype", "use_pose",
            "use_encoder_pose", "freeze_flow_at_identity", "pose_application",
        })

        # Two spellings of one flag. Conflicting values are a mistake worth naming
        # rather than resolving by precedence -- there is no reading of the config
        # under which the author meant both.
        use_pose = d.get("use_pose")
        use_encoder_pose = d.get("use_encoder_pose")
        if use_pose is not None and use_encoder_pose is not None \
                and bool(use_pose) != bool(use_encoder_pose):
            raise ValueError(
                f"use_pose={use_pose} contradicts use_encoder_pose={use_encoder_pose}; "
                f"they are the same flag. Keep one (use_pose is the current name)."
            )
        use_pose = bool(use_pose if use_pose is not None else (use_encoder_pose or False))

        field_cfg = FieldCfg.from_dict(d.get("field", {})) if "field" in d else FieldCfg()
        loss_cfg = LossCfg(**d.get("loss", {})) if "loss" in d else LossCfg()
        augmentation_cfg = AugmentationCfg(**d.get("augmentation", {})) if "augmentation" in d else AugmentationCfg()
        train_cfg = TrainCfg(**d.get("train", {})) if "train" in d else TrainCfg()
        init_cfg = InitCfg.from_dict(d["init"]) if d.get("init") else None

        centering = d.get("centering", "box")
        if centering not in CENTERING_DOMAINS:
            raise ValueError(
                f"unknown centering {centering!r}; expected one of "
                f"{sorted(CENTERING_DOMAINS)}"
            )

        dtype = d.get("dtype", "float32")
        if dtype not in DTYPES:
            raise ValueError(
                f"unknown dtype {dtype!r}; expected one of {list(DTYPES)}"
            )

        pose_application = d.get("pose_application", "post")
        if pose_application not in ("post", "pre"):
            raise ValueError(
                f"unknown pose_application {pose_application!r}; expected 'post' or 'pre'"
            )

        # Handle code config with special parsing for encoder_config and graph_spec
        code_dict = d.get("code", {})
        code_cfg = _parse_code_config(code_dict)

        return cls(
            source=d["source"],
            target=d["target"],
            output_dir=d["output_dir"],
            landmarks=d.get("landmarks"),
            field=field_cfg,
            code=code_cfg,
            loss=loss_cfg,
            augmentation=augmentation_cfg,
            train=train_cfg,
            init=init_cfg,
            centering=centering,
            dtype=dtype,
            use_encoder_pose=use_pose,
            freeze_flow_at_identity=d.get("freeze_flow_at_identity", False),
            pose_application=pose_application,
        )


def _parse_code_config(code_dict: dict) -> CodeCfg:
    """Parse code config, handling nested encoder_config and graph_spec.

    Separates encoder/graph config from base code config and creates
    EncoderConfig/GraphSpec objects when kind == "encoder".

    Note: EncoderConfig/GraphSpec are only imported when needed (lazy import),
    so basic config parsing doesn't require e3nn/torch_geometric.
    """
    # Extract nested configs before passing to CodeCfg
    encoder_cfg_dict = code_dict.pop("encoder_config", {})
    graph_cfg_dict = code_dict.pop("graph_spec", {})
    pose_cfg = PoseCfg.from_dict(code_dict.pop("pose", {}))

    # Parse encoder_config if provided and kind is encoder
    encoder_cfg = None
    if code_dict.get("kind") == "encoder":
        from src.spec import EncoderConfig, EncoderLayerConfig
        declares_latent_dim = "latent_dim" in encoder_cfg_dict
        if encoder_cfg_dict:
            # Parse layer configs
            layers_data = encoder_cfg_dict.pop("layers", [])
            layers = [EncoderLayerConfig(**layer) for layer in layers_data]
            encoder_cfg = EncoderConfig(layers=layers, **encoder_cfg_dict)
        else:
            # Use default EncoderConfig
            encoder_cfg = EncoderConfig()

        # n_z SEEDS latent_dim when encoder_config did not name one. Without this,
        # `code: {kind: encoder, n_z: 256}` trains a 5-dim code -- EncoderConfig's
        # default -- discarding the one dimension the config actually stated, and
        # saying so only in a printed warning. Via replace() rather than by editing
        # the dict before construction, because EncoderConfig.layers must stay
        # non-empty and defaults to a list this function would otherwise have to
        # rebuild. When BOTH are given the rule below is unchanged: the encoder wins.
        if not declares_latent_dim and "n_z" in code_dict:
            encoder_cfg = dataclass_replace(encoder_cfg, latent_dim=code_dict["n_z"])

        # Validate n_z matches encoder's latent_dim
        config_n_z = code_dict.get("n_z", 256)
        encoder_latent_dim = encoder_cfg.latent_dim
        if config_n_z != encoder_latent_dim:
            print(f"WARNING: code.n_z={config_n_z} does not match encoder_config.latent_dim={encoder_latent_dim}")
            print(f"         Using encoder_config.latent_dim={encoder_latent_dim} (encoder determines code dimension)")
            code_dict["n_z"] = encoder_latent_dim

    # Parse graph_spec if provided and kind is encoder
    graph_spec = None
    if code_dict.get("kind") == "encoder":
        from src.spec import GraphSpec
        graph_spec = GraphSpec(**graph_cfg_dict) if graph_cfg_dict else GraphSpec()

    return CodeCfg(encoder_config=encoder_cfg, graph_spec=graph_spec, pose=pose_cfg,
                   **code_dict)
