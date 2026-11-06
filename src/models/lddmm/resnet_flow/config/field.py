"""The velocity field config and the parsers for its modes, prior, symmetry, local field and widths."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import List, Optional
from src.shared.core.config import reject_unknown_keys


@dataclass
class FieldCfg:
    kind: str = "time_varying"              # time_varying | time_varying_fourier | time_varying_local | stationary | stationary_fourier
    num_steps: int = 10                     # K — configured in ONE place
    # Recompute each Euler step's field values in the backward pass instead of storing
    # them (integrators.Integrator): same gradients, far less memory, ~20-30% slower.
    grad_checkpointing: bool = False
    # Std of each velocity block's output layer at init. 0 (the default) is the zero init, so
    # the flow starts as the identity; a small value makes dv/dcode non-zero from the start,
    # which an auto-decoder needs to leave the z = 0, v = 0 saddle (see blocks.VelocityBlock).
    proj_init_std: float = 0.0
    mlp: tuple = (512,)                     # hidden widths per block, residual between them (ignored by stationary)
    activation: str = "relu"
    # "layer": a LayerNorm (with its affine gamma, beta) after each hidden linear map, before the
    # activation (fields/blocks.py); None = none. Fourier kinds only.
    norm: Optional[str] = None
    fa: tuple = (64, 64, 64)                # stationary: FA-NN layer widths
    df: tuple = (256, 256, 256, 256, 256)   # DF-NN layer widths
    fourier_n_e: int = 3                    # Fourier bands; also time_varying_fourier's, stationary_fourier's, and time_varying_local's COARSE field (0 = plain xyz there)
    # time_varying_fourier / stationary_fourier / time_varying_local (coarse field): {start, end} steps over
    # which the bands open, coarse to fine (callbacks/schedules/fourier.py FourierAnnealing).
    # None = all open from step 0.
    fourier_anneal: Optional[dict] = None
    # time_varying_local only: the fine field and its region -- see fields/local.py.
    # {landmarks, radius_mm, falloff_mm, mlp, fourier_n_e, fourier_anneal, heatmap_prefix}
    local: Optional[dict] = None
    # Designed deformation modes added to the field at a discounted kinetic cost --
    # see fields/modes.py. {spec, template, names, discount, lr, decompose}; None = no modes.
    modes: Optional[dict] = None
    # A frozen field of a finished run added to this one, which becomes the trainable
    # residual -- see fields/prior.py. {run, step, discount}; None = no prior.
    prior: Optional[dict] = None
    # Mirror symmetry of the trained network -- see fields/symmetry.py.
    # {mode, discount, pairs, midline, heatmap_prefix}; None = no symmetry.
    symmetry: Optional[dict] = None
    # Options of the removed equivariant_stationary field. Unused, but kept so that the config.json
    # of a finished run (asdict dumps every field) still loads.
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
            if kind not in FOURIER_FIELD_KINDS + ("time_varying_local",):
                raise ValueError(f"field.fourier_anneal only applies to kinds {FOURIER_FIELD_KINDS} "
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
        if d.get("prior") is not None:
            if kind not in MODE_FIELD_KINDS:
                raise ValueError(f"field.prior is supported for kinds {MODE_FIELD_KINDS}, not {kind!r}")
            d["prior"] = _parse_prior(d["prior"])
        if d.get("symmetry") is not None:
            if kind not in MODE_FIELD_KINDS:
                raise ValueError(f"field.symmetry is supported for kinds {MODE_FIELD_KINDS}, not {kind!r}")
            d["symmetry"] = _parse_symmetry(d["symmetry"])
        if d.get("norm") is not None:
            if d["norm"] not in NORMS:
                raise ValueError(f"unknown field.norm {d['norm']!r}; expected one of {list(NORMS)} or null")
            if kind not in FOURIER_FIELD_KINDS:
                raise ValueError(f"field.norm is supported for kinds {FOURIER_FIELD_KINDS}, not {kind!r}")
        return cls(**d)


# Field kinds a ModalField or a PriorResidualField can wrap: pointwise ones. The
# contextual kinds read the whole template (set_template), which the wrappers do not
# pass through.
MODE_FIELD_KINDS = ("time_varying", "time_varying_fourier", "time_varying_local", "stationary",
                    "stationary_fourier")


# Field kinds with ONE Fourier encoding of xyz, as ``.encoding``: what field.fourier_anneal
# opens (runner.complete_callbacks). time_varying_local has two, handled apart.
FOURIER_FIELD_KINDS = ("time_varying_fourier", "stationary_fourier")


# field.norm values: the normalisation inside each velocity block (fields/blocks.py).
NORMS = ("layer",)


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
               scripts/tools/decompose_modes.py CLI does it for a finished run).
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


def _parse_prior(prior) -> dict:
    """field.prior, validated, with defaults filled in -- a plain dict so config.json round-trips it.

    run       the output_dir of a finished run: its config.json gives the prior's
              architecture, its checkpoints/ the weights. The prior works in that
              run's frame, so a cohort run sets frame_source to the prior's members.
    step      which checkpoint of the run; None = final.pt, else the last step.
    discount  the prior's kinetic price relative to the residual network's: its
              energy is charged at discount * kinetic_weight. 1 = no cheaper than any
              other motion, 0 = free. Default 0.1.
    """
    if not isinstance(prior, dict):
        raise ValueError(f"field.prior must be a mapping {{run, step, discount}}, got {prior!r}")
    reject_unknown_keys(prior, {"run", "step", "discount"}, what="field.prior")
    if not isinstance(prior.get("run"), str) or not prior["run"]:
        raise ValueError(f"field.prior.run must be the output_dir of a finished run, got {prior.get('run')!r}")
    step = prior.get("step")
    discount = float(prior.get("discount", 0.1))
    if discount < 0:
        raise ValueError(f"field.prior.discount must be >= 0, got {discount}")
    return {"run": prior["run"], "step": None if step is None else int(step), "discount": discount}


# field.symmetry.mode: the field is mirror-symmetric, or its symmetric part is priced apart.
SYMMETRY_MODES = ("symmetric", "split")


def _parse_symmetry(symmetry) -> dict:
    """field.symmetry, validated, with defaults filled in -- a plain dict so config.json round-trips it.

    mode            symmetric = every flow is mirror-symmetric; split = the field is
                    unchanged, its symmetric part charged at discount * kinetic_weight and
                    the asymmetric rest at the full price
    discount        split only: the symmetric part's kinetic price relative to the
                    asymmetric part's. Default 0.3.
    pairs           [[right, left], ...] landmark labels mirrored by the plane: their
                    differences give its normal
    midline         landmark labels on the plane: with the pairs' midpoints they give
                    its position. Default none (midpoints only).
    heatmap_prefix  the template field of a label is f"{heatmap_prefix}{label}". Default "heatmap_".
    """
    if not isinstance(symmetry, dict):
        raise ValueError(f"field.symmetry must be a mapping {{mode, pairs, ...}}, got {symmetry!r}")
    reject_unknown_keys(symmetry, {"mode", "discount", "pairs", "midline", "heatmap_prefix"},
                        what="field.symmetry")
    mode = symmetry.get("mode")
    if mode not in SYMMETRY_MODES:
        raise ValueError(f"field.symmetry.mode must be one of {SYMMETRY_MODES}, got {mode!r}")
    if mode == "symmetric" and "discount" in symmetry:
        raise ValueError("field.symmetry.discount only applies to mode: split (a symmetric "
                         "field has no asymmetric part to price apart)")
    discount = float(symmetry.get("discount", 0.3)) if mode == "split" else None
    if discount is not None and discount < 0:
        raise ValueError(f"field.symmetry.discount must be >= 0, got {discount}")
    pairs = symmetry.get("pairs")
    if not isinstance(pairs, (list, tuple)) or not pairs or not all(
            isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(l, str) and l for l in p)
            for p in pairs):
        raise ValueError(f"field.symmetry.pairs must be a non-empty list of [right, left] landmark "
                         f"labels, e.g. [[enR, enL], [exR, exL]]; got {pairs!r}")
    midline = symmetry.get("midline", [])
    if not isinstance(midline, (list, tuple)) or not all(isinstance(l, str) and l for l in midline):
        raise ValueError(f"field.symmetry.midline must be a list of landmark labels, got {midline!r}")
    return {"mode": mode, "discount": discount, "pairs": [list(p) for p in pairs],
            "midline": list(midline), "heatmap_prefix": str(symmetry.get("heatmap_prefix", "heatmap_"))}


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
