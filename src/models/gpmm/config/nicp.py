"""NICP matching and optimization configs, and the per-fit specs they resolve to."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass
class CorrespondenceSpec:
    """Generic selector for a src.nicp correspondence method.

    ``name`` keys the strategy registry and ``kwargs`` is handed to it verbatim,
    so a method can be added without touching this schema. Unknown keys are
    rejected when the strategy is built, not silently dropped.
    """

    name: str = "constrained"                          # none | nearest | constrained
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchingConfig:
    """The pipeline's flat spelling of a :class:`CorrespondenceSpec`.

    The keys are typed and validated here (rather than living in a free-form
    kwargs mapping) because they are the ones the pipeline has been tuned with.
    ``to_spec()`` converts to the generic form the registry consumes.
    """

    # Correspondence method for the NICP data term:
    #   constrained: nearest neighbour among the k closest + distance/normal rejection
    #   nearest:     plain 1-NN for every model vertex, nothing rejected
    #   none:        no pairing at all - only usable with a data term that
    #                associates the surfaces itself (Chamfer and friends), which
    #                the pipeline's MAP step does not, so it is rejected there.
    mode: str = "constrained"
    dist_thresh: float = 0.4          # constrained only: max match distance
    normal_thresh_deg: float = 90.0   # constrained only: max angle between matched normals
    k: int = 5                        # constrained only: candidates examined per model vertex
    # False reproduces the original script, which passed the target *positions*
    # where the matcher expects the target *normals*. True uses the computed
    # target normals instead (changes the resulting match set).
    use_target_normals: bool = False
    # nearest/constrained only: reject many-to-one matches in either direction,
    # keeping only the closest pair per source vertex and per target point (see
    # src.shared.correspondence.compute.heuristic_estimate.deduplicate_pairs). Off by default - several model
    # vertices legitimately sharing one nearest target point is normal, not an
    # error, when the target is denser than the model.
    one_to_one: bool = False

    MODES = ("none", "nearest", "constrained")

    def __post_init__(self):
        if self.mode not in self.MODES:
            raise ValueError(f"matching.mode must be one of {list(self.MODES)}, got {self.mode!r}")

    def to_spec(self) -> CorrespondenceSpec:
        """Only the arguments the chosen method actually accepts.

        ``dist_thresh`` is this config's name for what the strategies call
        ``max_dist``; the rejection vocabulary is uniform across pairing methods
        there, while the pipeline keeps the name it was written with.
        """
        if self.mode == "constrained":
            kwargs = {
                "max_dist": self.dist_thresh,
                "normal_thresh_deg": self.normal_thresh_deg,
                "k": self.k,
                "use_target_normals": self.use_target_normals,
                "one_to_one": self.one_to_one,
            }
        elif self.mode == "nearest":
            kwargs = {"one_to_one": self.one_to_one}
        else:
            kwargs = {}
        return CorrespondenceSpec(name=self.mode, kwargs=kwargs)


@dataclass
class NicpVariant:
    name: str                             # used in the output filename
    gamma: float                          # MAP regularisation strength
    chi_clamp: Optional[float] = None     # plausibility clamp, null = off
    learning_rate: float = 1.0


@dataclass
class ModelOptimizationConfig:
    """The fit half of one NICP round: a MAP-regularised solve against whatever
    correspondence set ``nicp.matching`` most recently produced."""

    # Regularisation sweep: one fit + one .vtp per gamma, keyed "gamma_<value>".
    gammas: List[float] = field(
        default_factory=lambda: [0, 100, 10000, 20000, 30000, 50000, 80000, 100000]
    )
    learning_rate: float = 1.0
    chi_clamp: Optional[float] = None
    # Extra one-off fits, keyed by name.
    variants: List[NicpVariant] = field(
        default_factory=lambda: [
            NicpVariant(name="free", gamma=0, chi_clamp=None),
            NicpVariant(name="clamped", gamma=0, chi_clamp=0.99),
        ]
    )


@dataclass
class NicpConfig:
    """Step 2: a list of fits, each one a full (correspondence, loss, prior,
    solver, projection) choice against the step 1 result.

    Two spellings, and ``fits`` wins when both are present:

    ``fits`` (preferred) is the same language ``configs/gpmm/fit/nicp/nicp.yaml`` speaks -
    each entry is a partial override of ``defaults``, validated into a
    :class:`NicpFitSpec`. It is the only spelling that can reach a
    correspondence-free objective, because it can say ``correspondence: none``
    and name a loss that makes its own pairs.

    ``matching`` + ``model_optimization`` is the original flat spelling, kept
    working by desugaring into the same specs. It can only ever express matched
    L2 solved in closed form, which is what it was built for.
    """

    iterations: int = 1
    # Write an extra snapshot of the fitted surface every Nth round (1 = every
    # round). The canonical result is always the final round regardless.
    save_every: int = 1

    # -- compute ---------------------------------------------------------------
    device: str = "auto"           # auto | cpu | cuda
    dtype: str = "float64"         # float64 matches the original numpy solve
    seed: Optional[int] = 0        # seeds any subsampling a loss does
    # false treats the model as a point cloud, which is what this stage has
    # always done. true attaches the triangulation, so model normals come from
    # the surface and are consistently oriented - better matching, but it
    # changes results, so flip it deliberately.
    use_model_faces: bool = False

    # -- preferred spelling ----------------------------------------------------
    defaults: Dict[str, Any] = field(default_factory=dict)
    fits: List[Dict[str, Any]] = field(default_factory=list)

    # -- legacy spelling, desugared into the above -----------------------------
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    model_optimization: ModelOptimizationConfig = field(default_factory=ModelOptimizationConfig)

    def __post_init__(self):
        if self.iterations < 1:
            raise ValueError(f"nicp.iterations must be >= 1, got {self.iterations}")
        if self.save_every < 1:
            raise ValueError(f"nicp.save_every must be >= 1, got {self.save_every}")

    def resolve_fits(self, client) -> List["NicpFitSpec"]:
        """The fits this configuration runs, in order, whichever spelling it used."""
        entries = self.fits or self._desugar_legacy()
        if not entries:
            raise ValueError(
                "nicp has no fits: add entries under 'nicp.fits', or gammas/variants "
                "under 'nicp.model_optimization'"
            )
        return resolve_fit_specs(self.defaults, entries, client, source="nicp.fits")

    def _desugar_legacy(self) -> List[Dict[str, Any]]:
        """``matching`` + ``model_optimization`` as ``fits`` entries.

        The legacy ``gamma`` becomes ``solver.damping``, NOT a prior. The two
        are different equations: a prior on the coefficients carries a
        ``- gamma * alpha`` term on the right-hand side and pulls the fit
        towards the mean, while ``ShapeInstance.l2_closed_form_step`` - what
        this spelling used to call - has no such term and only shortens the
        step. Mapping gamma onto ``prior: isotropic`` would silently move every
        result this spelling has ever produced.
        """
        opt = self.model_optimization
        correspondence = self.matching.to_spec()

        def entry(name, gamma, learning_rate, chi_clamp):
            spec = {
                "name": name,
                "correspondence": {"name": correspondence.name, "kwargs": correspondence.kwargs},
                "loss": {"name": "matched_l2", "kwargs": {"reduction": "sum"}},
                "prior": {"name": "none"},
                "solver": {"name": "normal_equations",
                           "kwargs": {"iterations": self.iterations,
                                      "learning_rate": learning_rate,
                                      "damping": float(gamma)}},
                "refresh_every": 1,
            }
            if chi_clamp is not None:
                spec["projection"] = {"name": "mahalanobis_clamp", "kwargs": {"p": chi_clamp}}
            return spec

        entries = [entry(f"gamma_{_format_gamma(g)}", g, opt.learning_rate, opt.chi_clamp)
                   for g in opt.gammas]
        entries += [entry(v.name, v.gamma, v.learning_rate, v.chi_clamp) for v in opt.variants]
        return entries


def _format_gamma(gamma) -> str:
    """``5`` not ``5.0``, so a desugared sweep keeps the filenames it always had."""
    return str(int(gamma)) if float(gamma).is_integer() else str(gamma)


def deep_merge(base: Mapping, override: Mapping) -> Dict[str, Any]:
    """Recursive dict merge; ``override`` wins on leaves.

    One special case: a block that names an implementation is replaced wholesale
    when the name changes, instead of being merged. ``kwargs`` belong to the
    implementation, not to the slot it sits in, so a fit that switches from
    ``normal_equations`` to ``gradient`` must not inherit the former's
    ``learning_rate``.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            if value.get("name", current.get("name")) != current.get("name"):
                merged[key] = dict(value)
            else:
                merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def resolve_fit_specs(defaults: Mapping, entries: Sequence[Mapping], client,
                      source: str = "fits") -> List["NicpFitSpec"]:
    """Merges each entry onto ``defaults`` and validates it into a NicpFitSpec.

    Shared by the standalone runner's ``runs`` and the pipeline stage's
    ``fits``, so one fit means the same thing in both. Every entry is resolved
    before any of them runs: a typo in the last of twelve fails immediately
    rather than after the first eleven have been fitted.
    """
    specs, seen = [], set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{source}[{index}] must be a mapping, got {type(entry).__name__}")
        if "name" not in entry:
            raise ValueError(f"{source}[{index}] is missing 'name'; it becomes the file name")

        spec = client.from_dict(deep_merge(defaults, entry), NicpFitSpec,
                                source=f"{source}[{index}]")
        if spec.name in seen:
            raise ValueError(f"duplicate fit name '{spec.name}' in {source}; names become files")
        seen.add(spec.name)
        specs.append(spec)
    return specs


# --------------------------------------------------------------------------
# the ssm_nicp stage - the NICP step on its own
# --------------------------------------------------------------------------

@dataclass
class ComponentSpec:
    """Generic selector for a registered src.nicp component.

    ``name`` keys the relevant registry (loss, prior, solver, projection) and
    ``kwargs`` is handed to the constructor, which rejects keys it does not
    accept. New components are therefore usable from YAML without touching this
    schema.
    """

    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NicpFitSpec:
    """One fit: which pieces, and how they are configured.

    Each entry of ``NicpFitConfig.runs`` is deep-merged onto
    ``NicpFitConfig.defaults`` and validated into one of these, so a run only
    states what it changes.
    """

    name: str = "fit"
    correspondence: CorrespondenceSpec = field(default_factory=CorrespondenceSpec)
    loss: ComponentSpec = field(default_factory=lambda: ComponentSpec("matched_l2"))
    prior: ComponentSpec = field(default_factory=lambda: ComponentSpec("none"))
    solver: ComponentSpec = field(default_factory=lambda: ComponentSpec("normal_equations"))
    projection: ComponentSpec = field(default_factory=lambda: ComponentSpec("none"))
    # 0 estimates correspondences once from the initial shape (what the pipeline
    # does); n > 0 re-estimates them every n iterations, i.e. a real ICP loop.
    refresh_every: int = 0
