"""Config object classes, one tree per script under ``scripts/``.

Fields without a default are required in the YAML file; everything else may be
omitted and falls back to the value declared here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

# NOTE: annotations stay in the typing.List / typing.Dict spelling rather than
# the PEP 585 list[...] / dict[...] one. The config client resolves them with
# typing.get_type_hints(), which cannot evaluate the builtin generics on
# Python 3.8 - the version of the torch environment this project runs in.


# --------------------------------------------------------------------------
# scripts/vtu_2_vtp.py
# --------------------------------------------------------------------------

@dataclass
class VtuToVtpConfig:
    input_vtu: Path
    output_vtp: Path
    filter_largest_component: bool = True
    verbose: bool = True


# --------------------------------------------------------------------------
# scripts/sim_tf.py
# --------------------------------------------------------------------------

@dataclass
class SimTransformInputs:
    """Meshes to transform: either a whole directory or an explicit list."""

    dir: Optional[Path] = None
    pattern: str = "*.vtp"
    exclude: List[str] = field(default_factory=list)   # file names skipped when scanning dir
    files: List[Path] = field(default_factory=list)

    def __post_init__(self):
        if (self.dir is None) == (not self.files):
            raise ValueError("set exactly one of inputs.dir or inputs.files")


@dataclass
class SimTransformConfig:
    # .npz holding R (3,3), t (3,) - 'T' is also accepted - and s (scalar).
    # Written by scripts/pipeline.py (gmm_2_template.npz) and by
    # scripts/align_landmarks.py (rigid_transform.npz).
    transform: Path
    inputs: SimTransformInputs
    output_dir: Path
    inverse: bool = False              # apply the inverse of the saved transform
    # Appended to each stem. Omit it or set null to keep the original file name
    # (only allowed when output_dir differs from the source directory).
    suffix: Optional[str] = None
    verbose: bool = True


# --------------------------------------------------------------------------
# scripts/align_landmarks.py
# --------------------------------------------------------------------------

@dataclass
class AlignLandmarksOutputs:
    dir: Path
    transform: str = "rigid_transform.npz"
    original_mesh: str = "original_mean_mesh.vtp"
    aligned_mesh: str = "aligned_mean_mesh.vtp"

    def path(self, key: str) -> Path:
        return self.dir / getattr(self, key)


@dataclass
class AlignLandmarksConfig:
    shape_model_mat: Path
    output: AlignLandmarksOutputs
    # Either give the targets inline (name -> [x, y, z]) or point at a landmark JSON.
    target_landmarks: Dict[str, List[float]] = field(default_factory=dict)
    target_landmarks_json: Optional[Path] = None
    allow_scale: bool = True
    verbose: bool = True

    def __post_init__(self):
        if bool(self.target_landmarks) == (self.target_landmarks_json is not None):
            raise ValueError(
                "set exactly one of 'target_landmarks' (inline) or 'target_landmarks_json'"
            )


# --------------------------------------------------------------------------
# scripts/pipeline.py
# --------------------------------------------------------------------------

@dataclass
class PipelineInputs:
    # BabyFaceModel .mat statistical shape model, or a Deformetrica atlas
    # directory (see AtlasConfig). The atlas carries no landmarks, so
    # steps.align and steps.landmark_fit have to be off when one is used.
    shape_model_mat: Path
    mesh_vtp: Path               # target surface (ultrasound) to fit
    # Target landmark positions, name -> [x, y, z]. Only read by steps.align
    # and steps.landmark_fit; may be omitted when both are off.
    landmarks_json: Optional[Path] = None
    # The model's own mean-shape mesh (e.g. Templates/baby_mean.vtp), vertex-wise
    # corresponding with shape_model_mat - same vertex count, same ordering. The
    # model format (.mat or Deformetrica atlas) carries geometry and landmarks
    # but no arbitrary point-data fields, so every mesh this pipeline writes is
    # bare; if this is set, PipelineRunner._save_points copies template_vtp's
    # point-data arrays (e.g. heatmap_<label>, which resnet_lddmm's landmark
    # loss reads off the source mesh) onto any output whose point count matches
    # it. Optional - leave unset if nothing downstream needs those fields.
    template_vtp: Optional[Path] = None


@dataclass
class StepsConfig:
    """Which stages run. Every stage writes its shape into the result registry
    under a key, and later stages address earlier results by that key."""

    gmm_2_template: bool = False  # 0b: solve the GMM <-> template similarity transform
    align: bool = True            # 1a: similarity transform of the target surface into model space
    landmark_fit: bool = True     # 1b: fit SSM coefficients to the landmarks
    nicp: bool = True             # 2:  correspondence + MAP-regularised surface fit
    laplacian: bool = True        # 3:  spectral operators + regional deformation


@dataclass
class GmmToTemplateConfig:
    """Similarity transform between GMM (model) space and template space.

    template_vtp must be vertex-wise corresponding with the model, i.e. carry
    exactly n_vertices points in the model's ordering, which lets the transform
    be solved in closed form over every vertex.
    """

    template_vtp: Path
    # to_model:    map the template (and the target surface) into GMM space.
    #              PREFERRED - the SSM is untouched, so eigenfunctions and
    #              eigenvalues stay meaningful for sampling, and gamma/chi_clamp
    #              keep the meaning they were tuned with. Distance thresholds
    #              must then be given in model units.
    # to_template: rescale the SSM itself into template space (mean and
    #              eigenfunctions by s, eigenvalues by s^2). A valid change of
    #              variables in theory, but it invalidates two things here:
    #                * clamp_coefficients() hardcodes beta_sq = 10000**2 rather
    #                  than deriving it from the eigenvalues, so chi_clamp
    #                  silently changes meaning;
    #                * l2_closed_form_step solves (Phi.T @ Phi + gamma*I), and
    #                  Phi.T @ Phi scales by s^2, so nicp.gammas must be
    #                  rescaled by s^2 to mean the same thing.
    direction: str = "to_model"
    allow_scale: bool = True
    apply_to_target: bool = True   # to_model only: also map inputs.mesh_vtp + landmarks

    def __post_init__(self):
        directions = ("to_model", "to_template")
        if self.direction not in directions:
            raise ValueError(
                f"gmm_2_template.direction must be one of {list(directions)}, "
                f"got {self.direction!r}"
            )


@dataclass
class AtlasConfig:
    """How a Deformetrica atlas is turned into a linear SSM.

    Only read when the model path points at an atlas directory rather than a
    .mat file; ignored otherwise, since a .mat already holds a linear model.
    """

    # Deformation kernel width (sigma). null reads 'kwd' from the atlas params.json.
    kernel_width: Optional[float] = None
    # Geodesic shooting resolution. 11 is Deformetrica's default and is
    # converged here to ~0.03 mm; 2 would be a plain linear approximation and
    # is off by several mm on the larger deformations.
    time_points: int = 11
    # Mode truncation. null/1.0 keeps every non-degenerate mode (at most
    # n_subjects - 1 of them). The stricter of the two limits wins.
    n_components: Optional[int] = None
    variance_threshold: float = 1.0
    # Cache the shot decomposition next to the atlas - shooting the population
    # takes ~90 s, reading it back takes under a second.
    cache: bool = True
    # name -> vertex-index JSON annotating the atlas's mean shape, written by
    # scripts/place_landmarks.py (see src/model/landmarks.py). null leaves the
    # atlas with no landmarks, as before - every landmark-driven field then
    # stays off, same as documented in configs/mean_shape_fetus.yaml.
    landmarks: Optional[Path] = None

    def to_options(self) -> Dict[str, Any]:
        """The keyword arguments load_shape_model() hands to the atlas backend."""
        return {
            "kernel_width": self.kernel_width,
            "time_points": self.time_points,
            "n_components": self.n_components,
            "variance_threshold": self.variance_threshold,
            "cache": self.cache,
            "landmarks": self.landmarks,
        }


@dataclass
class ModelConfig:
    normalize: bool = False      # normalize the SSM to unit RMS scale on load
    atlas: AtlasConfig = field(default_factory=AtlasConfig)


@dataclass
class GpmmModelConfig:
    """Configuration for building one Gaussian Process Morphable Model.

    The kernel tree is intentionally untyped internally: composite kernels
    recurse through arbitrary ``children`` and each registered kernel owns its
    own kwargs validation. ``ConfigClient`` still validates the model-level
    fields and interpolates paths before ``build_model`` receives the mapping.
    """

    domain: Dict[str, Any]
    kernel: Dict[str, Any] = field(default_factory=lambda: {"name": "pca"})
    approximation: Optional[Dict[str, Any]] = None
    truncation: Optional[Dict[str, Any]] = None
    inference: Optional[Dict[str, Any]] = None
    normalize: bool = True
    output: "GpmmModelOutputConfig" = field(default_factory=lambda: GpmmModelOutputConfig())


@dataclass
class GpmmModelOutputConfig:
    """Destination for the serialized model archive."""

    dir: Path = Path("outputs/gpmm")
    filename: str = "model.npz"

    @property
    def path(self) -> Path:
        return self.dir / self.filename


# --------------------------------------------------------------------------
# scripts/align_meshes.py
# --------------------------------------------------------------------------

@dataclass
class MeshSource:
    """A mesh, given as a file, as a model's mean shape, or as both.

    The two are complementary rather than alternatives:

    * ``mesh_vtp`` supplies the geometry *and everything attached to it* - the
      per-vertex fields written by scripts/mean_shape.py travel through the
      transform with the points, so the aligned pair keeps its heatmaps.
    * ``shape_model`` supplies the landmarks, which are what the alignment is
      sanity-checked against, and the mean shape when no mesh is given.

    Give both to get both. They are then required to agree vertex for vertex,
    since they are supposed to be the same shape.
    """

    mesh_vtp: Optional[Path] = None
    # .mat or Deformetrica atlas.
    shape_model: Optional[Path] = None
    normalize: bool = False          # shape_model only
    # Split every polygon into triangles on load. A mesh clipped in ParaView
    # comes back with mixed cell sizes, which normals and any downstream
    # surface method would rather not see. Point data is carried over.
    triangulate: bool = True

    def __post_init__(self):
        if self.mesh_vtp is None and self.shape_model is None:
            raise ValueError("set at least one of mesh_vtp or shape_model")


@dataclass
class AlignMeshesOutputs:
    dir: Path
    transform: str = "similarity_transform.npz"
    aligned: str = "aligned_source.vtp"
    target: str = "target.vtp"
    write_target: bool = True        # keep both meshes in one frame, in one folder

    def path(self, key: str) -> Path:
        return self.dir / getattr(self, key)


@dataclass
class FocusConfig:
    """Where the alignment should spend its accuracy.

    An unweighted ICP minimises the average distance over the whole surface,
    which is the right thing only if every part matters equally. It does not
    here: a fetal and a neonatal face differ everywhere, and spreading that
    difference evenly leaves the one landmark a nonrigid step most needs
    anchored - the nose - as wrong as the cheeks.

    Weighting the correspondences concentrates the fit without dropping the
    rest of the surface, so the transform stays well determined.
    """

    # Landmark of the *target* model to centre the weighting on.
    landmark: str = "prn"
    # Falloff of the weight away from it, in target units - a Gaussian, so
    # there is no hard edge for the ICP to chatter across.
    radius: float = 0.003
    # Weight at the centre relative to the far field. 1 disables the focus.
    weight: float = 8.0
    # Use a point-data array of the target as the weight instead, e.g. one of
    # mean_shape.py's 'heatmap_<landmark>' fields, which follow the surface
    # rather than a Euclidean radius. Overrides landmark/radius when set.
    field: Optional[str] = None


@dataclass
class AlignMeshesConfig:
    """Similarity alignment of two meshes with no correspondence between them.

    Solves ``x_target ~= s R x_source + t`` from geometry alone, as the initial
    guess for a nonrigid step - it will not remove a shape difference, only the
    placement, orientation and scale.
    """

    source: MeshSource               # the mesh that moves
    target: MeshSource               # the frame to move it into
    outputs: AlignMeshesOutputs
    allow_scale: bool = True         # similarity (True) vs rigid-only (False)
    # null weights every correspondence equally.
    focus: Optional[FocusConfig] = field(default_factory=FocusConfig)
    # Principal axes match in 24 proper rotations; all are screened and this
    # many are refined with ICP. Lowering it risks keeping a near-miss that
    # would have lost to a slower starter.
    refine_count: int = 8
    iterations: int = 100            # ICP iterations per candidate
    # Fraction of the worst-matched source points dropped each iteration. The
    # two surfaces do not cover the same region, so an unmatched rim would drag
    # the transform outward if every point had to be explained.
    trim_ratio: float = 0.1
    tolerance: float = 1.0e-12       # ICP convergence: change in mean distance
    verbose: bool = True


# --------------------------------------------------------------------------
# scripts/sample_shapes.py
# --------------------------------------------------------------------------

@dataclass
class ShapeModelSource:
    """One selectable shape model, keyed by name under 'models'.

    Carries the viewing orientation as well as the model itself, because the
    two models do not share a coordinate convention - the axis that faces the
    camera for one is not the axis for the other.
    """

    # .mat statistical shape model, or a Deformetrica atlas directory.
    path: Path
    normalize: bool = False
    atlas: AtlasConfig = field(default_factory=AtlasConfig)

    # How a sample is built from the coefficients:
    #
    #   linear  x = mean + sum_k alpha_k phi_k. PCA on vertex coordinates, so a
    #           sample costs one matrix product. What every other stage of this
    #           project consumes.
    #   lddmm   PCA on the initial momenta, then geodesic shooting (atlas only).
    #           Every sample is a diffeomorphism of the template, so it cannot
    #           fold the surface through itself, and large deformations stay on
    #           the manifold instead of being summed in the ambient space. Costs
    #           an integration per sample - see lddmm_time_points.
    #
    # Measured on the fetus atlas, a +2 sd LDDMM sample sits 0.077 mm (max)
    # from its best fit in the full linear basis, on a sample that moves
    # 7.14 mm: near the population the two agree closely, and they part company
    # further out.
    parametrization: str = "linear"
    # Shooting resolution while a slider is being dragged. 6 is converged to
    # ~0.14 mm and takes ~0.3 s a sample; 11 is Deformetrica's default at ~0.6 s.
    lddmm_time_points: int = 6

    # Signed axis the camera looks *down* onto the shape, and which way is up.
    camera_axis: str = "+z"
    view_up: str = "+y"

    AXES = ("+x", "-x", "+y", "-y", "+z", "-z")
    PARAMETRIZATIONS = ("linear", "lddmm")

    def __post_init__(self):
        if self.parametrization not in self.PARAMETRIZATIONS:
            raise ValueError(
                f"parametrization must be one of {list(self.PARAMETRIZATIONS)}, "
                f"got {self.parametrization!r}"
            )
        if self.parametrization == "lddmm" and not self.path.is_dir():
            raise ValueError(
                f"parametrization 'lddmm' needs a Deformetrica atlas directory - the "
                f"momenta and the kernel it shoots with are not in a .mat: {self.path}"
            )
        for name in ("camera_axis", "view_up"):
            value = getattr(self, name)
            if value not in self.AXES:
                raise ValueError(f"{name} must be one of {list(self.AXES)}, got {value!r}")
        if self.camera_axis[1] == self.view_up[1]:
            raise ValueError(
                f"camera_axis and view_up cannot share an axis "
                f"({self.camera_axis!r} / {self.view_up!r})"
            )


@dataclass
class SlidersConfig:
    """The mode sliders: one per mode, in units of that mode's sigma.

    The panel scrolls, so 'how many modes exist' and 'how many are on screen'
    are separate numbers: every mode is reachable, and only ``visible_rows`` of
    them are drawn at once. The rows are a fixed pool of widgets re-bound to a
    sliding window of modes as it scrolls, so the cost does not grow with a
    model that has 183 of them.
    """

    # How many leading modes the panel exposes. null exposes every mode in the
    # model; a number truncates, and the modes past it keep alpha = 0.
    mode_count: Optional[int] = None
    # Sliders drawn at once - the height of the scrolling viewport, in rows.
    # Past ~14 the rows start to crowd at a 900 px window height.
    visible_rows: int = 10
    # Slider travel, in standard deviations either side of the mean. The model
    # is Gaussian, so +/-3 covers 99.7% of the training population; further out
    # is extrapolation and starts to look like it.
    sigma_range: float = 3.0

    # Keep the slider positions when the window switches model, instead of
    # dropping back to the mean. What carries over is the reading in *standard
    # deviations*, not the raw coefficient - lambda differs by orders of
    # magnitude between models, so alpha is not transferable and sigma is. The
    # point is to hold one set of slider positions still and watch what a
    # different kernel does with it.
    #
    # Modes are matched by index, and index is all they share: mode 3 of a
    # Matern model is not mode 3 of a Gaussian one, and across the two domains
    # they are unrelated. Within one domain the leading modes usually do
    # correspond, which is where this is worth using. Surplus modes are left at
    # zero and coefficients past the new model's mode count are dropped.
    carry_across_models: bool = True

    def __post_init__(self):
        if self.visible_rows < 1:
            raise ValueError(f"sliders.visible_rows must be at least 1, got {self.visible_rows}")
        if self.mode_count is not None and self.mode_count < 1:
            raise ValueError(f"sliders.mode_count must be at least 1, got {self.mode_count}")


@dataclass
class SampleShapesView:
    window_size: List[int] = field(default_factory=lambda: [1400, 900])
    # light | dark. Sets the whole palette, not just the background: the
    # sequential ramp runs the other way in each so that near-zero
    # displacement always recedes towards the surface. 't' toggles it live.
    theme: str = "light"

    #   displacement         |x - mean| per vertex, recomputed as the sliders
    #                        move - what the current coefficients are doing
    #   signed_displacement  (x - mean) . n, the component along the *mean*
    #                        shape's outward normal: positive where the sample
    #                        bulges out of the mean surface, negative where it
    #                        dents in. Drawn on a diverging ramp centred at 0.
    #                        Only the normal component - a vertex that slides
    #                        tangentially reads 0 - so |signed| <= displacement.
    #   total_std            the model's per-vertex standard deviation, fixed -
    #                        where the population varies at all
    #   none                 a flat surface colour, lit only by shading
    #
    # This is the *initial* mode; 'c' cycles through them in the window.
    color_by: str = "displacement"
    # Upper end of the colour ramp, in model units. null derives it from the
    # model as sliders.sigma_range * max total_std, which keeps the mapping
    # fixed while the sliders move - a range that rescaled every frame would
    # make the colours meaningless.
    color_range: Optional[float] = None
    show_scalar_bar: bool = True
    show_edges: bool = False
    # The mean shape held behind the sample as a translucent ghost, so the
    # displacement is visible as a shape rather than only as a colour.
    show_mean: bool = False
    mean_opacity: float = 0.15

    COLOR_MODES = ("displacement", "signed_displacement", "total_std", "none")
    THEMES = ("light", "dark")

    def __post_init__(self):
        if self.color_by not in self.COLOR_MODES:
            raise ValueError(
                f"view.color_by must be one of {list(self.COLOR_MODES)}, got {self.color_by!r}"
            )
        if self.theme not in self.THEMES:
            raise ValueError(f"view.theme must be one of {list(self.THEMES)}, got {self.theme!r}")
        if len(self.window_size) != 2:
            raise ValueError(f"view.window_size must be [width, height], got {self.window_size}")


@dataclass
class ModelLibraryConfig:
    """A folder of built models, offered in the window's model dropdown.

    The alternative to declaring every model by hand under 'models': point this
    at what ``scripts/SSM/build_model.py`` wrote and every archive under it
    becomes selectable without editing any YAML. The two sources merge, with a
    hand-written 'models' entry winning on a name collision, so a model that
    needs its own camera or an LDDMM parametrisation can still be spelled out
    while the rest are discovered.
    """

    # Folder to scan. Searched recursively; null disables discovery.
    root: Optional[Path] = None
    # Which archives to offer. The default matches build_model.py's output.
    pattern: str = "*.npz"
    # Entry names come from the path relative to 'root', without the suffix:
    # ``baby/baby_pca.npz`` -> ``baby/pca``. The redundant ``<folder>_`` prefix
    # is dropped because the folder is already in the name; set this false to
    # keep the filename verbatim.
    strip_folder_prefix: bool = True
    # Viewing defaults for every discovered model. Discovery cannot know which
    # way a mesh faces, and the two domains in this project happen to agree.
    normalize: bool = False
    camera_axis: str = "+z"
    view_up: str = "+y"

    def discover(self) -> Dict[str, "ShapeModelSource"]:
        """``name -> ShapeModelSource`` for every archive under ``root``."""
        if self.root is None:
            return {}
        if not self.root.is_dir():
            raise ValueError(f"library.root is not a directory: {self.root}")

        found = {}
        for path in sorted(self.root.rglob(self.pattern)):
            relative = path.relative_to(self.root)
            stem = relative.stem
            parent = relative.parent.as_posix()
            if self.strip_folder_prefix and parent != ".":
                lead = relative.parent.name + "_"
                if stem.startswith(lead) and len(stem) > len(lead):
                    stem = stem[len(lead):]
            name = stem if parent == "." else f"{parent}/{stem}"
            found[name] = ShapeModelSource(
                path=path, normalize=self.normalize,
                camera_axis=self.camera_axis, view_up=self.view_up,
            )
        return found


@dataclass
class DropdownConfig:
    """The model picker in the top-left of the window."""

    # Rows drawn at once. The list scrolls past this with the wheel, so it caps
    # the height rather than the number of models.
    visible_rows: int = 14
    # Width of the closed bar and of the open list, in pixels.
    width: int = 340
    # Height of one row, in pixels.
    row_height: int = 22

    def __post_init__(self):
        if self.visible_rows < 1:
            raise ValueError(f"dropdown.visible_rows must be at least 1, got {self.visible_rows}")


@dataclass
class SampleShapesExport:
    """Where 's' writes the shape currently on screen.

    The export is a real mesh, not a picture: the sample's vertices on the
    model's own triangulation, carrying every scalar the window can colour by
    plus the displacement as a vector field, so it opens in ParaView with the
    same information the window had. The coefficients that produced it travel
    with it as field data, which is what makes a saved sample reproducible
    rather than merely viewable.
    """

    dir: Path = Path("outputs/shape_space")
    # Appended-binary + zlib. false writes ASCII, which is ~4x larger and only
    # worth it for diffing a small mesh by eye.
    binary: bool = True
    # Attach displacement / signed_displacement / total_std / displacement
    # vectors as point data. false writes bare geometry.
    include_fields: bool = True


@dataclass
class SampleShapesConfig:
    """Interactive explorer: one slider per mode, drag to walk the shape space.

    'model' names which model to open. It may be an entry of 'models' or one
    discovered under 'library.root'; either way the window's dropdown switches
    between all of them at runtime, so comparing two kernels is a click rather
    than an edit and a restart.
    """

    model: str
    models: Dict[str, ShapeModelSource] = field(default_factory=dict)
    library: ModelLibraryConfig = field(default_factory=ModelLibraryConfig)
    dropdown: DropdownConfig = field(default_factory=DropdownConfig)
    sliders: SlidersConfig = field(default_factory=SlidersConfig)
    view: SampleShapesView = field(default_factory=SampleShapesView)
    export: SampleShapesExport = field(default_factory=SampleShapesExport)
    # False renders one frame and exits instead of opening a window, which is
    # what makes the viewer usable headlessly - pair it with 'screenshot'.
    interactive: bool = True
    screenshot: Optional[Path] = None
    verbose: bool = True

    def __post_init__(self):
        # Discovered first, hand-written second: an explicit entry overrides a
        # discovered one of the same name rather than the other way round.
        merged = self.library.discover()
        merged.update(self.models)
        self.models = merged

        if not self.models:
            raise ValueError(
                "no shape models: declare at least one under 'models', or point "
                "'library.root' at a folder of built .npz models"
            )
        if self.model not in self.models:
            raise ValueError(
                f"model {self.model!r} is not available. "
                f"Available: {sorted(self.models)}"
            )

    def source(self) -> ShapeModelSource:
        """The selected model's entry."""
        return self.models[self.model]

    def model_names(self) -> List[str]:
        """Every selectable model, in the order the dropdown lists them."""
        return sorted(self.models)


@dataclass
class RigidConfig:
    iterations: int = 1                        # rigid align <-> landmark fit rounds
    landmark_fit_learning_rate: float = 0.01   # ShapeInstance.fit_landmarks step size
    allow_scale: bool = True                   # similarity (True) vs rigid-only (False)


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
    # src.nicp.correspondence._deduplicate). Off by default - several model
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

    ``fits`` (preferred) is the same language ``configs/ssm/nicp.yaml`` speaks -
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
# scripts/mean_shape.py
# --------------------------------------------------------------------------

@dataclass
class MeanShapeFields:
    """Which per-vertex fields to attach to the mean shape.

    All of them are properties of the *model*, not of any particular fit, so
    this is the reference picture of what the SSM can do and where it is
    confident.
    """

    # sqrt(sum_k lambda_k ||phi_k(v)||^2): the standard deviation of each
    # vertex's position under the model. The heatmap that says where the
    # training population actually varies.
    total_std: bool = True
    # 1-sigma displacement magnitude of each of the leading modes, one field
    # each, so a mode's spatial support is visible. 0 disables.
    mode_count: int = 5
    mode_vectors: bool = False   # the same modes as 3-vectors, for glyphs
    curvature: bool = True       # mean curvature of the mean surface
    landmark_distance: bool = True   # distance to the nearest landmark, and its id
    # One field per landmark, heatmap_<name>: heat released at that landmark and
    # allowed to diffuse over the surface for time t. Unlike a Euclidean radius
    # it follows the geometry, so it does not leak across a fold.
    landmark_heatmaps: bool = True
    # heatmap_landmarks: one field for the whole landmark set.
    combined_heatmap: bool = True
    #   max  element-wise maximum of the per-landmark maps, each normalised to
    #        peak 1, so every landmark reads exactly 1 at its own vertex
    #        whatever its neighbours are doing.
    #   sum  heat from a sum of Dirac deltas. Physically the field of all
    #        landmarks releasing at once, but overlapping bumps add, so a dense
    #        cluster outweighs an isolated landmark.
    combined_mode: str = "max"
    # Spread of the bump, as a length in model units: the diffusion time is
    # sigma^2, so this is directly comparable with the distance thresholds used
    # for matching. null derives it as ten times the mean edge length.
    heat_sigma: Optional[float] = None
    heat_normalize: bool = True      # rescale each heatmap to peak at 1

    COMBINED_MODES = ("max", "sum")

    def __post_init__(self):
        if self.combined_mode not in self.COMBINED_MODES:
            raise ValueError(
                f"fields.combined_mode must be one of {list(self.COMBINED_MODES)}, "
                f"got {self.combined_mode!r}"
            )


@dataclass
class MeanShapeOutputFiles:
    mean_shape: str = "mean_shape.vtp"
    landmarks_csv: str = "landmark_coordinates.csv"
    landmarks_vtp: str = "landmarks.vtp"


@dataclass
class MeanShapeOutputs:
    dir: Path
    files: MeanShapeOutputFiles = field(default_factory=MeanShapeOutputFiles)

    def path(self, key: str) -> Path:
        path = self.dir / getattr(self.files, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class MeanShapeConfig:
    # BabyFaceModel .mat statistical shape model, or the directory of a
    # Deformetrica atlas (see AtlasConfig) - the loader dispatches on which.
    shape_model_mat: Path
    outputs: MeanShapeOutputs
    model: ModelConfig = field(default_factory=ModelConfig)
    fields: MeanShapeFields = field(default_factory=MeanShapeFields)
    verbose: bool = True


# --------------------------------------------------------------------------
# scripts/nicp.py - the NICP step on its own
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


@dataclass
class ToModelSpace:
    """How the target mesh gets into model space, which is where the fit happens.

    Everything measured in model units - correspondence ``dist_thresh``, chamfer
    ``max_dist``, the prior weights - is meaningless until this is right. The
    model spans about +/-0.01 with an RMS radius of 5.6e-3 while the raw meshes
    span about +/-1, a factor of ~154.

    Two ways to get there, and at most one may be set:

    template_vtp   solve it. The template is vertex-wise corresponding with the
                   model (both 31027 points in the same order), so the
                   similarity transform is a closed-form Procrustes fit over
                   every vertex - no landmarks. This is the pipeline's step 0b,
                   direction to_model, and it makes the run self-contained.
    transform_npz  replay a solved one, e.g. the outputs/gmm_2_template.npz the
                   pipeline writes.

    Both null means mesh_vtp is already in model space.
    """

    template_vtp: Optional[Path] = None
    transform_npz: Optional[Path] = None
    inverse: bool = False           # transform_npz only: apply its inverse
    allow_scale: bool = True        # template_vtp only: similarity vs rigid-only
    save_transform: bool = True     # template_vtp only: write the solved transform out

    def __post_init__(self):
        if self.template_vtp is not None and self.transform_npz is not None:
            raise ValueError(
                "set at most one of inputs.to_model_space.template_vtp / .transform_npz"
            )


@dataclass
class NicpInputs:
    # BabyFaceModel .mat statistical shape model, or a Deformetrica atlas
    # directory (see AtlasConfig).
    shape_model_mat: Path
    mesh_vtp: Path                   # target surface to fit (the ultrasound scan)
    to_model_space: ToModelSpace = field(default_factory=ToModelSpace)
    # Optional starting point. init_alpha_npz reads an 'alpha' array (as written
    # by an earlier run); init_vtp projects a mesh onto the model basis instead.
    # Both unset starts from the mean shape.
    init_vtp: Optional[Path] = None
    init_alpha_npz: Optional[Path] = None

    def __post_init__(self):
        if self.init_vtp is not None and self.init_alpha_npz is not None:
            raise ValueError("set at most one of inputs.init_vtp / inputs.init_alpha_npz")


@dataclass
class NicpComputeConfig:
    device: str = "auto"             # auto | cpu | cuda
    # float32 halves the memory traffic and, on a consumer GPU, runs ~32x faster
    # than float64. The K x K system inside the closed-form solver is promoted to
    # float64 regardless, because at 163 x 163 that costs nothing.
    dtype: str = "float32"
    seed: Optional[int] = 0          # seeds the loss subsampling
    # Attach the model's triangulation to every instance. This changes matching:
    # normals then come from the surface and are consistently oriented, where a
    # point cloud gets them from a local SVD whose sign is arbitrary. False
    # treats the model as a point cloud, which is what the pipeline does.
    use_model_faces: bool = True


@dataclass
class NicpOutputFiles:
    # Per run, written under outputs.dir / run_dir. Results appear in both
    # frames: the fit happens in model space, but the scan and everything
    # downstream of it live in world space, so each result is also mapped back
    # through the inverse of the to_model_space transform.
    fit: str = "model_space/fit.vtp"
    fit_world: str = "world/fit.vtp"
    alpha: str = "alpha.npz"
    matching_field: str = "model_space/matches.vtp"
    iteration: str = "model_space/iterations/iter_{iteration:04d}.vtp"
    iteration_world: str = "world/iterations/iter_{iteration:04d}.vtp"
    # shared, written once under outputs.dir. The inputs are saved in both
    # frames because the fit happens in model space while the data lives in
    # world space, and comparing them is the fastest way to see a bad transform.
    world_mesh: str = "world/mesh.vtp"
    world_template: str = "world/template.vtp"
    model_mesh: str = "model_space/mesh.vtp"
    model_template: str = "model_space/template.vtp"
    model_init: str = "model_space/init_shape.vtp"
    model_mean: str = "model_space/mean.vtp"
    transform: str = "gmm_2_template.npz"        # written when solved from the template
    metrics: str = "metrics.csv"
    summary: str = "summary.json"


@dataclass
class NicpOutputs:
    dir: Path
    run_dir: str = "{run}"             # one subfolder per run; "" puts them all in dir
    files: NicpOutputFiles = field(default_factory=NicpOutputFiles)
    save_alpha: bool = True            # lets a later step start where this one stopped
    save_matching_field: bool = True   # skipped by correspondence-free runs
    save_history: bool = True          # per-iteration objective, into summary.json
    # Also write every result mapped back to the frame the inputs arrived in.
    # Has no effect when to_model_space is off, since the two frames coincide.
    save_world: bool = True
    # One mesh per iteration, so the fit can be played back. ~370 KB each, so a
    # 300-iteration gradient run is ~110 MB - raise the stride to thin it out.
    save_iterations: bool = False
    iteration_stride: int = 1

    def __post_init__(self):
        if self.iteration_stride < 1:
            raise ValueError(f"iteration_stride must be >= 1, got {self.iteration_stride}")

    def path(self, key: str, **fmt) -> Path:
        """A shared output path; parent directories are created."""
        path = self.dir / getattr(self.files, key).format(**fmt)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def run_path(self, key: str, run: str, **fmt) -> Path:
        """An output path belonging to one run; parent directories are created."""
        folder = self.dir / self.run_dir.format(run=run) if self.run_dir else self.dir
        path = folder / getattr(self.files, key).format(run=run, **fmt)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class NicpFitConfig:
    inputs: NicpInputs
    outputs: NicpOutputs
    model: ModelConfig = field(default_factory=ModelConfig)
    compute: NicpComputeConfig = field(default_factory=NicpComputeConfig)
    # Raw mappings: 'runs' entries are partial overrides of 'defaults', so they
    # are merged before being validated into NicpFitSpec.
    defaults: Dict[str, Any] = field(default_factory=dict)
    runs: List[Dict[str, Any]] = field(default_factory=list)
    verbose: bool = True


@dataclass
class SpectralConfig:
    path: Path                        # .npz holding L, M, eigenvalues, eigenvectors
    compute: bool = False             # False = load 'path', True = compute from the NICP result
    save: bool = True                 # write the operators to 'path' after computing
    source_key: Optional[str] = None  # result to build the operators on (default: laplacian.template_key)
    k_eig: int = 20
    downsampling: bool = False        # must stay False: the Laplacian has to keep every vertex
    test_spectrum: bool = True


@dataclass
class RegionConfig:
    landmark: str = "prn"   # landmark the deformation region is centred on
    radius: float = 0.3     # region radius in model units


@dataclass
class LaplacianConfig:
    # Keys address the result registry: "refined" (step 1), "gamma_<value>"
    # (the NICP sweep) or a NICP variant name.
    template_key: str = "gamma_80000"   # result used as the deformation template
    target_key: str = "gamma_10000"     # result the region is pulled towards
    lambda_reg: float = 1.0             # weight of the Laplacian (shape preserving) term
    region: RegionConfig = field(default_factory=RegionConfig)


@dataclass
class PipelineOutputFiles:
    mean: str = "mean.vtp"
    init_shape: str = "init_shape.vtp"
    surface: str = "surface.vtp"
    rigid_alignment: str = "rigid_alignment.vtp"
    refined_vertices: str = "refined_vertices.vtp"
    original_landmarks: str = "orig_langmarks.vtp"
    transformed_landmarks: str = "new_target_landmarks.vtp"
    matching_field: str = "matching_field.vtp"
    template_aligned: str = "template_aligned.vtp"
    gmm_2_template_transform: str = "gmm_2_template.npz"
    # One family for every fit, keyed by the fit's own name. A desugared gamma
    # sweep names its fits "gamma_5", so it still writes
    # nicp_refined_vertices_gamma_5.vtp - the filenames never moved, and no
    # gamma-aware code is needed to keep them.
    nicp_fit: str = "nicp_refined_vertices_{name}.vtp"
    nicp_fit_matching: str = "nicp_matching_field_{name}.vtp"
    nicp_fit_iteration: str = "nicp_refined_vertices_{name}_iter_{iteration:04d}.vtp"
    region_vertices: str = "n_region_verts.vtp"
    deformed_vertices: str = "deformed_vertices.vtp"


@dataclass
class PipelineOutputs:
    dir: Path
    files: PipelineOutputFiles = field(default_factory=PipelineOutputFiles)

    def path(self, key: str, **fmt) -> Path:
        return self.dir / getattr(self.files, key).format(**fmt)


@dataclass
class PipelineConfig:
    inputs: PipelineInputs
    outputs: PipelineOutputs
    steps: StepsConfig = field(default_factory=StepsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    gmm_2_template: Optional[GmmToTemplateConfig] = None
    rigid: RigidConfig = field(default_factory=RigidConfig)
    nicp: NicpConfig = field(default_factory=NicpConfig)
    spectral: SpectralConfig = field(default_factory=lambda: SpectralConfig(path=Path("spectral_data.npz")))
    laplacian: LaplacianConfig = field(default_factory=LaplacianConfig)
    verbose: bool = True


# --------------------------------------------------------------------------
# scripts/place_landmarks.py
# --------------------------------------------------------------------------

# The BabyFM's own 23 anatomical landmarks (outputs/mean_shape/landmark_coordinates.csv),
# used as the default checklist so a hand-placed set on another model - the
# fetus atlas - lines up name for name with the baby's. R/L suffixes are right/
# left as seen on the face (not the viewer's left/right).
DEFAULT_LANDMARK_NAMES: List[str] = [
    "exR", "enR", "n", "enL", "exL",     # eye corners + nasion
    "acR", "aR", "prn", "aL", "acL",     # nose
    "sn",                                 # subnasale
    "chR", "cphR", "ls", "cphL", "chL",  # mouth, upper lip
    "li", "sl", "pg",                     # mouth, lower lip + chin
    "tR", "oiR", "tL", "oiL",             # ears
]


@dataclass
class PlaceLandmarksConfig:
    """Interactive vertex picker: click the mean shape, name what you clicked.

    Built for the Deformetrica atlas (the fetus model), which - unlike the
    .mat - carries no landmark annotation of its own (see the module docstring
    of src.model.deformetrica). Writes name -> vertex-index JSON at
    landmarks_json, the format src.model.landmarks reads back and what
    model.atlas.landmarks in another config (mean_shape_fetus.yaml,
    sample_shapes.yaml) points at to give that atlas model the same
    landmark-driven fields the .mat model has built in.
    """

    # .mat statistical shape model, or a Deformetrica atlas directory - only
    # its mean shape is used here, so atlas options that shape the modes
    # (n_components, variance_threshold, ...) make no difference to this tool.
    shape_model: Path
    landmarks_json: Path
    model: ModelConfig = field(default_factory=ModelConfig)
    # The checklist walked while placing landmarks, in order. Names already in
    # landmarks_json that are not in this list are kept and appended after it,
    # rather than dropped, so re-opening a file never loses a name.
    landmark_names: List[str] = field(default_factory=lambda: list(DEFAULT_LANDMARK_NAMES))
    # The same slider panel sample_shapes.py opens the shape space with, so a
    # landmark can be placed on a deformed instance - useful where an
    # anatomical point is easier to spot away from the mean - rather than only
    # on alpha = 0. A landmark is still just a vertex *index*, so it stays
    # valid at every point in shape space once placed.
    sliders: SlidersConfig = field(default_factory=SlidersConfig)

    camera_axis: str = "+z"
    view_up: str = "+y"
    window_size: List[int] = field(default_factory=lambda: [1400, 900])
    theme: str = "light"

    # curvature shows the anatomy - nose tip, eye sockets, chin - that
    # landmarks sit on, which is what makes them findable by eye. none is a
    # flat surface colour, for a mesh where curvature reads noisy. Computed
    # once from the mean shape and held fixed as the sliders move - it is a
    # guide to where things are, not a live property of the current sample.
    color_by: str = "curvature"
    # Picked-vertex marker radius, in model units. null derives it as 0.6% of
    # the mesh's bounding radius, which reads at a consistent size whether the
    # model spans ~0.013 (the .mat) or ~70 mm (the fetus atlas).
    marker_radius: Optional[float] = None
    # vtkPointPicker tolerance: fraction of the render window's diagonal a
    # click may miss the nearest vertex by and still hit it.
    pick_tolerance: float = 0.01
    # Vertex size used when 'v' switches the mesh to a point cloud - a plain
    # surface can hide a landmark sitting just under a fold or occlude the far
    # side of the shape, where the raw vertices stay visible from anywhere.
    point_size: float = 4.0
    # Geodesic shooting resolution when 'l' toggles the sliders onto the LDDMM
    # parametrization (an atlas only - see src.GUIs.set_landmarks.shape_space).
    # 6 is converged to ~0.14 mm for the fetus atlas, same default as
    # ShapeModelSource.lddmm_time_points in sample_shapes.yaml.
    lddmm_time_points: int = 6

    verbose: bool = True

    COLOR_MODES = ("curvature", "none")
    AXES = ("+x", "-x", "+y", "-y", "+z", "-z")

    def __post_init__(self):
        if self.color_by not in self.COLOR_MODES:
            raise ValueError(
                f"color_by must be one of {list(self.COLOR_MODES)}, got {self.color_by!r}"
            )
        for name in ("camera_axis", "view_up"):
            value = getattr(self, name)
            if value not in self.AXES:
                raise ValueError(f"{name} must be one of {list(self.AXES)}, got {value!r}")
        if self.camera_axis[1] == self.view_up[1]:
            raise ValueError(
                f"camera_axis and view_up cannot share an axis "
                f"({self.camera_axis!r} / {self.view_up!r})"
            )
        if len(self.window_size) != 2:
            raise ValueError(f"window_size must be [width, height], got {self.window_size}")
        if not (0.0 < self.pick_tolerance < 0.5):
            raise ValueError(f"pick_tolerance must be in (0, 0.5), got {self.pick_tolerance}")
        if self.point_size <= 0:
            raise ValueError(f"point_size must be positive, got {self.point_size}")
        if self.lddmm_time_points < 2:
            raise ValueError(f"lddmm_time_points must be at least 2, got {self.lddmm_time_points}")


# --------------------------------------------------------------------------
# scripts/upsample_atlas.py
# --------------------------------------------------------------------------

@dataclass
class UpsampleAtlasConfig:
    """Resamples a Deformetrica atlas's template to roughly ``scale`` x its
    vertex count, and writes the result as a new, independent atlas directory.

    Only the template surface changes resolution. The control points and every
    subject's momenta are copied through unchanged, because they do not depend
    on how densely the carrier mesh is sampled: :func:`src.model.deformetrica.shoot`
    takes the deformation kernel and *any* set of points, so a denser template
    is shot by exactly the code path the original resolution already uses -
    this is not an approximation layered on top of the model, it is the same
    model evaluated at more points. Reusing that path is also what makes the
    output a normal atlas directory: point ``shape_model`` / ``shape_model_mat``
    at it in any other config and every loader treats it like any other.
    """

    source: Path            # the original atlas directory
    output_dir: Path        # where the new, denser atlas is written
    scale: float = 2.0      # target vertex-count multiplier
    # Shoots the whole population at the new resolution and PCAs it once,
    # right after writing, so the expensive part (see time_points below) is
    # already cached beside the new atlas - src.model.deformetrica caches by
    # kernel width and time_points, so this has to match whatever another
    # config's model.atlas.time_points will ask for to actually save that run
    # its ~90s-and-up shoot. False skips it: the first real use pays that cost
    # instead, exactly like a freshly-decomposed original atlas would.
    warm_cache: bool = True
    warm_cache_time_points: int = 11
    verbose: bool = True

    def __post_init__(self):
        if self.scale <= 1.0:
            raise ValueError(f"scale must be greater than 1.0, got {self.scale}")
        if self.warm_cache_time_points < 2:
            raise ValueError(
                f"warm_cache_time_points must be at least 2, got {self.warm_cache_time_points}"
            )
