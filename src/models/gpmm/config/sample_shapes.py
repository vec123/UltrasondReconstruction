"""Config for scripts/GUIs/sample_shapes.py: the shape model library, sliders, view and export."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .model import AtlasConfig


# --------------------------------------------------------------------------
# scripts/GUIs/sample_shapes.py
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
    at what the ssm_build stage wrote and every archive under it
    becomes selectable without editing any YAML. The two sources merge, with a
    hand-written 'models' entry winning on a name collision, so a model that
    needs its own camera or an LDDMM parametrisation can still be spelled out
    while the rest are discovered.
    """

    # Folder to scan. Searched recursively; null disables discovery.
    root: Optional[Path] = None
    # Which archives to offer. The default matches the ssm_build stage's output.
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
