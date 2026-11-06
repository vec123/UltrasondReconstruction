"""Config dataclasses of the evaluation stages.

``src/evaluation/stages/recon_target_distance.py`` and
``distance_distribution_overlay.py`` import their class from here (aliased
back to the local name ``Config``).
"""

import dataclasses
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class ReconTargetDistanceConfig:
    target_path: str = ""
    recon_path: str = ""

    # Or: a model stage's output folder, which gives recon_path and target_path
    # (then leave those two unset). It is read through a pair layout
    # (src/evaluation/loader/layouts.py): "auto" finds the layout from the
    # folder's marker file, e.g. "result" for a folder with result.json.
    # layout_kwargs selects in the folder, e.g. {result: map_1x} for a GPMM run.
    result_dir: str = ""
    layout: str = "auto"
    layout_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # The frame the distances are computed in:
    #   "landmarks"  a similarity fitted from the recon's heatmap landmarks onto
    #                landmarks_path (needs landmarks_path)
    #   "scan"       the recon->scan transform of result_dir: the scan, in mm
    #   "recon"      the frame recon_path and target_path are written in
    frame: str = "landmarks"

    # The labelled landmarks of the target's scan (CSV). Unset, and with a
    # frame other than "landmarks", no landmark distances are computed.
    landmarks_path: str = ""
    landmarks_frame: str = "LPS"  # frame landmarks_path is written in: "LPS" (as-is) or "RAS" (flipped to LPS)

    # Distance thresholds for the `fraction_within` block of each direction's
    # stats - the share of points closer than each one, in the SAME units as
    # `units` (so these are mm only while landmarks_path is a mm-frame CSV).
    # Empty, the block is omitted.
    coverage_thresholds: List[float] = dataclasses.field(
        default_factory=lambda: [1.0, 2.0, 3.0]
    )

    # Label for the units of `frame`, used in the printed lines, the plot axis
    # and the JSON outputs. Left unset, "landmark frame units", "mm" or
    # "model frame units".
    units: str = ""

    # 3x3 metric matrix M of the landmark Mahalanobis distance
    # sqrt(d^T M d), d = (recon landmark - CSV landmark) in landmark-frame
    # coordinates. Left unset, the identity - i.e. plain Euclidean distance.
    # Pass the inverse of a landmark covariance here to weight the axes.
    landmark_metric: List[List[float]] = dataclasses.field(default_factory=list)

    # Optional: a resnet_lddmm_train run's dumped config.json. When set, the
    # target landmarks that run guided on (its `landmarks` folder, filtered by
    # loss.landmark_kwargs and averaged per label) are carried into the
    # landmark frame and saved as lddmm_landmarks.vtp. They are in that run's
    # world frame, so only set this when recon_path / target_path are too
    # (e.g. its apply's world/deformed.vtp against the fused or merged scan).
    lddmm_train_config: str = ""

    # Directory to save target_w_distance.vtp / recon_w_distance.vtp /
    # distance_summary.json / landmark_distances.json into. Left unset, saves
    # next to target_path / recon_path respectively (the .json alongside
    # recon_path).
    output_dir: str = ""


@dataclasses.dataclass
class CorrespondenceFieldConfig:
    # A model stage's output folder of one case, read through a pair layout
    # (src/evaluation/loader/layouts.py): "auto" finds the layout from the
    # folder's marker file. layout_kwargs selects in the folder, e.g.
    # {fit: map_1x} for direct_ssm_fit.
    case_dir: str = ""
    layout: str = "auto"
    layout_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # The correspondence method (src/shared/correspondence) and its settings.
    method: str = "constrained"
    kwargs: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"max_dist": 2.0, "normal_thresh_deg": 60.0, "one_to_one": True}
    )

    # The frame of the distance settings: "surface" (the scan, in mm) or
    # "recon" (the reconstruction's own frame).
    distance_frame: str = "surface"

    # Left unset, <case_dir>/<a name that spells out the method and every one
    # of its settings>, e.g. correspondence_constrained_k-5_max_dist-2surface_...
    output_dir: str = ""


@dataclasses.dataclass
class CorrespondenceViewsConfig:
    # The folder scripts/SSM/correspondence_field.py wrote: the two
    # *_w_correspondence.vtp files and correspondence_summary.json.
    correspondence_dir: str = ""

    # The fused scan the target points were taken from: a mesh in mm, with
    # heatmap landmark fields (they set the cameras). Left unset, it is read
    # from original_scans_path by the case id of the summary.
    surface_path: str = ""

    # The cohort's original_scans.json: {case id: fused scan path}.
    original_scans_path: str = ""

    # Recon frame -> scan frame (mm): the cohort's similarity.json.
    similarity_path: str = ""

    # Left unset: <correspondence_dir>/correspondence_field.png.
    output_path: str = ""

    panel_px: int = 420

    # Largest gap, in mm, between a target point and the surface vertex it is.
    surface_tolerance: float = 1e-3


@dataclasses.dataclass
class CorrespondenceOverviewConfig:
    # A folder with one subfolder per case, each with the files of
    # correspondence_views' correspondence_dir (target_w_correspondence.vtp and
    # correspondence_summary.json).
    cases_dir: str = ""

    # As in CorrespondenceViewsConfig.
    original_scans_path: str = ""
    similarity_path: str = ""

    # The reference mesh the flow deforms (the template, in the recon frame),
    # drawn as the first tile, "template", with the same field: the
    # correspondence_id is the index of its own vertex. Left unset, no tile.
    reference_path: str = ""

    # The template deformed with the mean of all codes (the apply option
    # mean_code), drawn as the next tile, "mean_recon". Left unset, no tile.
    mean_recon_path: str = ""

    # The mesh with the landmark field, in the vertex order of the recon, and
    # the name of that field. Left unset, the field "heatmap_landmarks" of each
    # case's recon file.
    landmark_path: str = ""
    landmark_field: str = "heatmap_landmarks"

    # The field the tiles show: "correspondence_id", or "landmark_from_recon"
    # (the recon's heatmap_landmarks carried to the scan through the correspondence).
    field: str = "correspondence_id"

    # Left unset: <cases_dir>/correspondence_overview.png.
    output_path: str = ""

    columns: int = 17
    tile_px: int = 200
    surface_tolerance: float = 1e-3


@dataclasses.dataclass
class DistanceDistributionSourceConfig:
    """One reconstruction whose distance distribution is drawn as a line."""

    # Legend text, e.g. "LDDMM from baby mean". Avoid ' + ' - in a pipeline YAML
    # that is the path-concatenation operator (see src/shared/core/config/variables.py)
    # and the surrounding spaces would be eaten.
    label: str = ""

    # The `output_dir` of the recon_target_distance run that produced this
    # reconstruction's distances - the directory holding recon_w_distance.vtp
    # and target_w_distance.vtp.
    eval_dir: str = ""

    # Explicit overrides, for meshes that don't sit in a recon_target_distance
    # output_dir. Left unset, both are derived from eval_dir.
    recon_path: str = ""
    target_path: str = ""

    color: str = ""  # hex override; left unset, takes the next palette slot


@dataclasses.dataclass
class DistanceDistributionOverlayConfig:
    """Overlays several runs' point-distance histograms as lines."""

    sources: List[DistanceDistributionSourceConfig] = dataclasses.field(default_factory=list)

    # Directory the two .png files and the summary .json are written into.
    output_dir: str = ""
    recon_to_target_plot_name: str = "recon_to_target_distribution.png"
    target_to_recon_plot_name: str = "target_to_recon_distribution.png"
    summary_name: str = "distance_distribution_overlay.json"

    bins: int = 100

    # Upper percentile of the pooled distances the x-axis is cut at, so a
    # handful of far outliers don't squash the bulk of the distribution into
    # the first few bins. 100 draws the full range. Points beyond the cut are
    # still counted (in the last bin), never dropped.
    clip_percentile: float = 99.5

    # "fraction" (share of that mesh's points, comparable across meshes of
    # different vertex counts), "density" (integrates to 1), or "count" (raw).
    normalize: str = "fraction"

    # Log y-axis. Distance distributions are heavy-tailed - the near-zero spike
    # can flatten the whole tail - and a log y separates the curves out there.
    log_y: bool = False

    # Distance-over-percentage curves: one figure per direction carrying every
    # source's curve, plus the sampled curves themselves as JSON.
    recon_to_target_coverage_plot_name: str = "recon_to_target_coverage.png"
    target_to_recon_coverage_plot_name: str = "target_to_recon_coverage.png"
    coverage_name: str = "distance_coverage_overlay.json"

    # Marked on the coverage figures and reported per source, in `units`.
    coverage_thresholds: List[float] = dataclasses.field(
        default_factory=lambda: [1.0, 2.0, 3.0]
    )

    # Per-landmark comparison, read from the landmark_distances.json
    # recon_target_distance writes into each source's eval_dir. Sources
    # without that file are left out of the figure and listed in the JSON.
    landmark_plot_name: str = "landmark_distances.png"
    landmark_summary_name: str = "landmark_distance_overlay.json"

    units: str = "landmark frame units"
    recon_to_target_title: str = "Reconstruction to target"
    target_to_recon_title: str = "Target to reconstruction"


@dataclasses.dataclass
class CompareConfig:
    """Several models, compared over the cases that all of them have."""

    # Model name -> the pattern of its evaluation folder (the output_dir of
    # recon_target_distance) of one case, e.g. ${OUT}/fetus_ssm/${SESSION}/eval.
    # The variables without a value are the case variables: the stage finds
    # their values on the disk.
    models: Dict[str, str] = dataclasses.field(default_factory=dict)

    # Directory the table, the summary and the plots are written into.
    output_dir: str = ""

    # The units of the evaluation folders. Left unset, read from them; they
    # must all agree.
    units: str = ""

    coverage_thresholds: List[float] = dataclasses.field(
        default_factory=lambda: [1.0, 2.0, 3.0]
    )

    # The distribution plots, as in DistanceDistributionOverlayConfig.
    bins: int = 100
    clip_percentile: float = 99.5
    normalize: str = "fraction"
    log_y: bool = False


@dataclasses.dataclass
class ViewColumnConfig:
    """One column of render_views: one mesh per case."""

    title: str = ""
    # The mesh of a case, as a pattern; the variables without a value are the
    # case variables. All meshes of a case must be in one frame.
    mesh: str = ""
    # The rows (views) this column is drawn in; empty: every view of the stage.
    views: List[str] = dataclasses.field(default_factory=list)
    # Draw the camera landmarks on this panel.
    landmarks: bool = False
    # This mesh frames the cameras (with the other framing meshes of the case).
    camera: bool = True
    # The plain colour of the mesh: "mesh" (a model, skin) or "scan" (blue).
    tint: str = "mesh"


@dataclasses.dataclass
class ViewScansConfig:
    """All scans of a case, frontal, above the result rows (wrapped over as many rows as needed)."""

    # Each scan, as a pattern with the case variables and one more (e.g. ${SCAN}).
    mesh: str = ""
    # Its landmarks (CSV with label, x, y, z), the same pattern; empty: none.
    landmarks: str = ""


@dataclasses.dataclass
class ViewColorConfig:
    """Colour the column meshes by a point field, on a sequential ramp from 0 to `max`."""

    field: str = ""
    max: float = 3.0


@dataclasses.dataclass
class RenderViewsConfig:
    columns: List[ViewColumnConfig] = dataclasses.field(default_factory=list)
    scans: Optional[ViewScansConfig] = None

    # The landmarks that set the cameras of a case (.vtp with a 'label' array,
    # or CSV), as a pattern; then the cases are those that have them. Empty:
    # the heatmap fields of the case's first mesh, and the cases of all columns.
    landmarks: str = ""

    views: List[str] = dataclasses.field(default_factory=lambda: ["frontal", "left", "right"])
    # per_case: one PNG per case, one row per view (and the scan rows above).
    # grid: one PNG views.png, one row per case, in the first view only.
    layout: str = "per_case"
    color: Optional[ViewColorConfig] = None
    panel_px: int = 400
    overwrite: bool = True

    # The name of the one case when every case variable has a value (a run of
    # one case, e.g. in a pipeline: ${CASE_KEY}); empty: "case".
    case_name: str = ""

    output_dir: str = ""
