"""Config dataclasses for every US ultrasound preprocessing pipeline stage.

One dataclass per stage (named ``<StageName>Config``), plus the handful of
nested config fragments a stage's own Config embeds. Each
``scripts/US/pipeline/stages/*.py`` module imports its class from here
(aliased back to the local name ``Config``) instead of defining it inline, so
every stage's configuration surface lives in one place.
"""

import dataclasses
from typing import List, Optional, Tuple, Union


@dataclasses.dataclass
class FlatRegionConfig:
    enabled: bool = False
    n_rings: int = 5
    thresh: float = 0.1
    max_area: float = 0.05  # drop flat patches covering > this fraction of total area


@dataclasses.dataclass
class VolumeConfig:
    denoise: Optional[str] = None  # "median" | "curvature" | None
    denoise_radius: int = 5
    normalize: bool = False
    clip_percentiles: Tuple[float, float] = (1.0, 99.0)
    norm_out_max: float = 255.0
    crop: bool = False
    crop_threshold: int = 0
    crop_margin: float = 0
    # 4D (x, y, z, t) volumes: "skip" = skip with a warning; "mean" | "median" = reduce
    # over time to one 3D scan; int = keep only that time frame (see reduce_4d_volume)
    four_d: Union[str, int] = "skip"


@dataclasses.dataclass
class NrrdToVtpConfig:
    input_dir: str = ""
    output_dir: str = ""
    cache_dir: str = ""

    contour_value: float = 70.0  # level relative to norm_out_max after normalization; ignored if auto_contour
    auto_contour: bool = False  # pick contour_value per volume via Otsu's method (consistent across NRRD sources)
    reduction_factor: float = 0.1  # quadric decimation factor (0 = none, 0.5 = 50% reduction)

    min_bridge_width: float = 1.0  # bridges narrower than this do not connect regions; 0 disables
    min_bridge_relative: bool = False  # interpret min_bridge_width as a multiple of the mean edge length

    compute_curvedness: bool = False
    curvedness_n_rings: int = 5
    curvedness_scale_invariant: bool = False
    operators: bool = False  # compute Laplace-Beltrami operators (k_eig eigenpairs)
    k_eig: int = 200

    flat: FlatRegionConfig = dataclasses.field(default_factory=FlatRegionConfig)
    volume: VolumeConfig = dataclasses.field(default_factory=VolumeConfig)


@dataclasses.dataclass
class EchotextureConfig:
    vtp_dir: str = ""  # nrrd_to_vtp's output_dir (pre-alignment meshes)
    nrrd_dir: str = ""  # the SAME source directory as nrrd_to_vtp.input_dir
    output_dir: str = ""

    # Must mirror nrrd_to_vtp's own volume.crop* settings - see the stage module docstring.
    crop: bool = False
    crop_threshold: int = 0
    crop_margin: float = 0
    four_d: Union[str, int] = "skip"  # must mirror nrrd_to_vtp's volume.four_d, else a 4D scan is sampled in another volume

    # What signal to measure; independent of the coordinate frame above.
    denoise: Optional[str] = None  # "median" | "curvature" | None
    denoise_radius: int = 5
    normalize: bool = False
    clip_percentiles: Tuple[float, float] = (1.0, 99.0)
    norm_out_max: float = 255.0

    neighborhood_radius_mm: float = 1.5  # half-width of the sampled voxel neighborhood
    interp_order: int = 1  # 0 = nearest voxel, 1 = trilinear

    # Integration-scale radius (mm) for the structure-tensor orientation-
    # coherence fields (echo_linearity/planarity/sphericity/energy) - see
    # the stage module docstring and src.vtk.echotexture.sample_structure_tensor.
    structure_tensor_radius_mm: float = 1.5

    # Radius (mm, on the mesh's own un-aligned surface) each field's *_variability
    # companion is measured over - see the stage module docstring. Must run before any
    # alignment stage rescales the mesh, or this stops meaning a physical distance.
    variability_radius_mm: float = 10.0
    # Caps the neighbor search to each vertex's nearest N, so a dense mesh
    # doesn't turn variability_radius_mm into an unbounded (memory-blowing)
    # neighbor count - see compute_field_variability's docstring.
    variability_max_neighbors: int = 64

    field_prefix: str = "echo_"


@dataclasses.dataclass
class AlignToReferenceConfig:
    landmarks_csv_dir: str = ""
    vtp_shapes_dir: str = ""

    output_dir: str = ""
    raw_landmarks_vtp_dir: str = ""

    # The following default to subdirectories of output_dir when left blank;
    # they're real fields (rather than being hardcoded inside run()) so other
    # stages can reference them, e.g. `align_to_reference.aligned_shapes_vtp_dir`.
    aligned_landmarks_dir: str = ""  # defaults to <output_dir>/aligned_landmarks
    aligned_shapes_vtp_dir: str = ""  # defaults to <output_dir>/aligned_shapes_vtp
    transforms_dir: str = ""  # defaults to <output_dir>/transforms
    previous_transforms_dir: Optional[str] = None  # first alignment stage: no previous

    reference_subject_id: Optional[str] = None  # defaults to the first common subject id
    with_scale: bool = True
    landmarks_frame: str = "RAS"  # frame landmarks_csv_dir's CSVs are written in: "RAS" (flipped to the meshes' LPS) or "LPS"


@dataclasses.dataclass
class AlignToTemplateConfig:
    template_vtp: str = ""

    vtp_shapes_dir: str = ""
    landmarks_csv_dir: str = ""

    output_dir: str = ""

    # The following default to subdirectories of output_dir when left blank;
    # they're real fields (rather than being hardcoded inside run()) so other
    # stages can reference them, e.g. `align_to_template.out_shapes_vtp_dir`.
    out_shapes_vtp_dir: str = ""  # defaults to <output_dir>/shapes_vtp
    out_landmarks_csv_dir: str = ""  # defaults to <output_dir>/landmarks_csv
    out_landmarks_vtp_dir: str = ""  # defaults to <output_dir>/landmarks_vtp
    transforms_dir: str = ""  # defaults to <output_dir>/transforms
    previous_transforms_dir: Optional[str] = None  # align_to_reference's transforms_dir

    heatmap_prefix: str = "heatmap_"
    with_scale: bool = True
    landmarks_frame: str = "LPS"  # frame landmarks_csv_dir's CSVs are written in: "LPS" (as-is) or "RAS" (flipped to LPS)


@dataclasses.dataclass
class RoughnessFilterConfig:
    input_dir: str = ""
    output_dir: str = ""
    unfiltered_dir: str = ""  # if set, saves each mesh with the roughness field attached, before thresholding

    n_rings: int = 5
    scale_invariant: bool = False
    method: str = "quadric"  # "plane" | "quadric"
    threshold: float = 0.99  # keep vertices with roughness below this fraction of the field's range
    neighbors: str = "knn"  # "knn" | "rings"
    max_neighbors: int = 32

    keep_largest_component: bool = True  # re-extract the largest component after thresholding
    min_bridge_width: float = 1.0  # bridges narrower than this do not connect regions after thresholding; 0 disables
    min_bridge_relative: bool = False  # interpret min_bridge_width as a multiple of the mean edge length

    # If set (e.g. align_to_template's transforms_dir), converts min_bridge_width
    # from the original nrrd/ply mm space into each subject's own aligned/scaled
    # space using that subject's saved similarity-transform scale. Leave blank
    # if input_dir is already in the original, unscaled mm space.
    transforms_dir: str = ""


@dataclasses.dataclass
class LandmarkBboxCutConfig:
    shapes_dir: str = ""
    landmarks_dir: str = ""
    output_dir: str = ""

    # align_to_template's transforms_dir, used to convert the offsets below
    # (defined in the original nrrd/ply space) into each subject's own
    # scaled template-space units. Leave blank if shapes_dir is already in
    # the original, unscaled mm space.
    transforms_dir: str = ""

    # Padding added to each face of the landmarks' bounding box (mm, original
    # nrrd/ply space). x = Right(+)/Left(-), y = Anterior(+)/Posterior(-),
    # z = Superior(+, above eyes)/Inferior(-, toward chin).
    x_min_offset_mm: float = 5.0
    x_max_offset_mm: float = 5.0
    y_min_offset_mm: float = 5.0
    y_max_offset_mm: float = 5.0
    z_min_offset_mm: float = 25.0  # inferior: generous, keeps more of the chin
    z_max_offset_mm: float = 5.0   # superior: strict, stays close above the eyes

    keep_largest_component: bool = True  # re-extract the largest component after the box cut
    min_bridge_width: float = 1.0  # bridges narrower than this do not connect regions after the cut; 0 disables
    min_bridge_relative: bool = False  # interpret min_bridge_width as a multiple of the mean edge length


@dataclasses.dataclass
class LandmarkDistanceCutConfig:
    shapes_dir: str = ""
    landmarks_dir: str = ""
    output_dir: str = ""

    # align_to_template's transforms_dir, used to convert max_distance_mm
    # (defined in the original nrrd/ply space) into each subject's own
    # scaled template-space units. Leave blank if shapes_dir is already in
    # the original, unscaled mm space.
    transforms_dir: str = ""

    max_distance_mm: float = 10.0  # keep only points within this distance (mm, original nrrd/ply space) of a landmark

    keep_largest_component: bool = True  # re-extract the largest component after the distance cut
    min_bridge_width: float = 1.0  # bridges narrower than this do not connect regions after the cut; 0 disables
    min_bridge_relative: bool = False  # interpret min_bridge_width as a multiple of the mean edge length


@dataclasses.dataclass
class ScanConsistencyConfig:
    input_aligned_dir: str = ""

    output_consistency_dir: str = ""  # meshes with the 'consistency_score' field added
    output_denoised_dir: str = ""  # thresholded/denoised meshes
    output_merged_dir: str = ""  # single .vtp combining all denoised meshes

    transforms_dir: str = ""  # align_to_template's transforms_dir, used to scale sigma_mm

    sigma_mm: float = 2.0  # spatial tolerance in mm; points within sigma_mm score highly

    apply_denoise_threshold: bool = True
    # A single value denoises/merges once, saving "merged_high_confidence_scan.vtp"
    # (unchanged behavior). A list sweeps every value instead - one denoise +
    # merge pass per threshold - saving "merged_high_confidence_scan_threshold_
    # {value}.vtp" per pass into the same output_merged_dir (and, correspondingly,
    # a "threshold_{value}/" subdirectory per pass under output_denoised_dir), so
    # every threshold's result can be compared side by side.
    min_consistency_threshold: Union[float, List[float]] = 0.55
    denoise_min_bridge_width: float = 0.8
    denoise_min_bridge_relative: bool = True


@dataclasses.dataclass
class ScanFusionConfig:
    input_aligned_dir: str = ""  # per-subject aligned meshes to fuse (e.g. align_to_template's own output - scan_consistency is not required)
    reference_vtp: str = ""  # mesh providing the output topology (e.g. the align_to_template template)
    output_dir: str = ""

    transforms_dir: str = ""  # align_to_template's transforms_dir, used to scale sigma_mm

    sigma_mm: float = 2.0  # spatial tolerance in mm; points within sigma_mm pull a reference vertex toward them
    min_support: float = 2.0  # reference vertices with total kernel weight below this are dropped (too little cross-subject agreement)
    normal_power: float = 2.0  # down-weight contributors whose normal disagrees with the reference vertex's, by cos(angle)**normal_power; 0 disables (distance-only weighting)

    surface_fit_n_rings: int = 5  # neighborhood size for the post-fusion "roughness" field
    surface_fit_method: str = "quadric"  # "plane" | "quadric" - see compute_vertex_roughness
    compute_roughness: bool = True  # set False to skip the post-fusion "roughness" field entirely
    # Post-fusion per-vertex "normal" field. On by default because it is cheap
    # (one pass over the faces) and because resnet_lddmm's
    # data_name: field_weighted_point2plane_chamfer REQUIRES it on its target and
    # refuses to build without it. Must be computed here rather than upstream:
    # min_support drops vertices and fusion moves the survivors, so only the
    # final mesh's own faces describe the surface the normals belong to.
    compute_normals: bool = True
    normal_weighting: str = "unit"  # "unit" | "area" - see compute_vertex_normals


@dataclasses.dataclass
class SmoothingFilterConfig:
    input_dir: str = ""
    output_dir: str = ""

    iterations: int = 30  # higher = smoother
    pass_band: float = 0.1  # lower = smoother/blurrier, higher = keeps finer detail (try 0.05-0.2)
    boundary_smoothing: bool = True
    feature_edge_smoothing: bool = False


@dataclasses.dataclass
class LandmarkGeodesicsConfig:
    shapes_dir: str = ""
    landmarks_dir: str = ""
    output_dir: str = ""

    # Gaussian width of the heatmaps, in mesh units of geodesic distance along the surface.
    heatmap_sigma: float = 0.0014


@dataclasses.dataclass
class PointClassifyConfig:
    input_dir: str = ""  # a directory to glob (*.vtp/*.ply), OR a single mesh file path
    output_dir: str = ""

    # Explicit feature field names to cluster on; if left empty, every point
    # field whose name starts with feature_prefix (minus exclude_fields) on
    # the FIRST mesh is used, and every other mesh is required to carry the
    # same set.
    feature_fields: List[str] = dataclasses.field(default_factory=list)
    feature_prefix: str = "echo_"
    exclude_fields: List[str] = dataclasses.field(default_factory=lambda: ["echo_in_bounds"])

    # Named after the ~5 tissue types called out when this stage was scoped
    # (face, placenta, hand/limb, noise, other) - a starting guess, not a
    # constraint; there's no requirement the data actually separates into
    # exactly this many groups.
    n_clusters: int = 5
    algorithm: str = "minibatch_kmeans"  # "minibatch_kmeans" | "kmeans"
    random_state: int = 0
    n_init: int = 10
    batch_size: int = 10_000  # only used by "minibatch_kmeans"

    class_field_name: str = "tissue_cluster"


@dataclasses.dataclass
class NrrdToWorldVtpConfig:
    input_dir: str = ""
    output_dir: str = ""

    contour_value: float = 1.0  # label index/threshold for surface extraction
    reduction_factor: float = 0.3  # quadric decimation factor (0 = none, 0.5 = 50% reduction)


@dataclasses.dataclass
class TransformConfig:
    manifest_dir: Optional[str] = None  # a stage's transforms_dir, e.g. align_to_template's
    subject_id: Optional[str] = None
    apply_inverse: bool = False  # map vtp_path's points back to the manifest's source frame


@dataclasses.dataclass
class LandmarkAlignConfig:
    landmark_path: str = ""  # subject's own landmark CSV (Slicer fiducial export), in the nrrd's world frame
    # VTP carrying `heatmap_{label}` point fields (e.g. Templates/baby_mean.vtp,
    # Templates/fetus_mean.vtp) - the argmax vertex of each field gives that
    # label's vertex index into vtp_path's own point ordering, valid whenever
    # vtp_path preserves this template mesh's topology (as NICP-fitted
    # reconstructions do).
    template_vtp: str = ""
    heatmap_prefix: str = "heatmap_"
    landmarks_frame: str = "RAS"  # frame landmark_path is written in: "RAS" (flipped to the nrrd's LPS world frame) or "LPS"
    with_scale: bool = True


@dataclasses.dataclass
class SimilarityTransformConfig:
    """Rigidly (+ scale) moves source_vtp into landmark_path's frame, via
    shared named landmarks - see src.landmarks.vtk_landmarks.fit_landmark_alignment.

    Typical use: a standalone template (e.g. Templates/baby_mean.vtp, which
    carries its own heatmap_{label} point fields) needs to land in the same
    coordinate frame as an align_to_template output - e.g. before pairing it
    with a fused/consensus scan built in that frame as resnet_lddmm_train's
    source/target pair.
    """

    source_vtp: str = ""     # mesh to move; must carry its own heatmap_{label} point fields
    landmark_path: str = ""  # target frame's own named landmark CSV (x, y, z, label columns)
    output_path: str = ""    # source_vtp, saved after being moved into landmark_path's frame

    heatmap_prefix: str = "heatmap_"
    # Frame landmark_path is written in. align_to_template's out_landmarks_csv_dir
    # CSVs are already in the meshes' frame ("LPS", used as-is) - only set "RAS"
    # for a raw Slicer export written in RAS.
    landmarks_frame: str = "LPS"
    with_scale: bool = True


@dataclasses.dataclass
class OverlayLayer:
    vtp_path: str = ""
    line_color: tuple = (0, 255, 0)  # RGB
    thickness: Optional[float] = None  # None: use the stage's own `thickness`
    name: Optional[str] = None  # segment name in color_table_path; defaults to vtp_path's basename

    transform: TransformConfig = dataclasses.field(default_factory=TransformConfig)
    landmark_align: LandmarkAlignConfig = dataclasses.field(default_factory=LandmarkAlignConfig)


@dataclasses.dataclass
class VtpToNrrdOverlayConfig:
    nrrd_path: str = ""
    output_path: str = ""

    thickness: float = 1.0  # default line thickness / sphere radius, in mm; a layer may override

    layers: List[OverlayLayer] = dataclasses.field(default_factory=list)

    # Optional: one binary label-map NRRD + matching 2-entry color table (.ctbl)
    # per layer, written into this directory - see the stage module docstring. Left
    # unset, only output_path (the fixed, flattened RGB volume) is written.
    label_maps_dir: str = ""


@dataclasses.dataclass
class ReconTargetDistanceConfig:
    target_path: str = ""
    recon_path: str = ""
    landmarks_path: str = ""
    landmarks_frame: str = "LPS"  # frame landmarks_path is written in: "LPS" (as-is) or "RAS" (flipped to LPS)

    # Distance thresholds for the `fraction_within` block of each direction's
    # stats - the share of points closer than each one, in the SAME units as
    # `units` (so these are mm only while landmarks_path is a mm-frame CSV).
    # Empty, the block is omitted.
    coverage_thresholds: List[float] = dataclasses.field(
        default_factory=lambda: [1.0, 2.0, 3.0]
    )

    # Label for the frame `landmarks_path` is in, used in the printed lines,
    # the plot axis and the JSON outputs. Left unset, "landmark frame units".
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
class DistanceDistributionSourceConfig:
    """One reconstruction whose distance distribution is drawn as a line."""

    # Legend text, e.g. "LDDMM from baby mean". Avoid ' + ' - in a pipeline YAML
    # that is the path-concatenation operator (see scripts/pipeline/config.py)
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
