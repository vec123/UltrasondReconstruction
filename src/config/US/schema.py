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
    lps_to_ras: bool = True  # Slicer CSV landmarks are in LPS space, meshes in RAS


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
    lps_to_ras: bool = False  # set True if the landmark CSVs need LPS -> RAS conversion


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
    lps_to_ras: bool = True  # Slicer fiducial CSVs export in LPS; flip x, y to match the nrrd's world frame
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

    # Directory to save target_w_distance.vtp / recon_w_distance.vtp into.
    # Left unset, saves next to target_path / recon_path respectively.
    output_dir: str = ""

    # align_to_template's saved-transforms dir for this subject (one .npz per
    # sweep/scan of that subject). When set, reported distances are divided
    # by the median similarity scale over those transforms, into the
    # original (pre-alignment) data's physical scale. Left unset (""),
    # distances stay in the meshes' own template units.
    transforms_dir: str = ""

    # Which single transform within transforms_dir (e.g. one sweep's id)
    # defines the saved geometry's back-transform. When set, the FULL
    # inverse (rotation+translation+scale) of that one transform is applied
    # to the saved meshes, placing them back in that subject's original
    # frame - not just rescaled. Left unset (""), geometry is only rescaled
    # (if transforms_dir is set) or left untouched (if it isn't).
    subject_id: str = ""
