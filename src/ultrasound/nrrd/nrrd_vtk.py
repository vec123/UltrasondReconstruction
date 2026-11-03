import SimpleITK as sitk
import vtk
import numpy as np
from vtk.util import numpy_support


def preprocess_volume(sitk_image, denoise="median", radius=1,
                      normalize=True, clip_percentiles=(1.0, 99.0), out_max=255.0):
    """
    Speckle-reduce and (optionally) intensity-normalize an ultrasound volume
    before level-set extraction.

    Denoising and normalization are independent:
      - denoise ("median"/"curvature"/None): edge-preserving speckle reduction.
      - normalize (bool): percentile-clip and rescale intensities to [0, out_max]
        so a contour value becomes a *relative* level comparable across scans.
        Set normalize=False to keep the original intensity scale (so an absolute
        contour value keeps its old meaning) while still denoising.

    Returns a float32 numpy array in (z, y, x) order (same layout as
    sitk.GetArrayFromImage), so the caller can build vtkImageData unchanged.
    """
    # 1. Edge-preserving speckle reduction in image space (spacing-aware).
    if denoise == "median":
        r = int(radius)
        sitk_image = sitk.Median(sitk_image, [r, r, r])
    elif denoise == "curvature":
        # Anisotropic (curvature) diffusion: smooths speckle while preserving edges.
        sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
        cflow = sitk.CurvatureFlowImageFilter()
        cflow.SetTimeStep(0.0625)
        cflow.SetNumberOfIterations(5)
        sitk_image = cflow.Execute(sitk_image)
    elif denoise not in (None, "none"):
        raise ValueError(f"Unknown denoise mode: {denoise!r}")

    arr = sitk.GetArrayFromImage(sitk_image).astype(np.float32)

    # 2. Optional percentile clip (robust to bright specular outliers) + rescale.
    if normalize:
        lo, hi = np.percentile(arr, clip_percentiles)
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo + 1e-8) * out_max
    return arr


def crop_volume(sitk_image, threshold=0, margin=0, verbose=True):
    """
    Crop an ultrasound volume to the bounding box of its content, then trim an
    extra border off every side.

    This is the 3D analog of cutting the black borders off a 2D scan: it removes
    the empty padding around the acquired cone so downstream processing focuses
    on real anatomy.

      - threshold: intensity level defining "content" for the bounding box; the
        box is the tightest cube containing all voxels > threshold. Applied on
        the *raw* intensities (crop runs before denoise/normalize). Use None to
        skip bounding-box detection and only apply `margin`.
      - margin: how much extra border to cut off *every* side (the "how much"
        hyperparameter). An int is voxels per side; a float in (0, 1) is a
        fraction of each axis length. 0 = crop to the content box only.

    Cropping is done via SimpleITK slicing, which updates the image origin so the
    remaining voxels keep their correct physical coordinates. Returns a (possibly
    smaller) sitk.Image.
    """
    arr = sitk.GetArrayFromImage(sitk_image)          # (z, y, x)
    shape = np.array(arr.shape)

    # 1. Base bounding box from content above threshold (else the full volume).
    if threshold is not None:
        mask = arr > threshold
        if not mask.any():
            if verbose:
                print(f"crop_volume: no voxel > {threshold}; skipping crop.")
            return sitk_image
        coords = np.argwhere(mask)
        lo = coords.min(axis=0)            # (z0, y0, x0)
        hi = coords.max(axis=0) + 1        # exclusive upper bound
    else:
        lo = np.zeros(3, dtype=int)
        hi = shape.copy()

    # 2. Trim an extra margin off every side (int voxels, or float fraction).
    axis_len = hi - lo
    if isinstance(margin, float) and 0.0 < margin < 1.0:
        m = np.round(axis_len * margin).astype(int)
    else:
        m = np.full(3, int(margin), dtype=int)
    lo = lo + m
    hi = hi - m

    # 3. Guard against over-cropping, keep at least 1 voxel per axis.
    lo = np.clip(lo, 0, shape - 1)
    hi = np.clip(hi, lo + 1, shape)

    z0, y0, x0 = (int(v) for v in lo)
    z1, y1, x1 = (int(v) for v in hi)
    cropped = sitk_image[x0:x1, y0:y1, z0:z1]          # sitk is (x, y, z); updates origin

    if verbose:
        print(f"crop_volume: {tuple(int(s) for s in shape[::-1])} (x,y,z) -> "
              f"{tuple(cropped.GetSize())}")
    return cropped


def otsu_threshold(array, nbins=256):
    """
    Computes Otsu's threshold: the intensity level that maximizes between-
    class variance, splitting `array` into a low- and a high-intensity class.

    This is the standard parameter-free way to pick a foreground/background
    cutoff from a volume's own histogram, so the same threshold *setting*
    (auto_contour=True below) adapts per scan instead of requiring a hand-
    tuned absolute contour_value that only holds for one scanner/source.
    """
    finite = np.asarray(array, dtype=np.float64)
    finite = finite[np.isfinite(finite)]

    hist, bin_edges = np.histogram(finite, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    weight1 = np.cumsum(hist).astype(np.float64)
    weight2 = weight1[-1] - weight1

    sum1 = np.cumsum(hist * bin_centers)
    mean1 = np.divide(sum1, weight1, out=np.zeros_like(sum1), where=weight1 > 0)
    sum2 = sum1[-1] - sum1
    mean2 = np.divide(sum2, weight2, out=np.zeros_like(sum2), where=weight2 > 0)

    # Between-class variance at every candidate split; skip the last bin edge
    # (no "class 2" left to split against).
    variance12 = weight1[:-1] * weight2[:-1] * (mean1[:-1] - mean2[:-1]) ** 2
    best_idx = int(np.argmax(variance12))
    return float(bin_centers[best_idx])


def load_and_preprocess_volume(nrrd_path, preprocess=True, denoise="median", denoise_radius=1,
                               normalize=True, clip_percentiles=(1.0, 99.0), out_max=255.0,
                               crop=False, crop_threshold=0, crop_margin=0):
    """
    Loads a NRRD file and applies the same crop -> denoise/normalize pipeline
    `convert_single_nrrd` uses before contouring, returning the array together
    with the spacing/origin of the frame its voxel indices live in.

    Factored out so any other code that needs to map a WORLD-space point (e.g.
    a `convert_single_nrrd`-derived surface vertex) back to a voxel index -
    such as sampling texture around a mesh's own surface points - uses the
    exact same frame the surface was extracted in, rather than an
    independently-recomputed (and possibly drifted) crop/spacing/origin.

    Returns (np_array [z, y, x], spacing (sx, sy, sz), origin (ox, oy, oz)).
    """
    sitk_image = sitk.ReadImage(nrrd_path)

    # Crop first so spacing/origin below reflect the cropped volume.
    if crop:
        sitk_image = crop_volume(sitk_image, threshold=crop_threshold,
                                 margin=crop_margin)

    spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()

    if preprocess:
        np_array = preprocess_volume(sitk_image, denoise=denoise,
                                     radius=denoise_radius,
                                     normalize=normalize,
                                     clip_percentiles=clip_percentiles,
                                     out_max=out_max)
    else:
        np_array = sitk.GetArrayFromImage(sitk_image)

    return np_array, spacing, origin


def convert_single_nrrd(nrrd_path, contour_value=None, auto_contour=False, preprocess=True,
                        denoise="median", denoise_radius=1,
                        normalize=True, clip_percentiles=(1.0, 99.0), out_max=255.0,
                        crop=False, crop_threshold=0, crop_margin=0):
    """
    Converts a single NRRD file to a VTP file.

    When crop=True the volume is first cropped to its content bounding box (plus
    crop_margin trimmed off each side). When preprocess=True the volume is then
    denoised (and, if normalize=True, intensity-normalized to [0, out_max])
    before contouring. With normalize=True contour_value is a level relative to
    out_max; with normalize=False it stays on the original intensity scale.

    When auto_contour=True, contour_value is IGNORED and instead computed per
    volume via otsu_threshold() on the (preprocessed) intensity histogram -
    use this to get a consistent surface across NRRD sources with different
    intensity scales/calibrations, without hand-tuning contour_value per source.
    """
    np_array, spacing, origin = load_and_preprocess_volume(
        nrrd_path, preprocess=preprocess, denoise=denoise, denoise_radius=denoise_radius,
        normalize=normalize, clip_percentiles=clip_percentiles, out_max=out_max,
        crop=crop, crop_threshold=crop_threshold, crop_margin=crop_margin,
    )

    if auto_contour:
        contour_value = otsu_threshold(np_array)
        print(f"convert_single_nrrd: auto contour_value (Otsu) = {contour_value:.4g}")

    # 2. Convert to vtkImageData
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(np_array.shape[2], np_array.shape[1], np_array.shape[0])
    vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])
    vtk_image.SetOrigin(origin[0], origin[1], origin[2])
    
    vtk_data_array = numpy_support.numpy_to_vtk(
        num_array=np_array.ravel(), 
        deep=True, 
        array_type=numpy_support.get_vtk_array_type(np_array.dtype)
    )
    vtk_data_array.SetName("Scalars")
    vtk_image.GetPointData().SetScalars(vtk_data_array)
    
    # 3. Convert vtkImageData to vtkPolyData (Geometry generation)
    if contour_value is not None:
        contour_filter = vtk.vtkMarchingCubes()
        contour_filter.SetInputData(vtk_image)
        contour_filter.SetValue(0, contour_value)
        contour_filter.Update()
        poly_data = contour_filter.GetOutput()
    else:
        geometry_filter = vtk.vtkGeometryFilter()
        geometry_filter.SetInputData(vtk_image)
        geometry_filter.Update()
        poly_data = geometry_filter.GetOutput()


    return poly_data