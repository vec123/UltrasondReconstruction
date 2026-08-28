import os
import sys
import glob
import vtk
from vtk.util import numpy_support
import SimpleITK as sitk
# Allow running this script directly (`python scripts/US/nrrd2vti.py`) from any cwd.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.geometry import preprocess_volume

def convert_nrrd_to_vti(nrrd_path, vti_path, preprocess=True,
                        denoise="median", denoise_radius=1,
                        normalize=True, clip_percentiles=(1.0, 99.0), out_max=255.0):
    """
    Converts a single NRRD file to a VTI (VTK Image Data Volume) file.

    When preprocess=True the volume is denoised (and, if normalize=True,
    intensity-normalized to [0, out_max]) before writing. Set normalize=False to
    keep the original intensity scale, or preprocess=False to archive the raw
    volume untouched.
    """
    # 1. Load the NRRD file with SimpleITK
    sitk_image = sitk.ReadImage(nrrd_path)

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

    # 2. Convert to vtkImageData (The actual volume structure)
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(np_array.shape[2], np_array.shape[1], np_array.shape[0])
    vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])
    vtk_image.SetOrigin(origin[0], origin[1], origin[2])
    
    vtk_data_array = numpy_support.numpy_to_vtk(
        num_array=np_array.ravel(), 
        deep=True, 
        array_type=numpy_support.get_vtk_array_type(np_array.dtype)
    )
    vtk_data_array.SetName("Ultrasound_Intensities")
    vtk_image.GetPointData().SetScalars(vtk_data_array)
    
    # 3. Save as a VTI file using vtkXMLImageDataWriter (Keeps it as a volume!)
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(vti_path)
    writer.SetInputData(vtk_image)
    writer.Write()

def batch_convert_folder_to_vti(input_folder, output_folder, preprocess=True,
                                denoise="median", denoise_radius=1,
                                normalize=True, clip_percentiles=(1.0, 99.0), out_max=255.0):
    """
    Finds all .nrrd files in the input folder and converts them to .vti in the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
        
    search_path = os.path.join(input_folder, "*.nrrd")
    nrrd_files = glob.glob(search_path)
    
    if not nrrd_files:
        print(f"No .nrrd files found in {input_folder}")
        return

    print(f"Found {len(nrrd_files)} .nrrd files to process as volumes.\n" + "-"*40)

    for nrrd_path in nrrd_files:
        base_name = os.path.splitext(os.path.basename(nrrd_path))[0]
        vti_filename = f"{base_name}.vti"
        vti_path = os.path.join(output_folder, vti_filename)
        
        print(f"Processing Volume: {os.path.basename(nrrd_path)} -> {vti_filename}")
        try:
            convert_nrrd_to_vti(nrrd_path, vti_path, preprocess=preprocess,
                                denoise=denoise, denoise_radius=denoise_radius,
                                normalize=normalize,
                                clip_percentiles=clip_percentiles, out_max=out_max)
        except Exception as e:
            print(f"Failed to convert {base_name}: {e}")
            
    print("-"*40 + "\nBatch volume conversion finished!")

if __name__ == '__main__':
    # Configuration
    from dotenv import load_dotenv
    load_dotenv()

    PROJECT_ROOT = os.environ.get("PROJECT_ROOT")

    INPUT_DIR = os.path.join(PROJECT_ROOT, "US_samples","US_samples","1","20_semanas","NRRD")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "US_samples", "US_samples_hr","1","20_semanas", "VTK_vol")
    CACHE_DIR = os.path.join(PROJECT_ROOT, "US_samples", "US_samples_hr","1","20_semanas", "Cache")

    # --- Volume preprocessing (set PREPROCESS=False to archive the raw volume) ---
    PREPROCESS = False
    DENOISE = "median"             # "median", "curvature", or None to disable
    DENOISE_RADIUS = 1
    NORMALIZE = False               # False -> keep original intensity scale (denoise only)
    CLIP_PERCENTILES = (1.0, 99.0)
    NORM_OUT_MAX = 255.0

    batch_convert_folder_to_vti(
        INPUT_DIR,
        OUTPUT_DIR,
        preprocess=PREPROCESS,
        denoise=DENOISE,
        denoise_radius=DENOISE_RADIUS,
        normalize=NORMALIZE,
        clip_percentiles=CLIP_PERCENTILES,
        out_max=NORM_OUT_MAX,
    )