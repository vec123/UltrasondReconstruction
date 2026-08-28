
import os 
import glob
import vtk

from dotenv import load_dotenv
load_dotenv()

def convert_vtp_to_off_vtk(input_file, output_file):
    # 1. Read the VTP file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_file)
    reader.Update()
    polydata = reader.GetOutput()

    # 2. Extract Data (Points and Cells)
    # VTK's structure needs to be mapped to the text-based OFF format
    points = polydata.GetPoints()
    num_points = points.GetNumberOfPoints()
    
    cells = polydata.GetPolys()
    num_cells = cells.GetNumberOfCells()

    # 3. Write to .OFF format (Manually)
    # OFF files start with the header 'OFF'
    with open(output_file, 'w') as f:
        f.write("OFF\n")
        f.write(f"{num_points} {num_cells} 0\n")

        # Write vertices
        for i in range(num_points):
            pt = points.GetPoint(i)
            f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")

        # Write faces
        # Using vtkIdList to traverse cells
        id_list = vtk.vtkIdList()
        cells.InitTraversal()
        while cells.GetNextCell(id_list):
            num_ids = id_list.GetNumberOfIds()
            f.write(f"{num_ids}")
            for i in range(num_ids):
                f.write(f" {id_list.GetId(i)}")
            f.write("\n")

    print(f"File converted to {output_file}")




PROJECT_ROOT = os.environ.get("PROJECT_ROOT")

INPUT_DIR = os.path.join(PROJECT_ROOT, "US_samples","US_samples","1","20_semanas","VTK")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "US_samples", "US_samples","1","20_semanas", "OFF")

if __name__ == "__main__":


        if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
                print(f"Created output directory: {OUTPUT_DIR}")
            
        # Find all .nrrd files in the target folder
        search_path = os.path.join(INPUT_DIR, "*.vtp")
        vtp_files = glob.glob(search_path)
    
        if not vtp_files:
            raise ValueError(f"No .nrrd files found in {INPUT_DIR}")

        print(f"Found {len(vtp_files)} .nrrd files to process.\n" + "-"*40)

        for vtp_path in vtp_files:
            base_name = os.path.splitext(os.path.basename(vtp_path))[0]
            off_filename = f"{base_name}.off"
            off_file = os.path.join(OUTPUT_DIR,off_filename)
            convert_vtp_to_off_vtk(vtp_path, off_file)
