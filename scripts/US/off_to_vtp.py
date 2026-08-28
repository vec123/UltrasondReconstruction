import vtk
import os
import glob

from dotenv import load_dotenv
load_dotenv()

def convert_off_to_vtp_vtk(input_file, output_file):
    # 1. Parse the .off file
    with open(input_file, 'r') as f:
        # Skip the 'OFF' header
        header = f.readline().strip()
        if header != "OFF":
            # Some files have OFF on the same line as counts
            if header.startswith("OFF"):
                parts = header[3:].split()
            else:
                raise ValueError("Invalid OFF file format")
        else:
            parts = f.readline().split()
            
        num_points = int(parts[0])
        num_cells = int(parts[1])

        points = vtk.vtkPoints()
        polydata = vtk.vtkPolyData()

        # 2. Read vertices
        for _ in range(num_points):
            coords = list(map(float, f.readline().split()))
            points.InsertNextPoint(coords[0], coords[1], coords[2])
        polydata.SetPoints(points)

        # 3. Read faces
        cells = vtk.vtkCellArray()
        for _ in range(num_cells):
            data = list(map(int, f.readline().split()))
            num_verts = data[0]
            # Create a triangle/polygon
            cell = vtk.vtkPolygon()
            cell.GetPointIds().SetNumberOfIds(num_verts)
            for i in range(num_verts):
                cell.GetPointIds().SetId(i, data[i+1])
            cells.InsertNextCell(cell)
        polydata.SetPolys(cells)

    # 4. Save as .vtp using VTK's XML PolyData Writer
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(polydata)
    writer.Write()
    
    print(f"Successfully converted: {os.path.basename(input_file)}")

# --- Setup Paths ---
# Adjust paths as needed for your specific inversion workflow
INPUT_DIR_OFF = os.path.join(os.environ.get("PROJECT_ROOT"), "US_samples", "US_samples", "1", "20_semanas", "OFF")
OUTPUT_DIR_VTP = os.path.join(os.environ.get("PROJECT_ROOT"), "US_samples", "US_samples", "1", "20_semanas", "VTK_Converted")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR_VTP):
        os.makedirs(OUTPUT_DIR_VTP)
        
    search_path = os.path.join(INPUT_DIR_OFF, "*.off")
    off_files = glob.glob(search_path)
    
    for off_path in off_files:
        base_name = os.path.splitext(os.path.basename(off_path))[0]
        vtp_filename = f"{base_name}.vtp"
        vtp_file = os.path.join(OUTPUT_DIR_VTP, vtp_filename)
        
        convert_off_to_vtp_vtk(off_path, vtp_file)