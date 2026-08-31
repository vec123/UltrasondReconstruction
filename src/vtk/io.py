import glob as _glob
import os
import vtk

def load_vtp(vtp_path):
    if not os.path.exists(vtp_path):
        raise FileNotFoundError(f"File not found: {vtp_path}")

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    return reader.GetOutput()

def load_obj(obj_path):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_path)
    reader.Update()
    return reader.GetOutput()

def load_vtu(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()  # This executes the reader

    # Return the actual data object, not the port
    return reader.GetOutput()

def save_vtp(polydata, vtk_path, binary=True, verbose = False):
    """Saves the PolyData object to disk using the XML format (VTP)."""
    if verbose:
        print("Saving to:", vtk_path)
    os.makedirs(os.path.dirname(vtk_path), exist_ok=True)
    # Use vtkXMLPolyDataWriter instead of vtkPolyDataWriter
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(vtk_path)
    writer.SetInputData(polydata)

    if binary:
        # For XML files, 'binary' is the default and is handled automatically.
        # You can specify the data mode if needed:
        writer.SetDataModeToAppended()
        writer.SetCompressorTypeToZLib() # Optional: compresses the file
    else:
        writer.SetDataModeToAscii()

    # vtkXMLPolyDataWriter.Write() returns 0 on failure WITHOUT raising, so an ignored
    # return value turns a failed write into a silent no-op. Raise so callers can't miss it.
    if writer.Write() != 1:
        raise IOError(f"vtkXMLPolyDataWriter failed to write {vtk_path}")
    return True


_MESH_LOADERS_BY_EXT = {".vtp": load_vtp, ".obj": load_obj, ".vtu": load_vtu}


def load_mesh(path):
    """Loads a mesh file, dispatching on its extension (.vtp/.ply/.obj/.vtu/.vtk)."""
    path = str(path)
    ext = os.path.splitext(path)[1].lower()

    loader = _MESH_LOADERS_BY_EXT.get(ext)
    if loader is not None:
        return loader(path)

    if ext == ".ply":
        reader = vtk.vtkPLYReader()
        reader.SetFileName(path)
        reader.Update()
        return reader.GetOutput()

    if ext == ".vtk":
        from src.vtk.convert import _read_legacy_vtk  # deferred: convert imports this module
        return _read_legacy_vtk(path)

    raise ValueError(f"Unsupported mesh file extension {ext!r} for {path}")


def glob_meshes(directory, extensions=(".vtp", ".ply", ".obj", ".vtk")):
    """Lists mesh files (by extension) directly inside `directory`, sorted."""
    files = []
    for ext in extensions:
        files.extend(_glob.glob(os.path.join(directory, f"*{ext}")))
    return sorted(files)
