"""Stage: attach geodesic-distance / heatmap fields to each aligned mesh.

For each subject shape, finds the nearest mesh vertex to each landmark,
computes surface geodesic distances (Dijkstra over the mesh edge graph) from
every landmark to every vertex, and attaches per-landmark distance and
Gaussian-heatmap scalar fields plus their superposition.

Shapes and landmarks are expected to already live in the same coordinate
space (paired by subject id); this stage does not move geometry, so it saves
no transform.
"""

import dataclasses
import glob
import os

import numpy as np
import scipy.sparse as sp
import vtk
from scipy.sparse.csgraph import dijkstra
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from src.vtk.io import load_vtp, save_vtp

from ..landmarks import points_to_labeled_polydata
from ..matching import subject_id_from_path


@dataclasses.dataclass
class Config:
    shapes_dir: str = ""
    landmarks_dir: str = ""
    output_dir: str = ""

    # Gaussian width of the heatmaps, in mesh units of geodesic distance along the surface.
    heatmap_sigma: float = 0.0014


def build_mesh_graph(polydata):
    """
    Constructs a weighted sparse adjacency graph representation of a VTK surface mesh,
    where edge weights correspond to Euclidean distances between adjacent vertices.
    """
    num_points = polydata.GetNumberOfPoints()
    pts_np = vtk_to_numpy(polydata.GetPoints().GetData())

    rows, cols, data = [], [], []
    visited_edges = set()

    polys = polydata.GetPolys()
    polys.InitTraversal()
    id_list = vtk.vtkIdList()

    while polys.GetNextCell(id_list):
        n = id_list.GetNumberOfIds()
        for i in range(n):
            u = id_list.GetId(i)
            v = id_list.GetId((i + 1) % n)

            edge = (u, v) if u < v else (v, u)
            if edge not in visited_edges:
                visited_edges.add(edge)
                edge_len = np.linalg.norm(pts_np[u] - pts_np[v])
                rows.extend([u, v])
                cols.extend([v, u])
                data.extend([edge_len, edge_len])

    return sp.csr_matrix((data, (rows, cols)), shape=(num_points, num_points), dtype=np.float32)


def extract_landmarks(lmk_polydata):
    """Extracts landmark positions and labels from a landmark PolyData."""
    lmk_pts = vtk_to_numpy(lmk_polydata.GetPoints().GetData())
    label_array = lmk_polydata.GetPointData().GetAbstractArray("label")
    num_pts = lmk_polydata.GetNumberOfPoints()

    labels = [
        str(label_array.GetValue(i)) if label_array else f"lmk_{i}"
        for i in range(num_pts)
    ]
    return lmk_pts, labels


def add_point_array(polydata, name, values):
    """Attaches a float32 numpy array to the polydata's point data under `name`."""
    vtk_arr = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32), deep=True)
    vtk_arr.SetName(name)
    polydata.GetPointData().AddArray(vtk_arr)
    return vtk_arr


def compute_and_attach_geodesics(mesh_polydata, lmk_polydata, sigma):
    """
    Finds nearest mesh vertex for each landmark, computes surface geodesic distance
    fields to all other points using Dijkstra, and attaches them as scalar arrays:

      geo_dist_{label}      raw geodesic distance to that landmark
      heatmap_{label}       Gaussian heatmap exp(-d^2 / 2*sigma^2), peaking at 1.0
      geo_dist_min          distance to the closest landmark, over all landmarks
      superimposed_heatmap  all heatmap_{label} fields summed into one field
    """
    graph = build_mesh_graph(mesh_polydata)
    lmk_pts, lmk_labels = extract_landmarks(lmk_polydata)

    point_locator = vtk.vtkPointLocator()
    point_locator.SetDataSet(mesh_polydata)
    point_locator.BuildLocator()

    nearest_vertex_indices = [point_locator.FindClosestPoint(pt) for pt in lmk_pts]

    mesh_pts = vtk_to_numpy(mesh_polydata.GetPoints().GetData())
    snap_dists = np.linalg.norm(mesh_pts[nearest_vertex_indices] - lmk_pts, axis=1)
    print(
        f"Snapped {len(lmk_labels)} landmarks onto the surface "
        f"(max offset {snap_dists.max():.5f}, mean {snap_dists.mean():.5f})."
    )

    dist_matrix = dijkstra(
        csgraph=graph, directed=False, indices=nearest_vertex_indices, return_predecessors=False
    )

    num_points = mesh_polydata.GetNumberOfPoints()
    min_geodesic_dist = np.full(num_points, np.inf, dtype=np.float32)
    superimposed = np.zeros(num_points, dtype=np.float32)
    two_sigma_sq = 2.0 * (sigma**2)

    for i, label in enumerate(lmk_labels):
        dists = dist_matrix[i].astype(np.float32)
        min_geodesic_dist = np.minimum(min_geodesic_dist, dists)

        heatmap = np.exp(-np.square(dists) / two_sigma_sq).astype(np.float32)
        superimposed += heatmap

        add_point_array(mesh_polydata, f"geo_dist_{label}", dists)
        add_point_array(mesh_polydata, f"heatmap_{label}", heatmap)

    add_point_array(mesh_polydata, "geo_dist_min", min_geodesic_dist)
    add_point_array(mesh_polydata, "superimposed_heatmap", superimposed)
    mesh_polydata.GetPointData().SetActiveScalars("superimposed_heatmap")

    return mesh_polydata


def build_mean_landmarks(lmk_files):
    """
    Averages landmark positions label by label over several landmark .vtp files, giving
    one canonical landmark set for shapes that carry no subject id of their own.
    """
    sums, counts, labels = {}, {}, []

    for lmk_file in lmk_files:
        pts, file_labels = extract_landmarks(load_vtp(lmk_file))
        for pt, label in zip(pts, file_labels):
            if label not in sums:
                sums[label] = np.zeros(3, dtype=np.float64)
                counts[label] = 0
                labels.append(label)
            sums[label] += pt
            counts[label] += 1

    mean_pts = np.array([sums[label] / counts[label] for label in labels], dtype=np.float32)
    lmk_polydata = points_to_labeled_polydata(mean_pts, labels=labels)

    per_label = sorted(set(counts.values()))
    print(
        f"Averaged {len(labels)} landmarks over {len(lmk_files)} landmark files "
        f"({per_label} file(s) contributing per label)."
    )
    return lmk_polydata


def process_shape(shape_vtp_path, lmk_polydata, output_dir, sigma):
    """Computes geodesic heatmaps for one shape against one landmark set, and saves it."""
    os.makedirs(output_dir, exist_ok=True)

    mesh_poly = load_vtp(shape_vtp_path)
    print(f"  {mesh_poly.GetNumberOfPoints()} points against {lmk_polydata.GetNumberOfPoints()} landmarks, sigma = {sigma}")

    output_mesh = compute_and_attach_geodesics(mesh_poly, lmk_polydata, sigma=sigma)

    out_path = os.path.join(output_dir, os.path.basename(shape_vtp_path))
    save_vtp(output_mesh, out_path)
    print(f"  Saved: {out_path}")
    return out_path


def run(cfg: Config):
    shape_files = sorted(glob.glob(os.path.join(cfg.shapes_dir, "*.vtp")))
    if not shape_files:
        raise FileNotFoundError(f"No .vtp shapes found in '{cfg.shapes_dir}'.")

    lmk_files = sorted(glob.glob(os.path.join(cfg.landmarks_dir, "*.vtp")))
    if not lmk_files:
        raise FileNotFoundError(f"No .vtp landmarks found in '{cfg.landmarks_dir}'.")

    strip_suffixes = ("_landmarks", "_heatmap_aligned", "_aligned", "_geodesics")
    lmk_map = {subject_id_from_path(f, strip_suffixes): f for f in lmk_files}
    print(f"Found {len(shape_files)} shapes and {len(lmk_files)} landmark sets.")

    mean_lmk_polydata = None  # only built once, and only if some shape actually needs it

    for idx, shape_path in enumerate(shape_files):
        subject_id = subject_id_from_path(shape_path, strip_suffixes)
        print(f"Processing ({idx + 1}/{len(shape_files)}): {os.path.basename(shape_path)}")

        if subject_id in lmk_map:
            lmk_polydata = load_vtp(lmk_map[subject_id])
            print(f"  Landmarks: {os.path.basename(lmk_map[subject_id])}")
        else:
            if mean_lmk_polydata is None:
                mean_lmk_polydata = build_mean_landmarks(lmk_files)
            lmk_polydata = mean_lmk_polydata
            print(f"  Landmarks: mean over all {len(lmk_files)} sets (no match for '{subject_id}')")

        process_shape(shape_path, lmk_polydata, output_dir=cfg.output_dir, sigma=cfg.heatmap_sigma)

    print("Geodesic distance computation completed successfully!")
