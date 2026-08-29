import numpy as np
from scipy.spatial import cKDTree
import torch
import numpy as np
from scipy.spatial import KDTree

def build_kdtree(vertices):
    return cKDTree(vertices)


def closest_points(source_vertices,
                   target_vertices):

    tree = build_kdtree(target_vertices)

    _, indices = tree.query(source_vertices)

    return target_vertices[indices]


def compute_displacements(current,
                          targets):

    return targets - current


def match_constrained(source_verts, target_verts, source_normals, target_normals, 
                      dist_thresh, normal_thresh_deg):
    
    #  Build tree on target
    tree = KDTree(target_verts)
    
    # Query neighbors (find top K to allow for filtering)
    # K=5 is usually sufficient to find a valid match if one exists
    dists, indices = tree.query(source_verts, k=5)
    
    # Apply Constraints
    valid_matches = []
    
    # Precompute normal compatibility threshold (cos theta)
    cos_threshold = np.cos(np.deg2rad(normal_thresh_deg))
    
    for i in range(len(source_verts)):
        for k in range(5):
            idx = indices[i, k]
            dist = dists[i, k]
            
            # Distance Constraint
            if dist > dist_thresh:
                continue
                
            # Normal Constraint: dot product of normals must be > cos(45)
            # Assumes normals are normalized
            dot_prod = np.dot(source_normals[i], target_normals[idx])
            if dot_prod < cos_threshold:
                continue

            valid_matches.append((i, idx, dist))
            break # Stop at first valid match for this source point
            
    return valid_matches