import numpy as np

def pad_vertex_list(vertex_list):
    """
    Converts a list of vertex arrays into a single padded numpy array.
    
    Args:
        shape_vertices (list): List of numpy arrays, each of shape (N_i, 3).
        
    Returns:
        tuple: (padded_array, mask)
            - padded_array: shape (num_shapes, max_vertices, 3)
            - mask: boolean array of shape (num_shapes, max_vertices) 
                    where True indicates a valid vertex.
    """
    num_shapes = len(vertex_list)
    # Find the maximum number of vertices among all shapes
    max_vertices = max(v.shape[0] for v in vertex_list)
    
    # Initialize arrays with zeros
    # Shape: (number of shapes, max number of vertices, 3 coordinates)
    padded_array = np.zeros((num_shapes, max_vertices, 3))
    
    # Initialize mask: False by default
    mask = np.zeros((num_shapes, max_vertices), dtype=bool)
    
    for i, vertices in enumerate(vertex_list):
        n = vertices.shape[0]
        # Fill the padded array with actual vertices
        padded_array[i, :n, :] = vertices
        # Mark these positions as True in the mask
        mask[i, :n] = True
        
    return padded_array, mask