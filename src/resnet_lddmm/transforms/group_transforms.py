import torch
import numpy as np
import math

def get_y_rot(t):
    cos_t = torch.cos(t)
    sin_t = torch.sin(t)
    # Returns [3, 3] rotation matrix
    return torch.stack([
        torch.stack([cos_t,  torch.zeros_like(t), sin_t]),
        torch.stack([torch.zeros_like(t), torch.ones_like(t), torch.zeros_like(t)]),
        torch.stack([-sin_t, torch.zeros_like(t), cos_t])
    ])

def SE3_transform(nodes, n_node, rotations, translations, permute=False, key=None):
    """
    nodes: [Total_Nodes, 3]
    n_node: [Num_Graphs]
    rotations: [Num_Graphs, 3, 3]
    translations: [Num_Graphs, 3]
    """
    # 1. Create a batch index for every node: [Total_Nodes]
    # e.g., [0, 0, 1, 1, 1, 2, ...]
    batch_idx = torch.repeat_interleave(
        torch.arange(len(n_node), device=nodes.device), 
        n_node
    )
    
    # 2. Map rotations and translations to each node via batch_idx
    # node_rots: [Total_Nodes, 3, 3], node_trans: [Total_Nodes, 3]
    node_rots = rotations[batch_idx]
    node_trans = translations[batch_idx]
    
    # 3. Vectorized Transformation
    # Einstein summation is the most performant way to handle batched matrix-vector mult
    # 'ni,nij->nj' : (N, 3) @ (N, 3, 3) -> (N, 3)
    nodes_new = torch.einsum('ni,nij->nj', nodes, node_rots) + node_trans

    # Debug: check requires_grad
    if nodes.requires_grad or node_rots.requires_grad or node_trans.requires_grad:
        print(f"[SE3_DEBUG] nodes.requires_grad={nodes.requires_grad}, node_rots.requires_grad={node_rots.requires_grad}, node_trans.requires_grad={node_trans.requires_grad}, nodes_new.requires_grad={nodes_new.requires_grad}")
    
    # 4. Optional Permutation
    if permute:
        if key is None: raise ValueError("Permutation requires a torch.Generator")
        # For permutation, we must group by graph index
        # This is inherently slower than the linear transformation above
        for i in range(len(n_node)):
            mask = (batch_idx == i)
            nodes_new[mask] = nodes_new[mask][torch.randperm(nodes_new[mask].size(0), generator=key)]
            
    return nodes_new

def SE3_transform_numpy(nodes, n_node, rotations, translations, permute=False, seed=None):
    """
    nodes: [Total_Nodes, 3]
    n_node: [Num_Graphs]
    rotations: [Num_Graphs, 3, 3]
    translations: [Num_Graphs, 3]
    """
    # 1. Create a batch index for every node: [Total_Nodes]
    batch_idx = np.repeat(np.arange(len(n_node)), n_node)
    
    # 2. Map rotations and translations to each node via batch_idx
    node_rots = rotations[batch_idx]
    node_trans = translations[batch_idx]
    
    # 3. Vectorized Transformation
    # 'ni,nij->nj' : (N, 3) @ (N, 3, 3) -> (N, 3)
    # Note: Using Einstein summation for batched matrix-vector multiplication
    nodes_new = np.einsum('ni,nij->nj', nodes, node_rots) + node_trans
    
    # 4. Optional Permutation
    if permute:
        rng = np.random.default_rng(seed)
        # We process each graph separately to maintain the local graph structure
        for i in range(len(n_node)):
            mask = (batch_idx == i)
            # Extract current graph nodes
            subset = nodes_new[mask]
            # Shuffle indices and reassign
            indices = rng.permutation(subset.shape[0])
            nodes_new[mask] = subset[indices]
            
    return nodes_new