# from asyncio import graph

import math
import torch
import torch_cluster
from torch_cluster import radius
from torch_geometric.data import Data, Batch
from torch_geometric.nn import radius
import numpy as np
from torch_geometric.nn import radius_graph
from torch_scatter import scatter


def estimate_vertex_areas(pos, batch, k=8):
    """Per-vertex surface measure a_i from the CURRENT cloud (no mesh/canonical needed):
    a_i = pi * r_k^2 / k, r_k = distance to the k-th nearest neighbour -- a kNN density
    estimate (a_i ~ inverse local sampling density). So Sum a_i ~ surface area and
    Sum a_i f(x_i) ~ integral_S f dA at ANY resolution. O(N k), batch-aware."""
    N = pos.size(0)
    k = min(k, N - 1)
    row, col = torch_cluster.knn(pos, pos, k + 1, batch, batch)   # k+1 = self + k neighbours
    d = (pos[row] - pos[col]).norm(dim=-1)

    rk = scatter(d, row, dim=0, dim_size=N, reduce='max')
    return math.pi * rk.pow(2) / k


def apply_noise_and_masking(vertices, masks=None, noise_std=0.0, dropout_rate=0.0, key=None):
    """Stage 1: Apply stochastic augmentations."""
    device = vertices.device
    gen = key if isinstance(key, torch.Generator) else None
    
    # Apply Noise
    if noise_std > 0:
        vertices = vertices + torch.randn(vertices.shape, device=device, generator=gen) * noise_std
        
    # Apply Masking/Dropout
    mask = (masks > 0.5) if masks is not None else torch.ones((vertices.shape[0], vertices.shape[1]), dtype=torch.bool, device=device)
    if dropout_rate > 0:
        random_mask = torch.rand(mask.shape, device=device, generator=gen) > dropout_rate
        mask &= random_mask
        
    return vertices, mask


def sample_nodes(vertices, mask, num_samples=50, mode='fps', key=None, return_indices=False):
    """
    Samples nodes from batches of vertices. 
    Returns: 
        valid_nodes: Tensor [Total_Sampled_Nodes, 3]
        batch_vec: Tensor [Total_Sampled_Nodes]
        selected_indices (optional): Tensor [Total_Sampled_Nodes] of indices into the
            original per-batch vertex array.
    """
    batch_size = vertices.shape[0]
    device = vertices.device
    gen = key if isinstance(key, torch.Generator) else None
    
    selected_nodes_list = []
    batch_vec_list = []
    selected_indices_list = []

    for i in range(batch_size):
        # 1. Extract valid nodes for this batch
        valid_indices = mask[i].nonzero(as_tuple=True)[0]
        nodes = vertices[i, valid_indices]
        num_available = nodes.size(0)

        if num_available == 0:
            continue

        # num_samples=None -> keep all valid nodes (full-graph path)
        n = num_available if num_samples is None else min(num_samples, num_available)

        # 2. Sampling Logic
        if num_samples is None or n >= num_available:
            # Keep-all: no sub-sampling needed
            idx = torch.arange(num_available, device=device)

        elif mode == 'fps':
            # Farthest Point Sampling: returns indices relative to the 'nodes' subset.
            # random_start=False keeps sampling reproducible; slice to exact n.
            idx = torch_cluster.fps(nodes, ratio=n / num_available, random_start=False)
            idx = idx[:n]

        elif mode == 'gaussian':
            com = nodes.mean(dim=0)
            dist = torch.norm(nodes - com, dim=1)
            probs = torch.exp(-0.5 * (dist / (dist.std() + 1e-6))**2)
            idx = torch.multinomial(probs, num_samples=n, replacement=False, generator=gen)
            
        else: # Uniform
            rand_vals = torch.rand(num_available, device=device, generator=gen)
            idx = torch.argsort(rand_vals, descending=True)[:n]

        # 3. Collect results dynamically
        selected = nodes[idx]
        selected_nodes_list.append(selected)
        batch_vec_list.append(torch.full((selected.size(0),), i, device=device))
        if return_indices:
            selected_indices_list.append(valid_indices[idx])

    # 4. Final concatenation ensures x.size(0) == batch_vec.numel()
    if return_indices:
        return (torch.cat(selected_nodes_list, dim=0),
                torch.cat(batch_vec_list, dim=0),
                torch.cat(selected_indices_list, dim=0))
    return torch.cat(selected_nodes_list, dim=0), torch.cat(batch_vec_list, dim=0)


def build_super_graph(vertices, mask, full_graph, num_samples=50, r_max=0.2, mode = "fps", seed=None, max_num_neighbors=128):
    super_nodes, super_batch = sample_nodes(vertices, mask, num_samples=num_samples, mode=mode)

    # 4. Bipartite aggregation graph: each supernode gathers the full-graph nodes within
    #    r_max (the neighbourhood a supernode aggregates through the GNN). ``voronoi``/``seed``
    #    select the aggregation mode / reproducible neighbour cap (see build_bipartite_graph).
    super_graph = build_bipartite_graph(
            full_graph.pos, full_graph.batch, super_nodes, super_batch, r_max=r_max,
            max_num_neighbors=max_num_neighbors,  seed=seed
    )
    if hasattr(full_graph, 'area') and full_graph.area is not None:
        src_area = full_graph.area
        tgt, src = super_graph.edge_index                    # row0=super, row1=full

        deg = torch.bincount(src, minlength=src_area.size(0)).clamp_min(1)
        sa = torch.zeros(super_graph.num_nodes, dtype=src_area.dtype, device=src_area.device)
        sa.index_add_(0, tgt, src_area[src] / deg[src])
        
        super_graph.area = sa
        super_graph.source_area = src_area
    if hasattr(full_graph, 'normal') and full_graph.normal is not None:
        super_graph.source_normal = full_graph.normal
    return super_graph
    
def build_radius_graph(nodes, batch_vec, r_max=0.4, max_num_neighbors=256):
    """Compute graph structure from node positions.

    ``max_num_neighbors`` caps the degree per node: any node with more than this
    many neighbours within ``r_max`` gets its edge list SILENTLY TRUNCATED, so the
    result is no longer a true r-ball graph. Keep it comfortably above the densest
    expected neighbourhood (raise it if you see suspiciously uniform degrees).
    """
    edge_index = radius_graph(nodes, r=r_max, batch=batch_vec, max_num_neighbors=max_num_neighbors)
    return Data(pos=nodes, edge_index=edge_index, batch=batch_vec)

def _voronoi_bipartite_edges(full_nodes, full_batch, super_nodes, super_batch, r_max):
    """Voronoi (partition) assignment: connect each full node to its SINGLE nearest supernode
    within the same batch. Returns edge_index [2, E] with row0=super (target), row1=full
    (source) -- the same convention as the radius path.

    ``r_max`` gates the assignment: a full node farther than ``r_max`` from every supernode in
    its batch is left unassigned. With a sufficiently large ``r_max`` this is a full partition
    (each full node counted exactly once), so a supernode owns ~len(full)/len(super) nodes and
    never overflows the radius path's neighbour cap. Pass ``r_max=None`` to disable the gate."""
    tgt, src = [], []
    for b in torch.unique(full_batch):
        f_idx = (full_batch == b).nonzero(as_tuple=True)[0]
        s_idx = (super_batch == b).nonzero(as_tuple=True)[0]
        if f_idx.numel() == 0 or s_idx.numel() == 0:
            continue
        d = torch.cdist(full_nodes[f_idx], super_nodes[s_idx])       # [Fb, Sb]
        mind, nearest = d.min(dim=1)                                 # nearest supernode per node
        keep = mind <= r_max if r_max is not None else torch.ones_like(mind, dtype=torch.bool)
        tgt.append(s_idx[nearest[keep]])
        src.append(f_idx[keep])
    if not tgt:
        empty = torch.empty(0, dtype=torch.long, device=full_nodes.device)
        return torch.stack([empty, empty], dim=0)
    return torch.stack([torch.cat(tgt), torch.cat(src)], dim=0)


def _sample_neighbors_per_super(edge_index, num_super, max_num_neighbors, seed, device):
    """Reproducibly keep at most ``max_num_neighbors`` edges per supernode, chosen as a SEEDED
    RANDOM subset of that supernode's neighbours. ``edge_index`` is [2, E] (row0=super). Where
    a raw radius() call would truncate to whatever it happens to find first, this instead draws
    a reproducible random sample of the supernode's full neighbourhood."""
    row0, row1 = edge_index
    E = row0.numel()
    if E == 0:
        return edge_index
    gen = torch.Generator().manual_seed(int(seed))
    prio = torch.rand(E, generator=gen).to(device)                  # random priority per edge
    p_order = torch.argsort(prio)                                   # random shuffle
    row0, row1 = row0[p_order], row1[p_order]
    s_order = torch.argsort(row0, stable=True)                      # group by supernode (random within)
    row0, row1 = row0[s_order], row1[s_order]
    counts = torch.bincount(row0, minlength=num_super)
    offsets = torch.cumsum(counts, 0) - counts                     # start index of each group
    rank = torch.arange(E, device=device) - offsets[row0]          # within-group rank
    keep = rank < max_num_neighbors
    return torch.stack([row0[keep], row1[keep]], dim=0)


def build_bipartite_graph(full_nodes, full_batch, super_nodes, super_batch, r_max=0.4,
                          max_num_neighbors=128, seed=None):
    """
    Build the bipartite aggregation graph joining supernodes to full-graph nodes -- the
    supernode message-passing structure (n_s supernodes aggregate a spatial neighbourhood).

    Two aggregation modes:
      * radius ball (default, ``voronoi=False``): each supernode gathers ALL full nodes within
        ``r_max``. If a supernode has more than ``max_num_neighbors`` neighbours it is capped.
        With ``seed=None`` the cap is radius()'s built-in truncation (keeps whatever it finds
        first -- arbitrary order); pass an int ``seed`` to instead keep a REPRODUCIBLE RANDOM
        subset of the supernode's full neighbourhood.

    edge_index convention (radius(x=full, y=super)):
        edge_index[0] -> supernode (target / receiver) index into super_nodes
        edge_index[1] -> full node (source / sender)  index into full_nodes

    Returns a PyG ``Data`` describing the bipartite graph:
        pos          : [S, D] supernode positions   (target / output nodes)
        batch        : [S]    supernode batch vector
        source_pos   : [F, D] full-graph positions   (source nodes)
        source_batch : [F]    full-graph batch vector
        edge_index   : [2, E] bipartite edges (row0=super, row1=full)
    """
    assert full_nodes.size(0) == full_batch.size(0), \
        f"Mismatch: full_nodes {full_nodes.size(0)} != full_batch {full_batch.size(0)}"
    assert super_nodes.size(0) == super_batch.size(0), \
        f"Mismatch: super_nodes {super_nodes.size(0)} != super_batch {super_batch.size(0)}"

   
    if seed is not None:
        # Radius ball, but pick each supernode's <= max_num_neighbors as a SEEDED RANDOM
        # subset: retrieve ALL neighbours first (cap = |full|), then subsample per supernode.
        all_edges = radius(
            x=full_nodes, y=super_nodes, r=r_max,
            batch_x=full_batch, batch_y=super_batch,
            max_num_neighbors=full_nodes.size(0),
        )
        edge_index = _sample_neighbors_per_super(all_edges, super_nodes.size(0),
                                                max_num_neighbors, seed, full_nodes.device)
    else:
        # Radius ball with radius()'s built-in (arbitrary-order) cap -- original behaviour.
        edge_index = radius(
            x=full_nodes, y=super_nodes, r=r_max,
            batch_x=full_batch, batch_y=super_batch,
            max_num_neighbors=max_num_neighbors,
        )

    return Data(
        pos=super_nodes,
        batch=super_batch,
        source_pos=full_nodes,
        source_batch=full_batch,
        edge_index=edge_index,
    )

def get_graphs_from_vertices(vertices_padded, masks=None, r_max=0.4, dropout_rate=0.9,
                             noise_std=0.0, key=None, sampling_mode='uniform', num_samples=None,
                             max_num_neighbors=256, features=None, areas=None, normals=None,
                             recompute_area=False, area_k=8):

    if dropout_rate is not None and num_samples is not None:
        raise ValueError("Please provide either dropout_rate or num_samples, not both.")
    if isinstance(masks, np.ndarray):
        masks = torch.tensor(masks, dtype=torch.bool)

    # Standardize input
    v = vertices_padded.clone() if isinstance(vertices_padded, torch.Tensor) else torch.tensor(vertices_padded, dtype=torch.float32)
    
    #  Augment
    v, mask = apply_noise_and_masking(v, masks, noise_std, dropout_rate or 0.0, key)
    
    #  Sample
    if features is not None:
        features = features.clone() if isinstance(features, torch.Tensor) else torch.tensor(features, dtype=torch.float32)
    if areas is not None:
        areas = areas.clone() if isinstance(areas, torch.Tensor) else torch.tensor(areas, dtype=torch.float32)
    if normals is not None:
        normals = normals.clone() if isinstance(normals, torch.Tensor) else torch.tensor(normals, dtype=torch.float32)

    nodes, batch_vec, selected_indices = sample_nodes(v, mask, num_samples, sampling_mode, key, return_indices=True)
    
    #  Build Graph
    graph = build_radius_graph(nodes, batch_vec, r_max, max_num_neighbors=max_num_neighbors)
    if features is not None:
        graph.x = features[batch_vec, selected_indices]
    if areas is not None:
        graph.area = areas[batch_vec, selected_indices]
    if recompute_area:
        # resolution-agnostic: estimate a_i from the CURRENT (post-dropout) cloud, so it
        # reflects the actual sampling rather than inherited canonical areas.
        graph.area = estimate_vertex_areas(nodes, batch_vec, area_k)
    if normals is not None:
        graph.normal = normals[batch_vec, selected_indices]

    return graph

def get_individual_graph(batch_obj, index):
    """
    Extract a single (homogeneous) graph from a PyG Batch object.

    Returns:
        V : (N_i, D)  node positions for shape ``index``
        E : (M_i, 2)  edges as [src, dst] pairs, remapped to LOCAL 0..N_i-1 indices.

    The (M, 2) local-index convention matches ``create_polydata_w_lines`` and
    ``get_bipartite_graph``; callers must NOT transpose or offset the result.
    """
    # 1. Mask for nodes belonging to this graph
    node_mask = (batch_obj.batch == index)
    V = batch_obj.pos[node_mask]

    # 2. Extract edges where the source node belongs to the current graph.
    #    batch_obj.edge_index is (2, total_edges); for a homogeneous radius graph
    #    both endpoints of an in-shape edge live in the same shape.
    edge_mask = (batch_obj.batch[batch_obj.edge_index[0]] == index)
    E = batch_obj.edge_index[:, edge_mask]                      # (2, M) GLOBAL indices

    # 3. Remap global node indices -> local 0..N_i-1 so E references V directly.
    #    A scatter table handles non-contiguous node sets (e.g. dropout-decimated
    #    batches) that a plain ``- start_idx`` offset would get wrong. Sizing off the
    #    indices actually present (not batch_obj.num_nodes, which is ambiguous for a
    #    bipartite Data) keeps this well-defined even when misused on a bipartite graph;
    #    endpoints outside this shape simply map to -1 (visibly wrong, never OOB).
    node_indices = torch.nonzero(node_mask, as_tuple=False).flatten()
    size = int(node_indices.max().item()) + 1 if node_indices.numel() else 0
    if E.numel():
        size = max(size, int(E.max().item()) + 1)
    remap = torch.full((size,), -1, dtype=torch.long, device=E.device)
    remap[node_indices] = torch.arange(node_indices.numel(), device=E.device)
    E_local = remap[E].t().contiguous()                        # (M, 2) LOCAL indices

    return V, E_local

def get_bipartite_graph(graph, batch_idx):
    """Saves one shape's supernode aggregation: full nodes + supernodes joined by the
    bipartite edges each supernode aggregates over.

    The bipartite ``Data`` carries two node sets (``source_pos``/``source_batch`` for
    the full graph and ``pos``/``batch`` for the supernodes); we merge them into a
    single point cloud so the aggregation edges can be rendered as VTP lines.
    """
    # Source (full-graph) nodes for this shape.
    src_mask = (graph.source_batch == batch_idx)
    src_pos = graph.source_pos[src_mask]
    src_start = src_mask.nonzero(as_tuple=True)[0][0]

    # Target (super) nodes for this shape.
    tgt_mask = (graph.batch == batch_idx)
    tgt_pos = graph.pos[tgt_mask]
    tgt_start = tgt_mask.nonzero(as_tuple=True)[0][0]

    # Edges whose supernode (row 0) belongs to this shape, remapped to local indices.
    ei = graph.edge_index
    edge_mask = tgt_mask[ei[0]]
    e_super = ei[0, edge_mask] - tgt_start          # local supernode index
    e_full = ei[1, edge_mask] - src_start           # local full-node index

    # Combined point set [full ; super]; supernode indices are offset by n_full.
    n_full = src_pos.size(0)
    points = torch.cat([src_pos, tgt_pos], dim=0).detach().cpu().numpy()
    lines = torch.stack([e_full, e_super + n_full], dim=0).T.detach().cpu().numpy()


    n_full = src_pos.size(0)
    n_super = tgt_pos.size(0) # Need this to create the field
    
    points = torch.cat([src_pos, tgt_pos], dim=0).detach().cpu().numpy()
    lines = torch.stack([e_full, e_super + n_full], dim=0).T.detach().cpu().numpy()
    
    # Create the field: 0 for source nodes, 1 for supernodes
    # Using float32 or int32; Paraview handles both well for coloring
    node_type = np.zeros(n_full + n_super, dtype=np.int32)
    node_type[n_full:] = 1

    return points, lines, node_type

""" 
def get_individual_graph(batch_obj, index):

    mask = (batch_obj.batch == index)

    # 2. Extract the vertex positions for this graph
    V= batch_obj.pos[mask]
    source_nodes = batch_obj.edge_index[index]
    edge_mask = (batch_obj.batch[source_nodes] == index)
    E = batch_obj.edge_index[:, edge_mask]
    return V,E
"""

def get_vertices_and_edges(individual_graph):
    """
    Extracts vertices and edges from a single PyG Data object.
    
    Args:
        individual_graph (Data): A single graph object.
        
    Returns:
        points (np.array): (N, 3) float32 array
        lines (np.array): (M, 2) int64 array
    """
    # 1. Extract and convert points
    # Since this is an individual graph, no masking/shifting is needed.
    points = individual_graph.pos.detach().cpu().numpy().astype(np.float32)
    
    # 2. Extract and convert edges
    # edge_index is already relative (starts from 0) in a single Data object
    lines = individual_graph.edge_index.T.detach().cpu().numpy().astype(np.int64)
    
    return points, lines






def get_graphs_from_vertices_(
    vertices_padded, 
    masks=None, 
    r_max=0.4, 
    dropout_rate=0.9, 
    noise_std=0.0, 
    key=None, 
    sampling_mode='uniform',
    num_samples=None
):
    if dropout_rate is not None and num_samples is not None:
        raise ValueError("Please provide either dropout_rate or num_samples, not both.")
    
    if not isinstance(vertices_padded, torch.Tensor):
        vertices_padded = torch.tensor(vertices_padded, dtype=torch.float32)
    
    device = vertices_padded.device
    batch_size, num_nodes, _ = vertices_padded.shape
    
    # 1. Masking & Noise
    final_mask = (masks > 0.5) if masks is not None else torch.ones((batch_size, num_nodes), dtype=torch.bool, device=device)
    
    if noise_std > 0:
        # Use torch.randn with explicit shape instead of randn_like
        noise = torch.randn(vertices_padded.shape, device=device, generator=key)
        vertices_padded = vertices_padded + noise * noise_std

    # 2. Vectorized Dropout with optional key
    if dropout_rate is not None and dropout_rate > 0:
        random_values = torch.rand((batch_size, num_nodes), device=device, generator=key)
        random_mask = random_values > dropout_rate
        final_mask = final_mask & random_mask

    data_list = []
    for i in range(batch_size):
        nodes = vertices_padded[i][final_mask[i]]
        
        # Determine how many nodes to keep
        if num_samples is not None:
            n_keep = min(num_samples, nodes.size(0))
        elif dropout_rate is not None and dropout_rate > 0:
            n_keep = max(1, int(nodes.size(0) * (1 - dropout_rate)))
        else:
            # Both are None or dropout_rate is 0: keep all
            n_keep = nodes.size(0)

        # Sampling Logic
        if n_keep < nodes.size(0):
            if sampling_mode == 'gaussian':
                com = nodes.mean(dim=0)
                dist = torch.norm(nodes - com, dim=1)
                sigma = dist.std() + 1e-6
                probs = torch.exp(-0.5 * (dist / sigma)**2)
                keep_indices = torch.multinomial(probs, num_samples=n_keep, replacement=False, generator=key)
                nodes = nodes[keep_indices]
            
            elif sampling_mode == 'uniform':
                indices = torch.randperm(nodes.size(0), generator=key)[:n_keep]
                nodes = nodes[indices]

        # Build Radius Graph
        edge_index = radius(nodes, nodes, r=r_max, max_num_neighbors=128)
        data_list.append(Data(pos=nodes, edge_index=edge_index))
    
    return Batch.from_data_list(data_list)
