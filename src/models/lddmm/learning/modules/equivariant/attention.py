
import torch
import torch.nn as nn
from e3nn import o3

class EquivariantAttention(nn.Module):
    def __init__(self, irreps_in, irreps_out, sh_lmax=1, verbose = True):
        super().__init__()
        self.verbose = verbose
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.sh = o3.SphericalHarmonics(
            [l for l in range(1, sh_lmax + 1)], normalize=True
        )
        
        # Projections
        self.lin_q = o3.Linear(self.irreps_in, self.irreps_out)
        self.lin_k = o3.Linear(self.irreps_in + self.sh.irreps_out, self.irreps_out)
        self.lin_v = o3.Linear(self.irreps_in + self.sh.irreps_out, self.irreps_out)
        
        # Dot product: maps to "0e" (scalar)
        self.tp_attn = o3.FullyConnectedTensorProduct(
            self.irreps_out, 
            self.irreps_out, 
            "0e", 
            shared_weights=True # Set to False if you want unique weights per edge/node
        )            
        
    def forward(self, node_features, positions, senders, receivers, num_nodes):
        # 1. Geometry: Spherical Harmonics of relative positions
        rel_pos = positions[receivers] - positions[senders]
        edge_sh = self.sh(rel_pos)
        
        # 2. Queries (Per node)
        q = self.lin_q(node_features)
        
        # 3. Keys and Values (Per edge)
        sender_f = node_features[senders]
        msg_features = torch.cat([sender_f, edge_sh], dim=-1)
        
        k = self.lin_k(msg_features)
        v = self.lin_v(msg_features)
        
        # 4. Attention mechanism
        # Q (receiver) and K (edge) are combined before dot product
        alpha_raw = self.tp_attn(q[receivers], k)
        
        # Softmax normalized per receiver
        alpha = torch.zeros_like(alpha_raw)
        for i in range(num_nodes):
            mask = (receivers == i)
            if mask.any():
                alpha[mask] = torch.softmax(alpha_raw[mask], dim=0)
        
        # 5. Aggregate: weighted sum
        v_weighted = v * alpha
        
        f_out = torch.zeros((num_nodes, v_weighted.shape[-1]), device=v.device)
        f_out.index_add_(0, receivers, v_weighted)

        if self.verbose:
            print("--------------EquivariantAttention --------------")
            print("irreps_in: ", self.irreps_in)
            print("irreps_out: ", self.irreps_out)
            print("f_out shape: ", f_out.shape)
            print("alpha shape: ", alpha.shape)
            print("--------------Finished --------------")
        
        return f_out, alpha