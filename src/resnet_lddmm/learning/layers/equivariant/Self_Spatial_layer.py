import torch
import torch.nn as nn
from e3nn import o3
from src.learning.modules.equivariant.interaction import (
    SelfInteraction,
    SpatialConvolution,
)
from src.learning.modules.equivariant.layer_norm import (
    EquivariantLayerNorm
)

class EquiLayer(nn.Module):
    def __init__(self, in_irreps, target_irreps, spatial_sh_lmax = 1,
                 interaction_sh_lmax = 1, interaction_irreps = None, verbose=True):
        super().__init__()
        self.in_irreps = o3.Irreps(in_irreps)
        self.target_irreps = o3.Irreps(target_irreps)
        if interaction_irreps != None:
            self.interaction_irreps = o3.Irreps(interaction_irreps)
        else: 
            self.interaction_irreps = o3.Irreps(target_irreps)
        self.spatial_sh_lmax = spatial_sh_lmax
        self.interaction_sh_lmax = interaction_sh_lmax
        self.verbose = verbose

        # Self Interaction 
        self.self_int = SelfInteraction(
            in_irreps=self.in_irreps,
            target_irreps=self.interaction_irreps,
            sh_lmax=self.interaction_sh_lmax,
            verbose=verbose
        )

        # followed by Spatial Convolution
        self.spatial_conv = SpatialConvolution(
            in_irreps=self.interaction_irreps,
            target_irreps=self.target_irreps,
            sh_lmax=self.spatial_sh_lmax,
            verbose=verbose
        )

        # 3. Residual Projection (if irreps mismatch)
        self.res_proj = None
        if self.in_irreps != self.target_irreps:
            self.res_proj = o3.Linear(self.in_irreps, self.target_irreps)

        # 4. Layer Norm
        self.norm = EquivariantLayerNorm(self.target_irreps, verbose=verbose)

    def forward(self, x, pos, edge_index, area=None):
        # x is node features, pos is node positions, edge_index for graph structure.
        # area (optional): per-node surface measure -> area-weighted message aggregation.

        # Self Interaction (node-wise; no aggregation, so area does not apply here)
        h = self.self_int(x)

        # Spatial Convolution
        msg = self.spatial_conv(h, pos, edge_index, area=area)
        
        # Skip Connection
        if self.res_proj is not None:
            skip = self.res_proj(x)
        else:
            skip = x
            
        res = msg + skip
        
        if self.verbose:
            print("------Skip connection--------")
            print("msg + skip with msg = spatial_conv( self_interact(x) )")
            if self.res_proj:
                print(f"self.res_proj is active with"
                    f" self.res_proj.irreps_in: {self.res_proj.irreps_in}"
                    f" self.res_proj.irreps_out: {self.res_proj.irreps_out}")
            print("x.shape: ", x.shape)
            print("msg.shape: ", msg.shape)
            print("skip.shape: ", skip.shape)
            print("res.shape: ", res.shape)
            print("-------Finished--------")

        # Layer Norm
        h_norm = self.norm(res)

        if self.verbose:
            print("--------------Layer --------------")
            print("in.irreps : ", self.in_irreps)
            print("target.irreps : ", self.target_irreps)
            print("out.shape: ", h_norm.shape)
            print("------------Finished-------------")
            
        return h_norm