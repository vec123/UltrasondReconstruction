import math, torch, torch.nn as nn

class RadialFourier(nn.Module):
    """Edge-length -> [sin, cos] Fourier features with a smooth cutoff at r_max."""
    def __init__(self, num_freqs=8, r_max=0.25, p=6):
        super().__init__()
        self.r_max, self.p = r_max, p
        self.register_buffer("freqs", math.pi * torch.arange(1, num_freqs + 1) / r_max)

    def envelope(self, d):                       # DimeNet polynomial cutoff -> 0 at r_max
        x = (d / self.r_max).clamp(max=1.0)
        p = self.p
        return 1 - (p+1)*(p+2)/2*x**p + p*(p+2)*x**(p+1) - p*(p+1)/2*x**(p+2)

    def forward(self, d):                         # d: [E, 1]
        xd = d * self.freqs                        # [E, num_freqs]
        basis = torch.cat([torch.sin(xd), torch.cos(xd)], dim=-1)
        return basis * self.envelope(d)            # [E, 2*num_freqs], vanishes at r_max
