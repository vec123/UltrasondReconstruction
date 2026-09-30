"""Building blocks for velocity fields."""

from typing import List, Sequence
import torch
import torch.nn as nn


class ResidualLayer(nn.Module):
    """h -> skip(h) + act(W·h + b).

    skip is the identity when the width is unchanged, else a bias-free linear
    projection (the standard ResNet projection shortcut).
    """

    def __init__(self, in_dim: int, out_dim: int, act: nn.Module):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        self.act = act

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """[..., in_dim] -> [..., out_dim]."""
        return self.skip(h) + self.act(self.linear(h))


class VelocityBlock(nn.Module):
    """
    Residual block: f(x) = W3·(W2·R_n(…R_1(act(W1·x + b1))…) + b2).

    ``widths`` are the hidden layer widths: W1 lifts to widths[0], each further
    width adds one ResidualLayer R_i, and W2 mixes at widths[-1]. ``widths=(w,)``
    has no R_i -- it is the original single-hidden-layer block with the same
    parameter names, so checkpoints trained with ``field.width: w`` still load.
    """

    def __init__(self, in_dim: int, widths: Sequence[int], activation: str = "relu"):
        super().__init__()
        self.act = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
        }[activation]
        self.lift = nn.Linear(in_dim, widths[0])  # W1, b1
        # Empty for a single width, so it adds no state_dict keys
        self.hidden = nn.ModuleList(
            ResidualLayer(a, b, self.act) for a, b in zip(widths[:-1], widths[1:])
        )
        self.mix = nn.Linear(widths[-1], widths[-1])  # W2, b2
        self.proj = nn.Linear(widths[-1], 3, bias=False)  # W3 — no bias
        # Zero-init proj.weight so v ≡ 0 at init ⇒ flow == identity
        nn.init.zeros_(self.proj.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """[B, N, in_dim] -> [B, N, 3]."""
        h = self.act(self.lift(h))
        for layer in self.hidden:
            h = layer(h)
        return self.proj(self.mix(h))


class FourierFeatures(nn.Module):
    """Positional encoding: γ(h) = [h, sin(2⁰πh), cos(2⁰πh), …, sin(2^{Ne−1}πh), cos(2^{Ne−1}πh)].

    Base-2 log-spaced (AD-SVFD Methods); identity passthrough included.
    Output dim = (2·n_e + 1)·d where d is the input feature dimension.
    """

    def __init__(self, n_e: int = 3):
        super().__init__()
        # Register buffer: frequencies [2^0, 2^1, ..., 2^(n_e-1)] * π
        self.register_buffer("freqs", torch.pi * 2.0 ** torch.arange(n_e))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """[..., d] -> [..., (2*n_e + 1)*d]."""
        # h.unsqueeze(-1) broadcasts [..., d] to [..., d, 1]
        # freqs shape: [n_e]
        # angles shape: [..., d, n_e]
        angles = h.unsqueeze(-1) * self.freqs
        # sin/cos of angles: [..., d, n_e] each
        # flatten to [..., d*n_e] each
        sin_parts = angles.sin().flatten(-2)
        cos_parts = angles.cos().flatten(-2)
        # Concatenate: [h, sin, cos] -> [..., d + d*n_e + d*n_e] = [..., (2*n_e + 1)*d]
        return torch.cat([h, sin_parts, cos_parts], dim=-1)


def mlp(dims: List[int], act: str = "relu", final_bias: bool = True) -> nn.Sequential:
    """Build a multi-layer perceptron.

    Args:
        dims: list of layer dimensions [in, h1, h2, ..., out]
        act: activation function name ("relu" | "leaky_relu")
        final_bias: whether final layer has bias

    Returns:
        Sequential module with intermediate activations, final layer has no activation.
    """
    activation = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.2),
    }[act]

    layers = []
    for i in range(len(dims) - 1):
        is_final = (i == len(dims) - 2)
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=final_bias or not is_final))
        if not is_final:
            layers.append(activation)

    return nn.Sequential(*layers)


def last_linear(module: nn.Module) -> nn.Linear:
    """Extract the last Linear layer from a module.

    Recursively searches for the last Linear layer in Sequential or nested modules.
    """
    if isinstance(module, nn.Linear):
        return module
    if isinstance(module, nn.Sequential):
        for layer in reversed(module):
            if isinstance(layer, nn.Linear):
                return layer
    # Fallback: search all children
    for child in reversed(list(module.children())):
        result = last_linear(child)
        if result is not None:
            return result
    return None
