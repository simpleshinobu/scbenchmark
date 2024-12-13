from __future__ import annotations
import math
import torch
import torch.nn as nn
from .mamba.mamba_ssm import Mamba as MambaBlock

class Mamba(nn.Module):
    def __init__(self, d_model, n_layer, d_conv=4, d_state=4, dt_rank='auto', args=None):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        self.args.bias = False
        self.args.conv_bias = True
        self.args.d_conv = d_conv
        self.args.d_state = d_state
        self.args.d_model = d_model
        self.args.n_layer = n_layer
        self.args.d_inner = d_model
        self.args.dt_rank = math.ceil(d_model / 16)

        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, args):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(d_model=args.d_model,  d_state=args.d_state ,  d_conv=args.d_conv,  expand=2)
        self.norm = RMSNorm(args.d_model)
    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
