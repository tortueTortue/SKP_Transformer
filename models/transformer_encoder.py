"""
Transformer Encoder
"""

# TODO Reimplement

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import math


class Residual(nn.Module):
    def __init__(self, fn, with_avg_pooling=False, kernel_size=2):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn, with_avg_pooling=False, kernel_size=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, dim, no_of_blocks, mlp_dim, attention, dropout = 0.8, with_avg_pooling=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(no_of_blocks):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, attention, with_avg_pooling=with_avg_pooling), with_avg_pooling=with_avg_pooling),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))

    def forward(self, x, idx = None):
        for attn, ff in self.layers:
            x = attn(x, idx = idx)
            x = ff(x)
        return x
