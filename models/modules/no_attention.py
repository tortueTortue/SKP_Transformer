import torch
from torch import nn
from typing import Optional


class NoAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, indices: Optional[torch.Tensor] = None):
        return x