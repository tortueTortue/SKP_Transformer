import torch
from torch import nn

import math

class NoAttention(nn.Module):
    def __init__(self):
        super(NoAttention, self).__init__()

    def forward(self, x):
        return x