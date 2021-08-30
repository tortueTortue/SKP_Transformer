import torch
from torch import nn

import math

class NoAttention(nn.Module):
    def __init__(self, no_of_features, feature_dim, no_of_images):
        super(NoAttention, self).__init__()

    def forward(self, x, idx=[]):

        return x


