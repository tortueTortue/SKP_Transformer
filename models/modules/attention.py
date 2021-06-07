import torch
from torch import nn
from einops.layers.torch import Rearrange
from torch.distributions.categorical import Categorical

import math

class Attention(nn.Module):
    def __init__(self, no_of_features, feature_dim, no_of_images):
        super(Attention, self).__init__()

        self.w_query = nn.Linear(no_of_features, feature_dim, bias=False)
        self.w_key = nn.Linear(no_of_features, feature_dim, bias=False)
        self.w_value = nn.Linear(no_of_features, feature_dim, bias=False)

        # TODO Check
        self.soft = nn.Softmax(dim=1)

    def forward(self, x, idx=[]):
        batch_size, no_of_features, feature_size = x.shape
        # b, 64, 64
        # b, f, d
        query = self.w_query(x).cuda()
        key = self.w_key(x).cuda()
        value = self.w_value(x).cuda()

        #TODO Verify formula
        # Shape is supposed to be (b, f)
        attention_scores = self.soft(torch.einsum('b f d , b f d -> b f', query, key))
        
        
        # Use attention score for attention
        attention = torch.einsum('b s , b v d -> b v d', attention_scores, value)

        return attention


