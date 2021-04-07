import torch
from torch import nn
from einops.layers.torch import Rearrange

import math

class StochasticAttention(nn.Module):
    def __init__(self, no_of_features, feature_dim, no_of_images):
        super(StochasticAttention, self).__init__()
        self.sqrt_dim = math.sqrt(feature_dim)
        self.attention_scores = [torch.ones(no_of_features) / self.sqrt_dim for i in range(no_of_images)]

        self.w_query = nn.Linear(no_of_features, feature_dim, bias=False)
        self.w_key = nn.Linear(no_of_features, feature_dim, bias=False)
        self.w_value = nn.Linear(no_of_features, feature_dim, bias=False)

        # TODO Check
        self.soft = nn.Softmax(dim=4)

    def forward(self, x, idx=[]):
        batch_size, no_of_features, feature_size = x.shape
        # b, 64, 256
        # b, f, d
        query = self.w_query(x)
        key = self.w_key(x)
        value = self.w_value(x)

        # Randomly select a key for every query
        key_ids = torch.randint(0,no_of_features,(batch_size, no_of_features)).cuda()
        
        current_attention_scores = []

        # Take keys and put em in a matrix
        # Matmul query keys in order
        # Fetch saved q*k attention and replace key_ids post by new value
        for i, curr_id in enumerate(idx): # for every picture
            for j, feat_id in enumerate(key_ids[i]): # for every query
                self.attention_scores[curr_id][j] = torch.matmul(query[i][j], key[i][feat_id]) / self.sqrt_dim

            current_attention_scores.append(self.attention_scores[curr_id].unsqueeze(dim=0))

        # Shape is supposed to be (b, f)
        current_attention_scores = torch.cat(current_attention_scores).cuda()
        
        # Use attention score for attention
        attention = torch.einsum('b f, b f d -> b f d', current_attention_scores, value)

        return attention


