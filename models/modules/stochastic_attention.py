import torch
from torch import nn
from einops.layers.torch import Rearrange
from torch.distributions.categorical import Categorical

import math

class StochasticAttention(nn.Module):
    def __init__(self, no_of_features, feature_dim, no_of_images):
        super(StochasticAttention, self).__init__()
        
        # Nombre de photos * 64 * 64
        self.attention_scores = [torch.ones(no_of_features, no_of_features) for _ in range(no_of_images)]
        # self.attention_scores = [torch.ones(no_of_features, no_of_features, dtype=torch.float16) for _ in range(no_of_images)]

        self.w_query = nn.Linear(no_of_features, feature_dim, bias=False)
        self.w_key = nn.Linear(no_of_features, feature_dim, bias=False)
        self.w_value = nn.Linear(no_of_features, feature_dim, bias=False)

        # TODO Check
        self.soft = nn.Softmax(dim=2)

    def forward(self, x, idx=[]):
        batch_size, no_of_features, feature_size = x.shape
        # b, 64, 64
        # b, f, d
        query = self.w_query(x)
        key = self.w_key(x)
        value = self.w_value(x)

        
        current_attention_scores = []

        # Take keys and put em in a matrix
        # Matmul query keys in order
        # Fetch saved q*k attention and replace key_ids post by new value
        for i, curr_id in enumerate(idx): # for every picture
            key_ids = Categorical(logits=self.attention_scores[curr_id.item()]).sample() 
            for j, key_id in enumerate(key_ids): # for every query
                self.attention_scores[curr_id][j][key_id] = torch.matmul(query[i][j], key[i][key_id]) 
            current_attention_scores.append(self.attention_scores[curr_id].unsqueeze(dim=0))

        # Shape is supposed to be (b, f)
        current_attention_scores = self.soft(torch.cat(current_attention_scores).cuda())
        
        
        # Use attention score for attention
        attention = torch.einsum('b q s , b v d -> b q v', current_attention_scores, value)

        return attention


