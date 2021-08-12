"""
Adapted from https://github.com/lukemelas/simple-bert
"""
 
import numpy as np
from torch import nn
from torch import Tensor 
from torch.nn import functional as F

from torch.nn.parameter import Parameter

import math


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class GaussianSelfAttention(nn.Module):
    """Attention"""
    def __init__(self, dim, num_heads, dropout, no_of_imgs, no_of_patches):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        # self.scores = None
        self.grid_dim = np.sqrt(no_of_patches)
        self.avgs = Parameter(no_of_imgs, 2, no_of_patches) # no_of_imgs * 2 (x and y)
        self.std_devs = Parameter(no_of_imgs, 2, no_of_patches)# no_of_imgs * 2 (x and y)

    def forward(self, x, mask, img_ids):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        #Algo
        """
        1. Find patch index in x and y of the key we want for each query using normal dis

        """

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        # q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])


        scores = torch.zeros()

        att = []

        # for j, img_id in enumerate(img_ids):
        #     indexes = list
        #     att_patch = []
        #     for i in range(self.grid_dim ** 2):
        #         # Note 1: The samples are continuous
        #         # Note 2: Need to find a way to sample for every image from batch at once
        #         key_x = torch.normal(mean=self.avgs[img_id][0][i], std=self.std_devs[img_id][0][i])
        #         key_y = torch.normal(mean=self.avgs[img_id][1][i], std=self.std_devs[img_id][1][i])

        #         key_x_1 = torch.ceil(key_x)
        #         key_x_2 = torch.floor(key_x)
        #         key_y_1 = torch.ceil(key_y)
        #         key_y_2 = torch.floor(key_y)
        #         # 256 * 256
        #         # 16 * 16 --> s

        #         key_index = []
        #         key_index[0] = self.grid_dim * key_y_1 + key_x_1 
        #         key_index[1] = self.grid_dim * key_y_1 + key_x_2 
        #         key_index[2] = self.grid_dim * key_y_2 + key_x_1 
        #         key_index[3] = self.grid_dim * key_y_2 + key_x_2 

                
        #         sampled_keys = torch.stack((k[key_index[0]], k[key_index[1]], k[key_index[2]], k[key_index[3]]) #4 * 256
        #         sampled_values = torch.stack((v[key_index[0]], v[key_index[1]], v[key_index[2]], v[key_index[3]])) #4 * 256
        #         att_patch.append(F.softmax(q[j][i] * sampled_keys, dim=1) * sampled_values)
            
        #     att.stack(att_patch)
        #     att_patch = []

        # return torch.stack(att)

        # att = []

        for j in img_ids:
            indexes = list
        
            # 256
            key_x = torch.normal(mean=self.avgs[img_id][0], std=self.std_devs[img_id][0])
            key_y = torch.normal(mean=self.avgs[img_id][1], std=self.std_devs[img_id][1])

            key_x_1 = torch.ceil(key_x)
            key_x_2 = torch.floor(key_x)
            key_y_1 = torch.ceil(key_y)
            key_y_2 = torch.floor(key_y)


            key_index = []
            key_index[0] = self.grid_dim * key_y_1 + key_x_1 
            key_index[1] = self.grid_dim * key_y_1 + key_x_2 
            key_index[2] = self.grid_dim * key_y_2 + key_x_1 
            key_index[3] = self.grid_dim * key_y_2 + key_x_2 

            #k - b, 256 * 256

            # k -> b * 256 * 256 --> k[j] -> 256 * 256, k[j][1] --> 256 

            # Error n2 --> On veut 256 * 4 * 256, donc pour 256 queries, on veut 4 key de dimensions 256
            sampled_keys = torch.stack((k[j][key_index[0]], k[j][key_index[1]], 
                                        k[j][key_index[2]], k[j][key_index[3]])).transpose(dim0=0, dim1=1)#4 * 256 * 256
            sampled_values = torch.stack((v[j][key_index[0]], v[j][key_index[1]], 
                                          v[j][key_index[2]], v[j][key_index[3]])).transpose(dim0=0, dim1=1)#4 * 256 * 256

            # q -> b * 256 *256 ---> q[j] -> 256*256
            # sampled_keys 4 * 256 * 256
            # 4 --> keys * q
            # q[j] * 
            att.stack(F.softmax(q[j] * sampled_keys, dim=1) * sampled_values)


        return torch.stack(att)



        """
        For each query, dot product with one key 
        """

        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores

        return h



class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout, no_of_imgs, no_of_patches):
        super().__init__()
        self.attn = GaussianSelfAttention(dim, num_heads, dropout, no_of_imgs, no_of_patches)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.drop(self.proj(self.attn(self.norm1(x), mask)))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, no_of_imgs_for_training = 50000, no_of_patches = 16 *16):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout, no_of_imgs_for_training, no_of_patches) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x


def propagate_attention(model, lr, indexes, momentum):
    for block in model.blocks:
        block.attn.avgs[indexes] -= lr * block.attn.avgs.gradient[indexes]
        block.attn.std_devs[indexes] -= lr * block.attn.std_devs.gradient[indexes]
