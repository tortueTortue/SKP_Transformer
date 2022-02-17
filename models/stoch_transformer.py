"""
Adapted from https://github.com/lukemelas/simple-bert
"""
 
import numpy as np
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
from torch.autograd import grad

import torch

from torch.nn.parameter import Parameter

import math

from training.utils.utils import get_default_device, to_device

def to_indices(tensor):
    return tensor.detach().type(torch.long)

def bilinear(p, s):
    """
        (1 - abs(Pn_x - Sample_x)) * (1 - abs(Pn_y - Sample_y))
    """
    return (1 - abs(p[0] - s[0])) * (1 - abs(p[1] - s[1])).squeeze(0)

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
        self.no_of_imgs = no_of_imgs
        self.no_of_patches = no_of_patches
        self.grid_dim = np.sqrt(no_of_patches)
        self.avgs = Parameter(torch.zeros(no_of_imgs, 2, no_of_patches, requires_grad=True)) # no_of_imgs * 2 (x and y)
        self.std_devs = Parameter(torch.ones(no_of_imgs, 2, no_of_patches, requires_grad=True))# no_of_imgs * 2 (x and y)


    def forward(self, x, img_ids, mask):
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
        att = []

        # Load on GPU
        self.cuda_avgs = Parameter(self.avgs[img_ids].cuda(), requires_grad=True)
        self.cuda_std_devs = Parameter(self.std_devs[img_ids].cuda(), requires_grad=True)

        for j, img_id in enumerate(img_ids):
            norm_x = torch.normal(mean=torch.zeros(1, self.no_of_patches, requires_grad=True), std=torch.ones(1, self.no_of_patches, requires_grad=True)).cuda()
            norm_y = torch.normal(mean=torch.zeros(1, self.no_of_patches, requires_grad=True), std=torch.ones(1, self.no_of_patches, requires_grad=True)).cuda()
            key_x = (norm_x - self.cuda_avgs[j][0])/ self.cuda_std_devs[j][0]
            key_y = (norm_y - self.cuda_avgs[j][1])/ self.cuda_std_devs[j][1]

            key_x_1 = torch.ceil(key_x)
            key_x_2 = torch.floor(key_x)
            key_y_1 = torch.ceil(key_y)
            key_y_2 = torch.floor(key_y)


            key_index = [0,0,0,0]
            key_index[0] = to_indices(self.grid_dim * key_y_1 + key_x_1)
            key_index[1] = to_indices(self.grid_dim * key_y_1 + key_x_2)
            key_index[2] = to_indices(self.grid_dim * key_y_2 + key_x_1)
            key_index[3] = to_indices(self.grid_dim * key_y_2 + key_x_2)


            # TODO Refactor this : compute once bilinear for both values and keys, muultiply key and val once
            # SAMPLED KEY = E{ (1 - abs(Pn_x - Sample_x)) * (1 - abs(Pn_y - Sample_y)) * Kn
            sample = (key_x, key_y)
            sampled_key = (bilinear((key_x_1 , key_y_1), sample) * k[j][key_index[0]].transpose(dim0=1, dim1=2) + \
                           bilinear((key_x_2 , key_y_1), sample) * k[j][key_index[1]].transpose(dim0=1, dim1=2) + \
                           bilinear((key_x_1 , key_y_2), sample) * k[j][key_index[2]].transpose(dim0=1, dim1=2) + \
                           bilinear((key_x_2 , key_y_2), sample) * k[j][key_index[3]].transpose(dim0=1, dim1=2)).transpose(dim0=1, dim1=2)

            sampled_value = (bilinear((key_x_1 , key_y_1), sample) * v[j][key_index[0]].transpose(dim0=1, dim1=2) + \
                             bilinear((key_x_2 , key_y_1), sample) * v[j][key_index[1]].transpose(dim0=1, dim1=2) + \
                             bilinear((key_x_1 , key_y_2), sample) * v[j][key_index[2]].transpose(dim0=1, dim1=2) + \
                             bilinear((key_x_2 , key_y_2), sample) * v[j][key_index[3]].transpose(dim0=1, dim1=2)).transpose(dim0=1, dim1=2)
            

            # Lets add ones vector for class embedding
            _, _, k_dim = sampled_key.shape
            class_emb = to_device(torch.ones(1, 1, k_dim), get_default_device())
            sampled_key = torch.cat((class_emb, sampled_key), dim=1)
            sampled_value = torch.cat((class_emb, sampled_value), dim=1)

            at_sc = torch.matmul(sampled_key.transpose(dim0=0, dim1=1), q[j].unsqueeze(dim=2))
            full_att = F.softmax(at_sc, dim=1).transpose(dim0=0, dim1=1) * sampled_value.squeeze(dim=0)
            
            att.append(torch.sum(full_att, dim=0)) 


        return torch.stack(att)

#TODO : Add test mode without estimation


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

    def forward(self, x, ids, mask):
        h = self.drop(self.proj(self.attn(self.norm1(x), ids, mask)))
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

    def forward(self, x, ids, mask=None):
        for block in self.blocks:
            x = block(x, ids, mask)
        return x

    def compute_gradients(self, loss, indexes):
        for block in self.blocks:
            block.attn.avgs[indexes].retain_grad()
            block.attn.avgs[indexes].retain_graph = True
            block.attn.avgs[indexes].grad = grad(loss, block.attn.avgs[indexes])

            block.attn.std_devs[indexes].retain_grad()
            block.attn.std_devs[indexes].retain_graph = True
            block.attn.std_devs[indexes].grad = grad(loss, block.attn.std_devs[indexes])

    def propagate_attention(self, lr, indexes, momentum):
        for block in self.blocks:

            with torch.no_grad():
                block.attn.cuda_avgs -= lr * block.attn.cuda_avgs.grad
                block.attn.cuda_std_devs -= lr * block.attn.cuda_std_devs.grad
                block.attn.avgs[indexes] = block.attn.cuda_avgs.cpu()
                block.attn.std_devs[indexes] = block.attn.cuda_std_devs.cpu()
