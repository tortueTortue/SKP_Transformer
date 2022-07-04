"""
Adapted from https://github.com/lukemelas/simple-bert
"""
 
from importlib_metadata import requires
import numpy as np
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
from torch.autograd import grad

import torch

from torch.nn.parameter import Parameter

import math
from training.metrics.metrics import print_accuracy

from training.utils.utils import get_default_device, to_device
    


def to_indices(tensor):
    return tensor.detach().type(torch.long)

def take_x(tensor, index):
    a,b,c = tensor.shape
    return tensor.view(a,c,b)[index].view(a,b,c)

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
    def __init__(self, dim, num_heads, dropout, no_of_imgs, no_of_patches, sigma=1):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.no_of_imgs = no_of_imgs
        self.no_of_patches = no_of_patches
        self.grid_dim = np.sqrt(no_of_patches)
        self.avgs = Parameter(torch.zeros(no_of_imgs, 2, no_of_patches, requires_grad=True, dtype=torch.float32)) # no_of_imgs * 2 (x and y)
        self.std_devs = Parameter(torch.ones(no_of_imgs, 2, no_of_patches, requires_grad=True, dtype=torch.float32))# no_of_imgs * 2 (x and y)
        self.sigma = sigma
        self.temperature_att_sc = 0.01


    def forward_no_loop(self, x, img_ids, mask):
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
        batch_size, _, dim = x.shape
        att = []

        # Load on GPU
        self.cuda_avgs = Parameter(self.avgs[img_ids].cuda(), requires_grad=True,)
        self.cuda_std_devs = Parameter(self.std_devs[img_ids].cuda(), requires_grad=True)

        norm_x = torch.normal(mean=torch.zeros(batch_size, 1, self.no_of_patches, requires_grad=True), std=self.sigma * torch.ones(1, self.no_of_patches, requires_grad=True)).cuda()
        norm_y = torch.normal(mean=torch.zeros(batch_size, 1, self.no_of_patches, requires_grad=True), std=self.sigma * torch.ones(1, self.no_of_patches, requires_grad=True)).cuda()
        
        avg_b, _, avg_amnt = self.cuda_avgs.shape
        avgs_x = torch.tensor_split(self.cuda_avgs, avg_amnt, dim=1)[0]
        avgs_y = torch.tensor_split(self.cuda_avgs, avg_amnt, dim=1)[1]
        stds_x = torch.tensor_split(self.cuda_std_devs, avg_amnt, dim=1)[0]
        stds_y = torch.tensor_split(self.cuda_std_devs, avg_amnt, dim=1)[1]
        
        sample_x = torch.tanh((norm_x + avgs_x) * stds_x)
        sample_y = torch.tanh((norm_y + avgs_y) * stds_y)

        grid_dim = int(self.grid_dim)
        grid = torch.reshape(torch.cat((sample_x, sample_y), dim=1), (batch_size, grid_dim, grid_dim, 2))

        # k_ce = k[0:batch_size, :1, :] # Class embeddings, not used, figure out later what to do
        # v_ce = v[0:batch_size, :1, :]
        k = k[0:batch_size, 1:, :]
        v = v[0:batch_size, 1:, :]

        k_input = torch.reshape(torch.transpose(k, dim0=1, dim1=2), (batch_size, dim, grid_dim, grid_dim))
        v_input = torch.reshape(torch.transpose(v, dim0=1, dim1=2), (batch_size, dim, grid_dim, grid_dim))

        sampled_key = F.grid_sample(k_input, grid, mode='bilinear', padding_mode='zeros')
        sampled_value = F.grid_sample(v_input, grid, mode='bilinear', padding_mode='zeros')

        # Swap back
        sampled_key = torch.transpose(torch.reshape(sampled_key, (batch_size, dim, grid_dim * grid_dim)), dim0=1, dim1=2)
        sampled_value = torch.transpose(torch.reshape(sampled_value, (batch_size, dim, grid_dim * grid_dim)), dim0=1, dim1=2)

        class_embedding = to_device(torch.ones(batch_size, 1, dim), get_default_device())
        sampled_key = torch.cat((class_embedding, sampled_key), dim=1)
        sampled_value = torch.cat((class_embedding, sampled_value), dim=1)

        attention_scores = torch.sum(sampled_key * q, dim=-1)
        attention = torch.sigmoid(self.temperature_att_sc * attention_scores).unsqueeze(dim=2) * sampled_value

        return attention

    def forward_no_loop_no_class_embeddings(self, x, img_ids, mask):
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
        batch_size, _, dim = x.shape

        # Load on GPU
        self.cuda_avgs = Parameter(self.avgs[img_ids].cuda(), requires_grad=True,)
        self.cuda_std_devs = Parameter(self.std_devs[img_ids].cuda(), requires_grad=True)

        norm_x = torch.normal(mean=torch.zeros(batch_size, 1, self.no_of_patches, requires_grad=True), std=self.sigma * torch.ones(1, self.no_of_patches, requires_grad=True)).cuda()
        norm_y = torch.normal(mean=torch.zeros(batch_size, 1, self.no_of_patches, requires_grad=True), std=self.sigma * torch.ones(1, self.no_of_patches, requires_grad=True)).cuda()
        
        avg_b, _, avg_amnt = self.cuda_avgs.shape
        avgs_x = torch.tensor_split(self.cuda_avgs, avg_amnt, dim=1)[0]
        avgs_y = torch.tensor_split(self.cuda_avgs, avg_amnt, dim=1)[1]
        stds_x = torch.tensor_split(self.cuda_std_devs, avg_amnt, dim=1)[0]
        stds_y = torch.tensor_split(self.cuda_std_devs, avg_amnt, dim=1)[1]
        
        sample_x = torch.tanh((norm_x + avgs_x) * stds_x)
        sample_y = torch.tanh((norm_y + avgs_y) * stds_y)

        grid_dim = int(self.grid_dim)
        grid = torch.reshape(torch.cat((sample_x, sample_y), dim=1), (batch_size, grid_dim, grid_dim, 2))


        k = torch.reshape(torch.transpose(k, dim0=1, dim1=2), (batch_size, dim, grid_dim, grid_dim))
        v = torch.reshape(torch.transpose(v, dim0=1, dim1=2), (batch_size, dim, grid_dim, grid_dim))

        sampled_key = F.grid_sample(k, grid, mode='bilinear', padding_mode='zeros')
        sampled_value = F.grid_sample(v, grid, mode='bilinear', padding_mode='zeros')

        # Swap back
        sampled_key = torch.transpose(torch.reshape(sampled_key, (batch_size, dim, grid_dim * grid_dim)), dim0=1, dim1=2)
        sampled_value = torch.transpose(torch.reshape(sampled_value, (batch_size, dim, grid_dim * grid_dim)), dim0=1, dim1=2)

        attention_scores = torch.sum(sampled_key * q, dim=-1)
        attention = torch.sigmoid(self.temperature_att_sc * attention_scores).unsqueeze(dim=2) * sampled_value

        return attention
    
    def forward(self, x, img_ids, mask):

        return self.forward_no_loop_no_class_embeddings(x, img_ids, mask)
    

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
    def __init__(self, dim, num_heads, ff_dim, dropout, no_of_imgs, no_of_patches, sigma=1):
        super().__init__()
        self.attn = GaussianSelfAttention(dim, num_heads, dropout, no_of_imgs, no_of_patches, sigma=sigma)
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
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, no_of_imgs_for_training = 50000, no_of_patches = 16 *16, sigma=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout, no_of_imgs_for_training, no_of_patches, sigma=sigma) for _ in range(num_layers)])

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

    def propagate_attention(self, lr, indexes, momentum): #TODO Try with momentum
        for block in self.blocks:

            with torch.no_grad():
                block.attn.cuda_avgs -= lr * block.attn.cuda_avgs.grad
                block.attn.cuda_std_devs -= lr * block.attn.cuda_std_devs.grad
                
                block.attn.avgs[indexes] = block.attn.cuda_avgs.cpu()
                block.attn.std_devs[indexes] = block.attn.cuda_std_devs.cpu()




    def log_gaussian(self, debug=True, no_of_imgs=3):
        for i in range(no_of_imgs):
            if debug:
                first_block = self.blocks[0].attn
                print(f"First block normal params for picture {i} avg : {first_block.avgs[i]} std_dev : {first_block.std_devs[i]}")



