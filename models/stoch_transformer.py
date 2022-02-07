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
        # self.scores = None
        self.grid_dim = np.sqrt(no_of_patches)
        # TODO force avgs and std out of gpu
        self.avgs = Parameter(torch.zeros(no_of_imgs, 2, no_of_patches, requires_grad=True)) # no_of_imgs * 2 (x and y)
        self.std_devs = Parameter(torch.ones(no_of_imgs, 2, no_of_patches, requires_grad=True))# no_of_imgs * 2 (x and y)
        # self.avgs.retain_grad()
        # self.std_devs.retain_grad()

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
        # q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])

        att = []

        
        print("self.avgs is on " + "cuda" if self.avgs.is_cuda else "cpu")

        # Load on GPU
        self.cuda_avgs = Parameter(self.avgs[img_ids].cuda(), requires_grad=True)
        self.cuda_std_devs = Parameter(self.std_devs[img_ids].cuda(), requires_grad=True)
        # self.cuda_avgs.requires_grad_(True)
        # self.cuda_std_devs.requires_grad_(True)
        # avgs = self.avgs[img_ids].cuda()
        # std_devs = self.std_devs[img_ids].cuda()

        for j, img_id in enumerate(img_ids):
            indexes = list
        
            # 256
            #TODO Add self.avgs[img_id][0] and self.std_devs[img_id][1]
            
            print(f"no of patches {self.no_of_patches} no of imgs {self.no_of_imgs} id of curr img {img_id}")
            norm_x = torch.normal(mean=torch.zeros(1, self.no_of_patches, requires_grad=True), std=torch.ones(1, self.no_of_patches, requires_grad=True)).cuda()
            norm_y = torch.normal(mean=torch.zeros(1, self.no_of_patches, requires_grad=True), std=torch.ones(1, self.no_of_patches, requires_grad=True)).cuda()
            key_x = (norm_x - self.cuda_avgs[j][0])/ self.cuda_std_devs[j][0]
            key_y = (norm_y - self.cuda_avgs[j][1])/ self.cuda_std_devs[j][1]
            # key_x = (torch.normal(mean=torch.zeros(self.no_of_imgs, self.no_of_patches, requires_grad=True), std=torch.ones(self.no_of_imgs, self.no_of_patches, requires_grad=True)) - avgs[img_id][0])/ std_devs[img_id][0]
            # key_y = (torch.normal(mean=torch.zeros(self.no_of_imgs, self.no_of_patches, requires_grad=True), std=torch.ones(self.no_of_imgs, self.no_of_patches, requires_grad=True)) - avgs[img_id][1])/ std_devs[img_id][1]

            key_x_1 = torch.ceil(key_x)
            key_x_2 = torch.floor(key_x)
            key_y_1 = torch.ceil(key_y)
            key_y_2 = torch.floor(key_y)

            


            key_index = [0,0,0,0]
            key_index[0] = to_indices(self.grid_dim * key_y_1 + key_x_1)
            key_index[1] = to_indices(self.grid_dim * key_y_1 + key_x_2)
            key_index[2] = to_indices(self.grid_dim * key_y_2 + key_x_1)
            key_index[3] = to_indices(self.grid_dim * key_y_2 + key_x_2)

            #k - b, 256 * 256

            # k -> b * 256 * 256 --> k[j] -> 256 * 256, k[j][1] --> 256 
            # TODO : Use sample for class token!!!!
            # Error n2 --> On veut 256 * 4 * 256, donc pour 256 queries, on veut 4 key de dimensions 256
            sampled_keys = torch.stack((k[j][key_index[0]], k[j][key_index[1]], 
                                        k[j][key_index[2]], k[j][key_index[3]])).transpose(dim0=0, dim1=1)#4 * 256 * 256
            sampled_values = torch.stack((v[j][key_index[0]], v[j][key_index[1]], 
                                          v[j][key_index[2]], v[j][key_index[3]])).transpose(dim0=0, dim1=1)#4 * 256 * 256

            # q -> b * 256 *256 ---> q[j] -> 256*256
            # sampled_keys 4 * 256 * 256
            # 4 --> keys * q
            # q[j] * 
            #a = q[j] * sampled_keys
            # Lets add ones vector for class embedding
            print(f"sampled keys dim {sampled_keys.shape}")
            ss, n_s, n_p, p_l = sampled_keys.shape
            class_emb = to_device(torch.ones(1, n_s, 1, p_l), get_default_device())
            print(f"sampled keys dim {sampled_keys.shape} claas emb shpa {class_emb.shape}")
            sampled_keys = torch.cat((class_emb, sampled_keys), dim=2)
            print(f"sampled keys dim {sampled_keys.shape}")
            print(f"sampled vals dim {sampled_values.shape}")
            sampled_values = torch.cat((class_emb, sampled_values), dim=2)

            print(f"size q {q[j].unsqueeze(dim=2).shape}")

            at_sc = torch.matmul(sampled_keys.squeeze().transpose(dim0=0, dim1=1), q[j].unsqueeze(dim=2))
            print(f"att score shape {at_sc.shape}")
            print(f"sampled_values shape {sampled_values.shape}")
            print(f"Po shape {F.softmax(at_sc, dim=1).transpose(dim0=0, dim1=1).shape}")
            print(f"s_v shape {sampled_values.squeeze(dim=0).shape}")
            full_att = F.softmax(at_sc, dim=1).transpose(dim0=0, dim1=1) * sampled_values.squeeze(dim=0)

            
            att.append(torch.sum(full_att, dim=0)) 

        print(f"{self.avgs[img_ids]}")




        return torch.stack(att)

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
        # q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])

        att = []

        
        print("self.avgs is on " + "cuda" if self.avgs.is_cuda else "cpu")

        # Load on GPU
        self.cuda_avgs = Parameter(self.avgs[img_ids].cuda(), requires_grad=True)
        self.cuda_std_devs = Parameter(self.std_devs[img_ids].cuda(), requires_grad=True)
        # self.cuda_avgs.requires_grad_(True)
        # self.cuda_std_devs.requires_grad_(True)
        # avgs = self.avgs[img_ids].cuda()
        # std_devs = self.std_devs[img_ids].cuda()

        for j, img_id in enumerate(img_ids):
            indexes = list
        
            # 256
            #TODO Add self.avgs[img_id][0] and self.std_devs[img_id][1]
            
            print(f"no of patches {self.no_of_patches} no of imgs {self.no_of_imgs} id of curr img {img_id}")
            norm_x = torch.normal(mean=torch.zeros(1, self.no_of_patches, requires_grad=True), std=torch.ones(1, self.no_of_patches, requires_grad=True)).cuda()
            norm_y = torch.normal(mean=torch.zeros(1, self.no_of_patches, requires_grad=True), std=torch.ones(1, self.no_of_patches, requires_grad=True)).cuda()
            key_x = (norm_x - self.cuda_avgs[j][0])/ self.cuda_std_devs[j][0]
            key_y = (norm_y - self.cuda_avgs[j][1])/ self.cuda_std_devs[j][1]
            # key_x = (torch.normal(mean=torch.zeros(self.no_of_imgs, self.no_of_patches, requires_grad=True), std=torch.ones(self.no_of_imgs, self.no_of_patches, requires_grad=True)) - avgs[img_id][0])/ std_devs[img_id][0]
            # key_y = (torch.normal(mean=torch.zeros(self.no_of_imgs, self.no_of_patches, requires_grad=True), std=torch.ones(self.no_of_imgs, self.no_of_patches, requires_grad=True)) - avgs[img_id][1])/ std_devs[img_id][1]

            key_x_1 = torch.ceil(key_x)
            key_x_2 = torch.floor(key_x)
            key_y_1 = torch.ceil(key_y)
            key_y_2 = torch.floor(key_y)




            key_index = [0,0,0,0]
            key_index[0] = to_indices(self.grid_dim * key_y_1 + key_x_1)
            key_index[1] = to_indices(self.grid_dim * key_y_1 + key_x_2)
            key_index[2] = to_indices(self.grid_dim * key_y_2 + key_x_1)
            key_index[3] = to_indices(self.grid_dim * key_y_2 + key_x_2)

            #k - b, 256 * 256

            # k -> b * 256 * 256 --> k[j] -> 256 * 256, k[j][1] --> 256 
            # TODO : Use sample for class token!!!!
            # Error n2 --> On veut 256 * 4 * 256, donc pour 256 queries, on veut 4 key de dimensions 256
            sampled_keys = torch.stack((k[j][key_index[0]], k[j][key_index[1]], 
                                        k[j][key_index[2]], k[j][key_index[3]])).transpose(dim0=0, dim1=1)#4 * 256 * 256
            sampled_values = torch.stack((v[j][key_index[0]], v[j][key_index[1]], 
                                          v[j][key_index[2]], v[j][key_index[3]])).transpose(dim0=0, dim1=1)#4 * 256 * 256

            # q -> b * 256 *256 ---> q[j] -> 256*256
            # sampled_keys 4 * 256 * 256
            # 4 --> keys * q
            # q[j] * 
            #a = q[j] * sampled_keys
            # Lets add ones vector for class embedding
            print(f"sampled keys dim {sampled_keys.shape}")
            ss, n_s, n_p, p_l = sampled_keys.shape
            class_emb = to_device(torch.ones(1, n_s, 1, p_l), get_default_device())
            print(f"sampled keys dim {sampled_keys.shape} claas emb shpa {class_emb.shape}")
            sampled_keys = torch.cat((class_emb, sampled_keys), dim=2)
            print(f"sampled keys dim {sampled_keys.shape}")
            print(f"sampled vals dim {sampled_values.shape}")
            sampled_values = torch.cat((class_emb, sampled_values), dim=2)

            print(f"size q {q[j].unsqueeze(dim=2).shape}")

            at_sc = torch.matmul(sampled_keys.squeeze().transpose(dim0=0, dim1=1), q[j].unsqueeze(dim=2))
            print(f"att score shape {at_sc.shape}")
            print(f"sampled_values shape {sampled_values.shape}")
            print(f"Po shape {F.softmax(at_sc, dim=1).transpose(dim0=0, dim1=1).shape}")
            print(f"s_v shape {sampled_values.squeeze(dim=0).shape}")
            full_att = F.softmax(at_sc, dim=1).transpose(dim0=0, dim1=1) * sampled_values.squeeze(dim=0)

            
            att.append(torch.sum(full_att, dim=0)) 

        print(f"{self.avgs[img_ids]}")




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
        print(f"att shape {h.shape} and x shape {x.shape}")
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
            # if block.attn.avgs[indexes].grad is None:
            #     block.attn.avgs[indexes].retain_grad()
            #     block.attn.avgs[indexes].grad = torch.zeros_like(block.attn.avgs[indexes])
            # if block.attn.std_devs[indexes].grad is None:
            #     block.attn.std_devs[indexes].retain_grad()
            #     block.attn.std_devs[indexes].grad = torch.zeros_like(block.attn.std_devs[indexes])


            block.attn.cuda_avgs -= lr * block.attn.cuda_avgs.grad
            block.attn.cuda_std_devs -= lr * block.attn.cuda_std_devs.grad

            with torch.no_grad():
                self.avgs[indexes] = self.cuda_avgs.cpu()
                self.std_devs[indexes] = self.cuda_std_devs.cpu()

            # TODO If self.cuda_ breaks mem, set to 0

            # or try this
            # block.attn.avgs[indexes].data.sub_(block.attn.avgs[indexes].grad.data * lr)
            # block.attn.std_devs[indexes].data.sub_(block.attn.std_devs[indexes].grad.data * lr)
