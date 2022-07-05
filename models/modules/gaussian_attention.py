from torch import nn
from torch.nn import functional as F, Parameter
import torch
import numpy as np

class GaussianSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, no_of_imgs, no_of_patches, sigma=1):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.n_heads = num_heads
        self.no_of_imgs = no_of_imgs
        self.no_of_patches = no_of_patches
        self.grid_dim = np.sqrt(no_of_patches)
        self.avgs = Parameter(torch.zeros(no_of_imgs, 2, no_of_patches, requires_grad=True, dtype=torch.float32))
        self.std_devs = Parameter(torch.ones(no_of_imgs, 2, no_of_patches, requires_grad=True, dtype=torch.float32))
        self.sigma = sigma
        self.temperature_att_sc = 0.01

    def forward_(self, x, img_ids, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
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

    def forward_no_class_embed(self, x, img_ids, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
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
        return self.forward_no_class_embed(x, img_ids, mask)