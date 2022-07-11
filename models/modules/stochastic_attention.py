from torch import nn
from torch.nn import  functional as F
import torch
import numpy as np

from training.utils.utils import get_default_device, to_device

class KeyFinderNet(nn.Module):
    def __init__(self, dim, mode='1D'):
        super().__init__()
        assert mode == '1D' or mode == '2D', f"Mode {mode}, does not exist, it has to be either 1D or 2D."

        self.fc1 = nn.Linear(dim*2, 60)
        self.fc2 = nn.Linear(60, 60)
        if mode == '1D':
            self.fc3 = nn.Linear(60, 1)
        elif mode == '2D':
            self.fc3 = nn.Linear(60, 2)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)

        return out

class StochSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.n_heads = num_heads
        self.key_net = KeyFinderNet(dim)
        self.key_net_2D = KeyFinderNet(dim, mode='2D')
        self.temperature_att_sc = 0.01

    def grid_sample_forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        batch_size, no_of_patches, dim = x.shape

        grid_dim = int(np.sqrt(no_of_patches))

        k = k[0:batch_size, 1:, :]
        v = v[0:batch_size, 1:, :]
        q_no_ce = q[0:batch_size, 1:, :]

        k_input = torch.reshape(torch.transpose(k, dim0=1, dim1=2), (batch_size, dim, grid_dim, grid_dim))
        v_input = torch.reshape(torch.transpose(v, dim0=1, dim1=2), (batch_size, dim, grid_dim, grid_dim))

        sample = self.key_net_2D(torch.concat((q_no_ce,k), dim=2))
        sample_x = torch.tensor_split(sample, no_of_patches, dim=2)[0]
        sample_y = torch.tensor_split(sample, no_of_patches, dim=2)[1]

        grid = torch.reshape(torch.cat((sample_x, sample_y), dim=1), (batch_size, grid_dim, grid_dim, 2))

        sampled_key = F.grid_sample(k_input, grid, mode='bilinear', padding_mode='zeros')
        sampled_value = F.grid_sample(v_input, grid, mode='bilinear', padding_mode='zeros')

        # Swap back
        sampled_key = torch.transpose(torch.reshape(sampled_key, (batch_size, dim, grid_dim * grid_dim)), dim0=1, dim1=2)
        sampled_value = torch.transpose(torch.reshape(sampled_value, (batch_size, dim, grid_dim * grid_dim)), dim0=1, dim1=2)

        class_embedding = to_device(torch.ones(batch_size, 1, dim), get_default_device())
        sampled_key = torch.cat((class_embedding, sampled_key), dim=1)
        sampled_value = torch.cat((class_embedding, sampled_value), dim=1)

        attention_scores = torch.sum(q * sampled_key, dim=-1)
        attention = torch.sigmoid(self.temperature_att_sc * attention_scores).unsqueeze(dim=2) * sampled_value

        return attention

    def grid_sample_forward_no_class_embedding(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        batch_size, no_of_patches, dim = x.shape

        grid_dim = int(np.sqrt(no_of_patches))

        sample = self.key_net_2D(torch.concat((q,k), dim=2))
        sample_x = torch.tensor_split(sample, no_of_patches, dim=2)[0]
        sample_y = torch.tensor_split(sample, no_of_patches, dim=2)[1]

        grid = torch.reshape(torch.cat((sample_x, sample_y), dim=1), (batch_size, grid_dim, grid_dim, 2))

        k = torch.reshape(torch.transpose(k, dim0=1, dim1=2), (batch_size, dim, grid_dim, grid_dim))
        v = torch.reshape(torch.transpose(v, dim0=1, dim1=2), (batch_size, dim, grid_dim, grid_dim))

        sampled_key = F.grid_sample(k, grid, mode='bilinear', padding_mode='zeros')
        sampled_value = F.grid_sample(v, grid, mode='bilinear', padding_mode='zeros')

        # Swap back
        sampled_key = torch.transpose(torch.reshape(sampled_key, (batch_size, dim, grid_dim * grid_dim)), dim0=1, dim1=2)
        sampled_value = torch.transpose(torch.reshape(sampled_value, (batch_size, dim, grid_dim * grid_dim)), dim0=1, dim1=2)

        attention_scores = torch.sum(q * sampled_key, dim=-1)
        attention = torch.sigmoid(self.temperature_att_sc * attention_scores).unsqueeze(dim=2) * sampled_value

        return attention

    def forward(self, x, indices):
        return self.grid_sample_forward_no_class_embedding(x)