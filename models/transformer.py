from pyclbr import Function
from typing import Optional
import copy

import torch
from torch import nn
from torch.nn import functional as F

from models.modules.gaussian_attention import GaussianSelfAttention
from models.modules.stochastic_attention import StochSelfAttention
from models.modules.self_attention import MultiHeadedSelfAttention
from models.modules.no_attention import NoAttention

from training.utils.utils import get_default_device, to_device
from models.tools.tools import as_tuple

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
    def __init__(self, dim, ff_dim, dropout, attn):
        super().__init__()
        self.attn = copy.deepcopy(attn)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, indices: Optional[torch.Tensor] = None):
        attn_out = self.attn(self.norm1(x), indices)
        h = self.drop(self.proj(attn_out))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x


class Transformer(nn.Module):
    def __init__(self, num_layers, dim, ff_dim, dropout, attn = NoAttention()):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, ff_dim, dropout, attn) for _ in range(num_layers)])

    def forward(self, x, indices: Optional[torch.Tensor] = None):
        for block in self.blocks:
            x = block(x, indices = indices)
        return x

class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
    
    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding


class StochViT(nn.Module):
    """
    Args:
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
        self,
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        positional_embedding: str = '1d',
        in_channels: int = 3, 
        image_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        no_of_imgs_for_training: int = 50000,
        sigma: float = 1,
        attention_type: str = 'Normal',
    ):
        super().__init__()

        if num_classes is None:
            num_classes = 1000
        if image_size is None:
            image_size = 384
        
        self.image_size = image_size
        self.attention_type = attention_type

        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw

        # Attention type
        assert attention_type == 'Gaussian' or attention_type == 'SamplingNetwork' or \
                attention_type == 'None' or attention_type == 'Normal', f"The attention type {attention_type} does not exist!"

        if attention_type == 'Gaussian':
            attention = GaussianSelfAttention(dim, num_heads, no_of_imgs_for_training, patches**2, sigma)
        elif attention_type == 'SamplingNetwork':
            attention = StochSelfAttention(dim, num_heads)
        elif attention_type == 'Normal':
            attention = MultiHeadedSelfAttention(dim, num_heads, attention_dropout_rate)
        else:
            attention = NoAttention()

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))

        # Class token
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim, device='cuda'))
            seq_len += 1
        
        # Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()
        
        self.transformer = Transformer(num_layers=num_layers, dim=dim, ff_dim=ff_dim, dropout=dropout_rate, attn=attention)
        
        # Representation layer
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
        else:
            pre_logits_size = dim

        # Classifier head
        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
        self.fc = nn.Linear(pre_logits_size, num_classes)

        # Initialize weights
        self.init_weights()
        
        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        if hasattr(self, 'class_token'):
            nn.init.constant_(self.class_token, 0)

    def forward(self, x, indices: Optional[torch.Tensor] = None):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'): 
            x = self.positional_embedding(x)  # b,gh*gw+1,d 
        x = self.transformer(x, indices=indices)  # b,gh*gw+1,d
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            x = self.norm(x)[:, 0]  # b,d
            x = self.fc(x)  # b,num_classes
        return x

    def backpropagate_attention(self, lr, indices, momentum:float = None):
        assert self.attention_type == 'Gaussian', "This method is made for Gaussian Attention Transformer."

        for block in self.transformer.blocks:
            with torch.no_grad():
                # if momentum: #TODO Verify
                #     v_avgs = momentum * block.attn.cuda_avgs + \
                #                            (1 - momentum) * block.attn.cuda_avgs.grad
                #     v_std_devs = momentum * block.attn.cuda_std_devs + \
                #                                (1 - momentum) * block.attn.cuda_std_devs.grad

                #     block.attn.cuda_avgs -= lr * v_avgs
                #     block.attn.cuda_std_devs -= lr * v_std_devs
                # else:
                #     block.attn.cuda_avgs -= lr * block.attn.cuda_avgs.grad
                #     block.attn.cuda_std_devs -= lr * block.attn.cuda_std_devs.grad
                block.attn.cuda_avgs -= lr * block.attn.cuda_avgs.grad
                block.attn.cuda_std_devs -= lr * block.attn.cuda_std_devs.grad

                print(f"grads: {block.attn.cuda_std_devs}")
                block.attn.avgs[indices] = block.attn.cuda_avgs.cpu()
                block.attn.std_devs[indices] = block.attn.cuda_std_devs.cpu()


    # TODO : Reimplement this, VERY BAD PRACTICE
    def load_on_gpu(self):
        assert self.attention_type == 'Gaussian', "This method is made for Gaussian Attention Transformer."

        device = get_default_device()
        
        to_device(self.patch_embedding, device)
        to_device(self.positional_embedding, device)
        if hasattr(self, 'class_token'):
            to_device(self.class_token, device)
            self.class_token.to()
        if hasattr(self, 'pre_logits'):
            to_device(self.pre_logits, device)
        to_device(self.norm, device)
        to_device(self.fc, device)

        for block in self.transformer.blocks:
            to_device(block.drop, device)
            to_device(block.proj, device)
            to_device(block.norm1, device)
            to_device(block.drop, device)
            to_device(block.pwff, device)
            to_device(block.norm2, device)

            to_device(block.attn.proj_q, device)
            to_device(block.attn.proj_k, device)
            to_device(block.attn.proj_v, device)

        return self
    
    def log_gaussian(self):
        assert self.attention_type == 'Gaussian', "This method is made for Gaussian Attention Transformer."

        attn: GaussianSelfAttention = self.transformer.blocks[0].attn

        print("First block gaussian averages:")
        print(attn.avgs)
        print("First block gaussian std deviation:")
        print(attn.std_devs)


def end_of_iteration_stoch_gaussian_ViT(learning_rate) -> Function:
    def f(model: StochViT, indices, epoch: int = -1, iteration: int = -1):
        model.backpropagate_attention(indices=indices, lr=learning_rate)
        if epoch % 10 == 0 and iteration < 10:
            print(f"Epoch : {epoch} and iteration :{iteration}")
            model.log_gaussian()

    return f
