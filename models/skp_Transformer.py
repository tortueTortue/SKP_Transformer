"""
                                           
1. Give all images an ID
(img) --> [Conv] --> [Conv] --> [Conv] --> 

[                               A T T E N T I O N                                   ]
[Query, Key, Value] --> Pick one key, set other to [1], compute softmax and save key]

"""
import torch.nn as nn

from models.transformer_encoder import Encoder
from models.modules.stochastic_attention import StochasticAttention

class SKP_Transformer(nn.Module):
    """
    Local Attention Encoder
    """
    def __init__(self, no_of_blocks, no_of_images, num_classes):
    # def __init__(self, feature_size, no_of_blocks, max_pooling_blocks=3, dropout = 0., memory_block_size=4, kernel_size=2):
        super(SKP_Transformer, self).__init__()
        dim = 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=7, stride=2,
                        padding=3, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        # nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # TODO compute feature dim automatically
        attention = StochasticAttention(dim, dim, no_of_images)
        self.encoder = Encoder(dim, no_of_blocks, dim, attention)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x, idx):
        x = self.conv(x)
        batch_size, no_of_features, _, _ = x.shape
        x = self.encoder(x.view(batch_size, no_of_features,-1), idx =idx)

        x = x[:, 0]
        x = self.to_latent(x)

        return self.mlp_head(x)