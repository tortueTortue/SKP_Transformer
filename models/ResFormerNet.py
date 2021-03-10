"""
 Architecture a ResNet18 to Extract a representation then sent to a patch tokenizer and 
 a Transformer encoder of 3 layers and an MLP.
 """

import torchvision.models as models


# Instantiate ResNet
resnet18 = models.resnet18()

# Verify layers
resnet18.summary()

# Remove last layers
resnet18_last_layer_removed = list(resnet18.children())[:-1]
resnet18_last_layer_removed = torch.nn.Sequential(*self.resnet18_last_layer_removed)

# Verify layers
resnet18_last_layer_removed.summary()

# Instantiate Encoder

# Instantiate Patches tokeniser
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py

# Instantiate or linear

# Build model

