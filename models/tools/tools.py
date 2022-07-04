import torch

def to_indices(tensor):
    return tensor.detach().type(torch.long)