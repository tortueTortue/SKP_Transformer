import torch

def to_indices(tensor):
    return tensor.detach().type(torch.long)

def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)