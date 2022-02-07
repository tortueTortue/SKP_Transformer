import torch
from torch.nn.parameter import Parameter
from torch.nn import Module, CrossEntropyLoss


class Block(Module):
    
    def __init__(self):
        super().__init__()
        self.a = Parameter(torch.ones(10, requires_grad=True))
    
    def forward(x):
        return self.a * x

cross_entropy = CrossEntropyLoss()
a = Block()
out = a(torch.ones(10))

print(f"A {a}")