from __future__ import print_function
import torch

# x = torch.empty(5, 3)
# y = torch.rand(5, 3)
# z = torch.zeros(5, 3, dtype=torch.long)
# k = torch.tensor([5.5, 3])
# i = k.new_ones(5, 3, dtype=torch.double)
# j = torch.randn_like(i, dtype=torch.float)
   
# print(x)
# print(y)
# print(z)
# print(k)
# print(i)
# print(j)

# if torch.cuda.is_available():
#     device = torch.device("cuda")          # a CUDA device object
#     y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
#     x = x.to(device)                       # or just use strings ``.to("cuda")``
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))   

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)