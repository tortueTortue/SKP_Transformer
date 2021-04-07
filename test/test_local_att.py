import torch
from torch import nn
from einops.layers.torch import Rearrange

import math

cifar_10_batch = torch.rand((1,3,32,32))

small_pic_batch = torch.ones((1,1,6,6))
small_pic_batch = small_pic_batch.view((1,1,6*6))

w_query = nn.Linear(6*6,6*6, bias=False)
w_key = nn.Linear(6*6,6*6, bias=False)
w_value = nn.Linear(6*6,6*6, bias=False)

w_query.eval()
w_key.eval()
w_value.eval()

# att 
soft = nn.Softmax(dim=4)

mem_blocks_divider = Rearrange('b c (h p1) (w p2) -> b c (h w) (p1 p2)', p1=2,p2=2)

# TODO Optimize
query =  mem_blocks_divider(w_query(small_pic_batch).view((1,1,6,6))).view((1,1,9,4,1))
key   =  mem_blocks_divider(w_key(small_pic_batch).view((1,1,6,6)))
value =  mem_blocks_divider(w_value(small_pic_batch).view((1,1,6,6))).view((1,1,9,4,1))

# q(i,j) * k(a,b) where i,j are pixel col and row AND a,b are memory block size
att_score = soft(torch.einsum('b c n g q, b c n p ->b c n g p', query, key))

# print(torch.matmul(att_score, value).view(1,1,9*4))

# avg pooling

def modif(q,k,v):
    q[0][0][0][0] = 0.81
    q[0][0][0][1] = 0.6
    q[0][0][0][2] = 0.54
    q[0][0][0][3] = 0.6

    k[0][0][0][0] = 1.08
    k[0][0][0][1] = 0.8
    k[0][0][0][2] = 0.72
    k[0][0][0][3] = 0.8

    v[0][0][0][0] = 1.35
    v[0][0][0][1] = 1
    v[0][0][0][2] = 1
    v[0][0][0][3] = 20


#### TEST

x = torch.ones((10,3,6,6))
x = x.view((10,3,6*6))

x[0][0][0] = 0.27
x[0][0][1] = 0.2
x[0][0][6] = 0.18
x[0][0][7] = 0.2

local_att = LocalAttention(6*6, 6*6, 2, 2)
# local_att.eval()

local_att.w_query.weight.data.fill_(3)
local_att.w_key.weight.data.fill_(4)
local_att.w_value.weight.data.fill_(5)

print("x")
print(x[0][0].view((6,6)))

print("att")
print(local_att(x)[0][0][0])