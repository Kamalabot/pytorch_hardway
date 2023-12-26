import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
# zeros, ones, rand, empty, randint, linspace, eye, 
# multinomial, cat, arange, unsqueeze(learn), masked_fill 
# stack, triu, tril, transpose, softmax, 
# Embedding, Linear functions, data loading
# view, broadcasting semantics, transforms
a = torch.zeros(1, 2)  # 1 row and 2 cols 
# print(a)

# ones
b = torch.ones(2, 2)  # 2 row and 2 cols
# print(b)

# rand
c = torch.rand(10, 10)  # 10 rows and 10 cols
# print(c)

# empty
d = torch.empty(5, 3)  # 5 rows and 3 cols
# print(d)

# randint
e = torch.randint(low=10, high=100,
                  size=(3, 2))  # 3 row and 2 cols random number between 10 and 100
# print(e)

# linspace
f = torch.linspace(start=0, end=13, steps=4, dtype=torch.int32)  # linear space between 0 to 7 with steps of 3
# can control the type of the output without worrying about type error
# print(f)
# can change the dimensions of linspace data with view
# eye or identity matrix
g = torch.eye(n=5, m=5)
# print(g)

# following are methods that check the efficiency or provides some easy way of manipulating the data

h = torch.rand(size=[100, 100, 100, 100])
# print(h.shape)

i = torch.rand(size=[100, 100, 100, 100])

# do torch multiplication
tm = (h @ i)
# print(h[0][0][0][0], 'h')
# print(i[0][0][0][0], 'i')
# print(tm[0][0][0][0], 'hm')
# do numpy multiplication
nm = np.multiply(h, i)
# print(nm[0][0][0][0], 'nm')

# embeddings, torch.stack, torch.multinomial, torch.tril, torch.triu
# input.T / input.transpose, nn.Linear, torch.cat, F.softmax
# unsqueeze and squeeze ( so squeeze(-1) would remove the last dimension and unsqueeze(-1) 
# would add a new dimension after the current last.)
# (show all the examples of functions/methods with pytorch docs)
j = torch.tensor([0.2, 0.3, 0.5])
samples = torch.multinomial(input=j, num_samples=10, replacement=True)
# draws samples with 0, 1, 2 
# print(samples)

k = torch.tensor([1, 2, 3, 4, 5])
l = torch.tensor([5])  # torch.Size([1])
# print(l.shape)
m = torch.tensor(5)  # this will have no size torch.Size([])
# print(m.shape)

o = torch.cat((k, l),dim=0)
# print(o)
# n = torch.cat((k, m), dim=0)  # will throw zero-dimensional tensor cannot be concatenated
# print(n)

p = torch.randint(low=5, high=10, size=(2, 3))
q = torch.randint(low=10, high=25, size=(2, 4))
# print(p)
# print(q)

r = torch.cat((p, q), dim=1)  # tensor 1 is in 2nd position, dim = 0 / 1 will work on two axes only
# print(r)

# stack expects each tensor to be equal size, but got [5] at entry 0 and [4] at entry 3
s1 = torch.arange(0, 5)  # 1 r and 5 c
s2 = torch.arange(1, 6)  # 1 r and 5 c
s3 = torch.arange(2, 7)
s4 = torch.arange(4, 9)
s5 = torch.stack((s1, s2, s3, s4))
# print(s5.shape)
# print(s5)

tl = torch.tril(torch.ones(3, 3) * 5)  # scalar int multiplication works
# print(tl)
tu = torch.triu(torch.ones(3, 3) * torch.tensor([5]))  # What happens when 
# another tensor is involved? Same result
# print(tu)

tu_try = torch.triu(torch.ones(3, 3) * torch.tensor(5))  # What happens when 
# another tensor with None size is involved? 
# print(tu_try)
# There has to be a base tensor, on which a mask is placed, and where the conditions 
# return True, update the value
maskout = torch.zeros(5, 5).masked_fill(torch.tril(torch.ones(5, 5)) == 0,
                                        float('-inf'))
# print(maskout, 'maskout')
# print(torch.exp(maskout), 'exponentiating maskout')

# print(torch.exp(torch.tensor([0])), 'mask out')
# print(torch.exp(torch.tensor([float('-inf')])), 'mask out')

# masked = torch.masked_fill(torch.zeros(5, 5), make_triu == 0, torch.tensor(67))
#  * (Tensor input, Tensor mask, Tensor value)
# print(masked)

input = torch.zeros(2, 3, 4)
# print(input.shape, 'input')

out1 = input.transpose(0, 1)
# help(torch.transpose)
out2 = input.transpose(-2, -1)
# The resulting :attr:`out`
# tensor shares its underlying storage with the :attr:`input` tensor, so
# changing the content of one would change the content of the other.
# LOOK AT THE SHAPE...

# print(out1.shape, 'out1')
# print(out2.shape, 'out2')

# How the linear works?

import torch.nn.functional as F

ten1 = torch.tensor([1., 2., 3.])
# print(type(ten1))
# the tensor is int64 type by default, need to make it float by adding a '.' point
lin1 = nn.Linear(3, 1, bias=False)
lin2 = nn.Linear(1, 1, bias=False)
# print(lin1(ten1))
# print(lin2(ten1))  # will error out as the dims don't match
# mat1 and mat2 shapes cannot be multiplied (1x3 and 1x1)

# How softmax works?
s_out = F.softmax(ten1)
print(s_out)

# How embedding works 
vocab_size = 80
embedding_dim = 6

r_in = nn.Embedding(num_embeddings=vocab_size,
                     embedding_dim=embedding_dim)
data_ind = torch.tensor([1, 5, 6, 8])
e_out = r_in(data_ind)
print(e_out)
print(e_out.shape)
"""
tensor([[-0.5251, -2.2980, -1.2629, -0.2184, -0.3236, -1.1250],
        [-2.0372, -0.7762,  1.1529, -1.7969,  0.3080, -0.4566],
        [ 0.3185,  1.7108, -0.4360,  1.5348, -1.1450,  0.2744],
        [-0.0502, -1.8797,  1.3616, -0.0599, -0.4435,  0.0271]],
       grad_fn=<EmbeddingBackward0>)
"""
# Matrix Multiplication
a = torch.tensor([[1, 2], [3, 4], [5, 6]])
b = torch.tensor([[7, 2, 9], [6, 3, 4]])

# print(a @ b)
# print(a.matmul(b))
# print(torch.matmul(a, b))

# playing with shapes

input = torch.rand((3, 8, 10))
B, T, C = input.shape
output = input.view(B * T, C)

# g = d.view(3, -1, -1)  # only one dimension can be inferred, runtime error  

# print(output.shape)
# print(output[:2, :-1])

b = torch.tensor([[1, 72, 68, 74, 67, 57, 72,  0],
                  [74, 73, 58, 71,  0, 56, 54, 75]], dtype=torch.float32)

d = torch.tensor([[1, 72, 68, 74, 67, 57, 72,  0, 73],
                  [74, 73, 58, 71,  0, 56, 54, 75, 58]], dtype=torch.float32)

# print(b.shape)
# print(d.shape)

print(b.view(2 * 8))
print(d.view(2 * 9))

ce = F.cross_entropy(b.view(2*8), d.view(2*9))
print(ce)

# some additional home work on cat and stack
part1 = torch.rand(size=(2, 2))  # 2 row / 2 col matrix
part2 = torch.rand(size=(2, 1))  # 2 row 1 col matrix
part3 = torch.rand(size=(1, 2))  # 1 row 2 col matrix

cat2 = torch.cat((part1, part3))  # will go through 
print(cat2)
# cat1 = torch.cat((part1, part2))  # will inform the expected dimensions are not present

part4 = torch.rand((4, 3))
part5 = torch.randint(high=100, low=10, size=(4, 3))

partstack = torch.stack([part4, part5])

print(partstack.shape)  # 2 tensors of 4 rows and 3 cols 

# partstackx0 = torch.stack(part4, part5)  # will throw syntax-error 

# print(partstackx0.shape)  # 2 tensors of 4 rows and 3 cols

catparts_dim0 = torch.cat((part4, part5), dim=0)
catparts_dim1 = torch.cat((part4, part5), dim=1)

# print(catparts_dim0.shape)  # 8 rows and 3 cols
# key is to expect the how the dimension will change
# print(catparts_dim1.shape)  # 4 rows and 6 cols

part9 = torch.linspace(start=10, end=20, steps=30)  # steps is required args, and ensure it is more than 1, else it will form a Size[1] tensor

print(part9.shape)

# print(part9.view(size=(10,3)))
# print(part9.view(size=(5,6)))
# print(part9.view(size=(3,10)))
# print(part9.view(size=(4,10)))  # RuntimeError: shape '[4, 10]' is invalid for input of size 30

print(part9.view(size=(-1, 3)))
# print(part9.view(size=(-1, 4)))  # will be invalid

class WineDataset(Dataset):
    # just use the init to load the data into the torch objects
    def __init__(self, x_data, y_data):
        self.x = torch.from_numpy(x_data)
        self.y = torch.from_numpy(y_data)
        self.n_samples = x_data.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

wine_ds = WineDataset(x_data=x_data, y_data=y_data)
winedataloader = DataLoader(wine_ds, shuffle=True, batch_size=3)

batch1 = next(iter(winedataloader))
print(batch1)  # List of tensor objects

# need to include the transforms 

# pytorch semantics
# Each tensor has at least one dimension.

# When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

x = torch.empty(5, 7, 8)
y = torch.empty(5, 7, 8)  # tensor of same shapes are broadcastable

x = torch.empty((0, ))
x = torch.empty(2, 2)
# cannot broadcast as one of the dimension is not 1

a = torch.rand((3, 3, 1))
b = torch.rand((3, 1))

# 1st trailing dimension: both have size 1
# 2nd trailing dimension: a size == b size
# 3rd trailing dimension: b dimension doesn't exist 