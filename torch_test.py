#from __future__ import print_function
import torch
import numpy as np
# tensor1 = torch.tensor([5.5, 3])
# print('tensor1: ',tensor1)

# tensor2 = tensor1.new_ones(5, 3, dtype=torch.double)  
# print('tensor2: ',tensor2)

# tensor3 = torch.randn_like(tensor2, dtype=torch.float)
# print('tensor3: ', tensor3)

# tensor4 = tensor2 + tensor3
# print('tensor2 + tensor3 :', tensor4)

# x = torch.randn(4, 4)
# y = x.view(16)
# z = x.view(-1, 8)
# print(x.size(), y.size(), z.size())
# print(x)
# print(y)
# print(z)
#????
# a = torch.ones(5)
# b = a.numpy()
# print(a)
# print(b)
# a.add_(1)
# print(a)
# print(b)
# a = np.ones(5)
# b = torch.from_numpy(a)
# print(a)
# print(b)
# np.add(a, 1, out=a)
# print(a)
# print(b)
x = torch.rand(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print("y:", y)
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print("x.grad:", x.grad)   # d(y)/dx
print("x.requires_grad:", x.requires_grad)
print("(x ** 2).requires_grad:", (x ** 2).requires_grad)
with torch.no_grad():
    print("(x ** 2).with torch no grad:", (x ** 2).requires_grad)